import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import multiprocessing as mp  # For parallel processing
from multiprocessing import Pool
from tqdm import tqdm  # For progress bars
import gc  # Garbage collection
import os
import psutil  # For checking available memory


# Function to ensure the output directory exists
def ensure_output_directory():
    output_dir = "output_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


# Function to check available memory
def get_available_memory():
    return psutil.virtual_memory().available / (1024**2)  # Convert to MB


# Function to save DataFrame in chunks
def save_dataframe_in_chunks(df, filename_prefix, chunk_size=500000):
    output_dir = ensure_output_directory()
    file_index = 1

    # Save in chunks if too large
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        filename = os.path.join(output_dir, f"{filename_prefix}_part{file_index}.csv")
        chunk.to_csv(filename, index=False)
        print(f"Saved chunk {file_index}: {filename}")
        file_index += 1


# Custom Hamming Distance Functions
def distance(p):
    """Calculate the number of positions that are not 1."""
    return sum(1 for bit in p if bit != 1)


def hamming(x, y, xy, num_potions):
    """
    Compute custom Hamming-based metrics.

    Parameters:
    - x: Binary inventory of agent1
    - y: Binary inventory of agent2
    - xy: Element-wise OR of x and y (Union excluding new potions)
    - num_potions: Total number of potions considered

    Returns:
    - c1 : the complementarity of x and y
    """
    c1 = (distance(x) + distance(y) - distance(xy)) / num_potions
    return c1


# Function to calculate entropy from a probability distribution
def entropy(probs):
    return -np.sum([p * np.log2(p) for p in probs if p > 0])


# Function to compute joint entropy for any number of agents' inventories
def joint_entropy(*agents_inventories):
    joint_inventories = list(zip(*agents_inventories))
    counts = Counter(joint_inventories)
    total = len(joint_inventories)
    probs = [count / total for count in counts.values()]
    return entropy(probs)


# Pairwise Mutual Information (MI)
def pairwise_mi(agent1, agent2):
    H_x = entropy([count / len(agent1) for count in Counter(agent1).values()])
    H_y = entropy([count / len(agent2) for count in Counter(agent2).values()])
    H_xy = joint_entropy(agent1, agent2)
    return H_x + H_y - H_xy


# Pairwise Conditional Mutual Information (CMI)
def pairwise_cmi(agent1, agent2, pooled_inventory):
    H_xyz = joint_entropy(agent1, agent2, pooled_inventory)
    H_xz = joint_entropy(agent1, pooled_inventory)
    H_yz = joint_entropy(agent2, pooled_inventory)
    H_z = entropy(
        [count / len(pooled_inventory) for count in Counter(pooled_inventory).values()]
    )
    return H_xz + H_yz - H_xyz - H_z


# Group-level redundancy and synergy (raw)
def group_redundancy_and_synergy(inventories, total_potions=14):
    inventories_sets = [set(agent) for agent in inventories]
    union = set().union(*inventories_sets)
    intersection = (
        set.intersection(*inventories_sets) if len(inventories_sets) > 1 else union
    )
    redundancy = len(intersection) / total_potions  # Scaled redundancy
    synergy = (len(union) - len(intersection)) / total_potions  # Scaled synergy
    return redundancy, synergy


# Binary Conversion with Consistent Potion Ordering
def potions_to_binary(agent_inventory, all_potions_sorted):
    """
    Convert an agent's inventory to a binary vector based on all_potions_sorted.

    Parameters:
    - agent_inventory: List of potions the agent has
    - all_potions_sorted: Sorted list of all possible potions

    Returns:
    - Binary list indicating presence (1) or absence (0) of each potion
    """
    inventory_set = set(agent_inventory)
    return [1 if potion in inventory_set else 0 for potion in all_potions_sorted]


# Function to create a network from the data
def make_network(data, plot=False):
    """Create and optionally plot a network graph from the data."""
    G = nx.Graph()
    for index, row in data.iterrows():
        agent = row["AgentID"]
        try:
            neighbors = (
                eval(row["Neighbors"])
                if isinstance(row["Neighbors"], str)
                else row["Neighbors"]
            )
        except Exception as e:
            print(f"Error parsing neighbors for AgentID {agent}: {e}")
            neighbors = []
        G.add_node(agent)
        for neighbor in neighbors:
            G.add_edge(agent, neighbor)
    if plot:
        nx.draw(G, with_labels=True, node_size=500, font_size=10)
        plt.show()
    return G


# Function to compute path lengths for a static network
def compute_path_lengths(G):
    """Compute all pairs shortest path lengths in the graph."""
    return dict(nx.all_pairs_shortest_path_length(G))


# Function to compute metrics with backtracking and Hamming distance
def compute_metrics_with_backtracking(df, all_potions, G, path_lengths):
    """
    Compute MI, CMI, and Hamming-based metrics for agents' inventories over time.

    Parameters:
    - df: DataFrame containing the inventory data
    - all_potions: List of all possible potions
    - G: Network graph
    - path_lengths: Precomputed path lengths in the network

    Returns:
    - global_results_df: DataFrame with global redundancy and synergy metrics
    - pairwise_results_df: DataFrame with pairwise MI, CMI, and Hamming-based metrics
    """
    timesteps = sorted(df["Step"].unique())
    results = []
    pairwise_results = []

    # Ensure consistent potion ordering by sorting all_potions
    all_potions_sorted = sorted(all_potions)
    num_potions = len(all_potions_sorted)

    # Initialize all agents with the same starting inventory (binary)
    all_agent_ids = df["AgentID"].unique()
    all_binary_inventories = {
        agent: potions_to_binary(
            ["a1", "a2", "a3", "b1", "b2", "b3"], all_potions_sorted
        )
        for agent in all_agent_ids
    }

    # Helper function to backtrack and find the most recent inventory for an agent
    def find_last_inventory(agent, current_step):
        for prev_step in range(current_step - 1, -1, -1):  # Check earlier timesteps
            prev_data = df[(df["Step"] == prev_step) & (df["AgentID"] == agent)]
            if not prev_data.empty:
                # Return the inventory if found
                inventory = prev_data.iloc[0]["Inventory"]
                try:
                    inventory_list = (
                        eval(inventory) if isinstance(inventory, str) else inventory
                    )
                except Exception as e:
                    print(
                        f"Error parsing inventory for AgentID {agent} at Step {prev_step}: {e}"
                    )
                    inventory_list = ["a1", "a2", "a3", "b1", "b2", "b3"]
                return potions_to_binary(
                    inventory_list,
                    all_potions_sorted,
                )
        # If no inventory is found, return the default starting inventory
        return potions_to_binary(
            ["a1", "a2", "a3", "b1", "b2", "b3"], all_potions_sorted
        )

    for t in timesteps:
        # Get data for the current timestep
        timestep_data = df[df["Step"] == t]
        agent_ids = timestep_data["AgentID"].tolist()
        inventories = timestep_data["Inventory"].tolist()

        # Update inventories for agents present at this timestep
        for agent, inventory in zip(agent_ids, inventories):
            try:
                inventory_list = (
                    eval(inventory) if isinstance(inventory, str) else inventory
                )
                all_binary_inventories[agent] = potions_to_binary(
                    inventory_list,
                    all_potions_sorted,
                )
            except Exception as e:
                print(f"Error parsing inventory for AgentID {agent} at Step {t}: {e}")
                all_binary_inventories[agent] = potions_to_binary(
                    ["a1", "a2", "a3", "b1", "b2", "b3"], all_potions_sorted
                )

        # Backtrack for missing agents and update their inventories
        for agent in all_agent_ids:
            if agent not in agent_ids:
                # Backtrack to find the most recent inventory
                last_inventory_binary = find_last_inventory(agent, t)
                all_binary_inventories[agent] = last_inventory_binary

        # Get binary inventories for all agents
        binary_inventories = {
            agent: all_binary_inventories[agent] for agent in all_agent_ids
        }

        # Group-level redundancy and synergy using binary inventories
        inventories_sets = [
            set(
                potion
                for potion, bit in zip(
                    all_potions_sorted, all_binary_inventories[agent]
                )
                if bit
            )
            for agent in all_agent_ids
        ]

        union = set().union(*inventories_sets)
        intersection = (
            set.intersection(*inventories_sets) if len(inventories_sets) > 1 else union
        )
        redundancy_val = len(intersection) / num_potions  # Scaled redundancy
        synergy_val = (len(union) - len(intersection)) / num_potions  # Scaled synergy

        # Pooled inventory (union of all agents)
        pooled_inventory_binary = [
            1 if any(agent[i] == 1 for agent in binary_inventories.values()) else 0
            for i in range(num_potions)
        ]

        # Compute pairwise MI, CMI, and Hamming-based metrics
        agents = list(binary_inventories.keys())
        mi_values_binary, cmi_values_binary, redundancy_values_binary = [], [], []
        hamming_metrics = []

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1 = agents[i]
                agent2 = agents[j]
                x = binary_inventories[agent1]
                y = binary_inventories[agent2]

                # Pairwise metrics (Binary MI and CMI)
                mi_bin = pairwise_mi(x, y)
                cmi_bin = pairwise_cmi(x, y, pooled_inventory_binary)
                red_bin = mi_bin - cmi_bin

                mi_values_binary.append(mi_bin)
                cmi_values_binary.append(cmi_bin)
                redundancy_values_binary.append(red_bin)

                # Hamming-based metrics
                # Element-wise OR for xy (union excluding new potions)
                xy = [max(x_bit, y_bit) for x_bit, y_bit in zip(x, y)]

                # Compute Hamming-based metrics
                c1 = hamming(x, y, xy, num_potions)

                # Store Hamming metrics
                hamming_metrics.append(c1)

                # Get path length between the two agents
                path_length = path_lengths.get(agent1, {}).get(agent2, np.inf)

                # Save pairwise results
                pairwise_results.append(
                    {
                        "Step": t,
                        "Agent1": agent1,
                        "Agent2": agent2,
                        "Pairwise_MI_Binary": mi_bin,
                        "Pairwise_CMI_Binary": cmi_bin,
                        "Redundancy_Binary": red_bin,
                        "Hamming_c1": c1,
                        "Path_Length": path_length,
                    }
                )

        # Compute mean metrics
        mean_mi_binary = np.mean(mi_values_binary) if mi_values_binary else 0
        mean_cmi_binary = np.mean(cmi_values_binary) if cmi_values_binary else 0
        mean_red_binary = (
            np.mean(redundancy_values_binary) if redundancy_values_binary else 0
        )

        # Store global results
        results.append(
            {
                "Step": t,
                "Global_Redundancy": redundancy_val,
                "Global_Synergy": synergy_val,
                "Mean_MI_Binary": mean_mi_binary,
                "Mean_CMI_Binary": mean_cmi_binary,
                "Mean_Redundancy_Binary": mean_red_binary,
            }
        )

    # Create DataFrames from the results
    global_results_df = pd.DataFrame(results)
    pairwise_results_df = pd.DataFrame(pairwise_results)
    return global_results_df, pairwise_results_df


# Function to process a single iteration
def process_single_iteration(args):
    """
    Process a single iteration: compute network metrics and pairwise metrics.

    Parameters:
    - args: Tuple containing (iteration, df, all_potions_sorted, default_inventory)

    Returns:
    - d: DataFrame with global metrics for this iteration.
    - pairwise: DataFrame with pairwise metrics for this iteration.
    """
    iteration, df, all_potions_sorted, default_inventory = args

    # Filter data for the current iteration
    data = df[df["Iteration"] == iteration]

    # Create the network
    G = make_network(data)

    # Compute path lengths
    path_lengths = compute_path_lengths(G)

    # Compute metrics
    d, pairwise = compute_metrics_with_backtracking(
        data, all_potions_sorted, G, path_lengths
    )

    # Add additional columns
    d["MaxStep"] = data["Step"].max()
    pairwise["MaxStep"] = data["Step"].max()

    # Assuming 'C1 Partner' exists in your DataFrame
    if "C1 Partner" in data.columns:
        d["MaxC1Partner"] = data["C1 Partner"].max()
        pairwise["MaxC1Partner"] = data["C1 Partner"].max()

    # Add iteration number
    d["Iteration"] = iteration
    pairwise["Iteration"] = iteration

    # Add ProbEdge (assuming it's consistent within the iteration)
    if "ProbEdge" in data.columns and not data["ProbEdge"].isnull().all():
        d["ProbEdge"] = data["ProbEdge"].iloc[0]
        pairwise["ProbEdge"] = data["ProbEdge"].iloc[0]
    else:
        d["ProbEdge"] = np.nan
        pairwise["ProbEdge"] = np.nan

    ##repeat for Theta
    if "Theta" in data.columns and not data["Theta"].isnull().all():
        d["Theta"] = data["Theta"].iloc[0]
        pairwise["Theta"] = data["Theta"].iloc[0]
    else:
        d["Theta"] = np.nan
        pairwise["Theta"] = np.nan

    return d, pairwise


def main():
    input_ = input("Which network to use?: ")
    firstiteration = int(input("Enter the iteration number to begin with:"))

    BASE_FOLDER = "./processedNW/"
    merged_csv_path = os.path.join(BASE_FOLDER, f"{input_}_merged.csv")

    if not os.path.isfile(merged_csv_path):
        print(f"Error: Merged CSV '{merged_csv_path}' does not exist.")
        return

    try:
        df_iter = pd.read_csv(merged_csv_path, chunksize=500000)  # Read in chunks
    except Exception as e:
        print(f"Error reading merged CSV '{merged_csv_path}': {e}")
        return

    # Define potions
    all_potions = [
        "a1",
        "a2",
        "a3",
        "b1",
        "b2",
        "b3",
        "1a",
        "2a",
        "3a",
        "1b",
        "2b",
        "3b",
        "4a",
        "4b",
    ]
    all_potions_sorted = sorted(all_potions)
    default_inventory = ["a1", "a2", "a3", "b1", "b2", "b3"]

    # Read first chunk to extract iteration numbers
    print("Extracting iterations from the dataset...")
    iterations = []
    for chunk in df_iter:
        iterations.extend(chunk["Iteration"].unique())
        if len(iterations) > 10000:  # Stop early if too many iterations exist
            break

    iterations = sorted(set(iterations))
    iterations = [i for i in iterations if i >= firstiteration]

    if not iterations:
        print("No iterations found after firstiteration.")
        return

    print(f"Total iterations to process: {len(iterations)}")

    batch_size = 100  # Process 100 iterations at a time

    for batch_start in range(0, len(iterations), batch_size):
        batch_end = min(batch_start + batch_size, len(iterations))
        batch_iterations = iterations[batch_start:batch_end]

        print(
            f"Processing iterations {batch_iterations[0]} to {batch_iterations[-1]}..."
        )

        df_iter = pd.read_csv(
            merged_csv_path, chunksize=500000
        )  # Reload to filter each batch
        df_batch = pd.concat(
            [chunk[chunk["Iteration"].isin(batch_iterations)] for chunk in df_iter]
        )

        args_list = [
            (iteration, df_batch, all_potions_sorted, default_inventory)
            for iteration in batch_iterations
        ]
        global_results, pairwise_results = [], []

        with Pool(processes=7) as pool:
            for d, pairwise in tqdm(
                pool.imap(process_single_iteration, args_list), total=len(args_list)
            ):
                global_results.append(d)
                pairwise_results.append(pairwise)

        if global_results:
            global_results_df = pd.concat(global_results, ignore_index=True)
            save_dataframe_in_chunks(
                global_results_df,
                f"globaldf_{input_}_from_{batch_iterations[0]}_to_{batch_iterations[-1]}",
            )

        if pairwise_results:
            pairwise_results_df = pd.concat(pairwise_results, ignore_index=True)
            available_memory = get_available_memory()
            print(f"Available Memory: {available_memory:.2f} MB")

            if len(pairwise_results_df) > 1_000_000 or available_memory < 1000:
                print("Pairwise DataFrame is too large, saving in chunks...")
                save_dataframe_in_chunks(
                    pairwise_results_df,
                    f"pairwise_{input_}_from_{batch_iterations[0]}_to_{batch_iterations[-1]}",
                )
            else:
                output_dir = ensure_output_directory()
                pairwise_results_df.to_csv(
                    os.path.join(
                        output_dir,
                        f"pairwise_{input_}_from_{batch_iterations[0]}_to_{batch_iterations[-1]}.csv",
                    ),
                    index=False,
                )
                print(
                    f"Pairwise metrics saved to '{output_dir}/pairwise_{input_}_from_{batch_iterations[0]}_to_{batch_iterations[-1]}.csv'."
                )

        # Clear memory
        del df_batch, global_results, pairwise_results
        gc.collect()
        print(
            f"Finished processing iterations {batch_iterations[0]} to {batch_iterations[-1]}.\n"
        )

    print("Processing complete. Results saved to output_data/ folder.")


if __name__ == "__main__":
    main()
