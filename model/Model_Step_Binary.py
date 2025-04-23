# Ring Network Model
# Script written by Cody Moser
# This script runs the Potions Task on Ring Networks
# Specify in BatchRunner or in Viz_Server your parameters and run that file

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from redundancy_functions import prob, joint_prob, redundancy, calculate_redundancy
from hamming import distance, hamming
import networkx as nx
import numpy as np
import pandas as pd
import random
import itertools

#Function to collect the average score of model by dividing the sum of agents scores by num agents
def average_score(model):
    agent_scores = [agent.score for agent in model.schedule.agents]
    N = model.num_agents
    scores = sum(agent_scores)/N
    return scores

#Function to collect the gini of the model based on agent scores
def compute_gini(model):
    agent_wealths = [agent.score for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    if N*sum(x) !=0:
        B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
        return (1 + (1/N) - 2*B)
    else:
        return "NA"

def average_mi(model):
    agent_scores = [agent.mi_random for agent in model.schedule.agents]
    agent_scores = [x for x in agent_scores if isinstance(x, (int, float))]
    if len(agent_scores) > 0:
        scores = sum(agent_scores)/len(agent_scores)
        return scores

def average_red(model):
    agent_scores = [agent.red_random for agent in model.schedule.agents]
    agent_scores = [x for x in agent_scores if isinstance(x, (int, float))]
    if len(agent_scores) > 0:
        scores = sum(agent_scores)/len(agent_scores)
        return scores

def average_cmi(model):
    agent_scores = [agent.cmi_random for agent in model.schedule.agents]
    agent_scores = [x for x in agent_scores if isinstance(x, (int, float))]
    if len(agent_scores) > 0:
        scores = sum(agent_scores)/len(agent_scores)
        return scores

#Create the model
class NetworkModel(Model):
    #Itertools to count which iteration the model is in during batch runs
    id_gen = itertools.count(1)

    #Define initialization parameters for the network; these are also defined in the batch script
    def __init__(self, change_link, prob_diff,
                 cliques,
                 cliqueK,
                 num_agents,
                 prob_edge,
                 theta
                 ):

        #Set crossover to 0 and the step the model is in to 0
        self.crossover = 0
        self.stage = 0
        self.uid = next(self.id_gen)

        #Set up the parameters for the network and agent behavior
        self.weighted = 1
        self.change_link = 1 - (change_link/100)
        self.prob_diff = 1 - (prob_diff/100)
        self.num_agents = num_agents
        self.prob_edge = prob_edge / 100
        self.theta = theta / 100
        self.cliques = cliques #Number of cliques
        self.cliqueK = cliqueK #Size of cliques
        #self.num_agents = self.cliques * self.cliqueK #Determine number of agents


        # G1 = nx.complete_graph(25)
        # G2 = nx.complete_graph(25)
        # mapping = dict(zip(G2,range(25,50)))
        # G2 = nx.relabel_nodes(G2,mapping)
        # self.G = nx.union(G1,G2)
        # rand_G1a = random.choice(list(G1.nodes))
        # rand_G2a = random.choice(list(G2.nodes))
        # self.G.add_edge(rand_G1a, rand_G2a)

        # rand_G1b = random.choice(list(G1.nodes))
        # if rand_G1b == rand_G1a:
        #     rand_G1b = random.choice(list(G1.nodes))
        # rand_G2b = random.choice(list(G2.nodes))
        # if rand_G2b == rand_G2a:
        #     rand_G2b = random.choice(list(G2.nodes))
        # self.G.add_edge(rand_G1b, rand_G2b)
        #self.G = nx.barbell_graph(25,0)

        #Create Ring graph from small world graph, k=2 p=0, n=num_agents
        #self.G = nx.read_edgelist(r'C:\Users\cmose\Documents\git\InformationRedundancy\Networks\Directional\edgdry2014.txt',data=(("weight", float),))        #self.read = pd.read_csv(r'C:\Users\cmose\Documents\git\InformationRedundancy\Networks\Directional\edgwet2013.csv')
        #self.G = nx.from_pandas_edgelist(self.read, source='k1', target='k2', edge_attr="weight") #Specify source and target
        #self.G = nx.erdos_renyi_graph(n=self.num_agents, p=self.prob_edge)
        #self.G = nx.random_regular_graph(n=self.num_agents, d=self.prob_edge)
        #self.G = nx.connected_caveman_graph(l=self.cliques, k=self.cliqueK)
        self.G = nx.connected_watts_strogatz_graph(n=self.num_agents, k=4, p=self.theta)
        #self.G = nx.caveman_graph(l=self.cliques, k=self.cliqueK)
        #self.read = pd.read_csv(r'C:\Users\cmose\Documents\git\CollectiveInfoProcessing\Networks\monkey_info.csv')
        #self.G = nx.from_pandas_edgelist(self.read, source='source', target='target')  # Specify source and target

        # p1 = 0.75 - self.theta
        # q = 0.05 + self.theta
        # p2 = 0.05
        # block_sizes = [25, 25]
        # prob_matrix = [[p1, q], [q, p2]]
        # self.G = nx.stochastic_block_model(block_sizes, prob_matrix)

        self.grid = NetworkGrid(self.G)
        self.num_agents = len(self.G)

        #Set up arrays for network summary statistics
        self.clustering = np.array([])
        self.pathlength = np.array([])
        self.diameter = []
        self.avgclustering = np.array([])
        self.efficiency = np.array([])
        #self.num_agents = nx.number_of_nodes(self.G)

        #Check if graph is connected. If not, do not collect path length
        if nx.is_connected(self.G):
            self.initpathlength = nx.average_shortest_path_length(self.G)
            self.Gdcentrality = nx.degree_centrality(self.G)
            self.Gbcentrality = nx.betweenness_centrality(self.G)
            self.Gccentrality = nx.closeness_centrality(self.G)
        else:
            self.initpathlength = 'NA'
            self.Gdcentrality = 'NA'
            self.Gbcentrality = 'NA'
            self.Gccentrality = 'NA'

        #Set schedule - in our case, the order of agent behavior is random at each step
        self.schedule = RandomActivation(self)

        self.datacollector = DataCollector(
            model_reporters={"NumAgents": lambda m: m.num_agents,
                            "ProbEdge": lambda m: m.prob_edge,
                             "CliqueSize": lambda m: m.cliqueK,
                             "CliqueNum": lambda m: m.cliques,
                             "Theta": lambda m: m.theta,
                             "ProbDiff": lambda m: m.prob_diff,
                             "ChangeLink": lambda m: m.change_link,
                             "Path Length": lambda m: m.pathlength,
                             "Diameter": lambda m: m.diameter,
                             "Efficiency": lambda m: m.efficiency,
                             "Clustering": lambda m: m.avgclustering,
                             "Average Score": average_score,
                             "Gini": compute_gini,
                             "Step": lambda m: m.stage,
                             "Crossover": lambda m: m.crossover,
                             #"IncompleteGraph": lambda m: m.incomplete
                             },
            agent_reporters = {"Agent": lambda a:a.unique_id,
                "Partner": lambda a:a.part,
                "Inventory": lambda a:a.potions,
                "Neighbors": lambda a:a.neighbors_nodes,
                              "NewPot": lambda a:a.newpot,
                               "DiffusedPot": lambda a:a.diffusedpotions,
                "Mutual Info Partner": lambda a:a.mi_partner,
                "Redundancy Partner": lambda a:a.red_partner,
                "Conditional MI Partner": lambda a:a.cmi_partner,
                  "Mutual Info Random": lambda a: a.mi_random,
                  "Redundancy Random": lambda a: a.red_random,
                "Random": lambda a: a.rand_pos,
                  "Conditional MI Random": lambda a: a.cmi_random,
                              "Item Mutual Info Partner": lambda a: a.mi_partner_item,
                              "Item Redundancy Partner": lambda a: a.red_partner_item,
                              "Item Conditional MI Partner": lambda a: a.cmi_partner_item,

                              "Item Mutual Info Random": lambda a: a.mi_random_item,
                              "Item Redundancy Random": lambda a: a.red_random_item,
                              "Item Conditional MI Random": lambda a: a.cmi_random_item,

                              "Mutual Info Targ1": lambda a: a.mi_targ1,
                              "Redundancy Targ1": lambda a: a.red_targ1,
                              "Conditional MI Targ1": lambda a: a.cmi_targ1,

                              "Mutual Info Targ2": lambda a: a.mi_targ2,
                              "Redundancy Targ2": lambda a: a.red_targ2,
                              "Conditional MI Targ2": lambda a: a.cmi_targ2,
        "C1 Partner": lambda a: a.hamc1_partner,
        "C2 Partner": lambda a: a.hamc2_partner,
        "C3 Partner": lambda a: a.hamc3_partner,
        "C1 Random": lambda a: a.hamc1_random,
        "C2 Random": lambda a: a.hamc2_random,
        "C3 Random": lambda a: a.hamc3_random,
                  "Final": lambda a: a.Final,
                  #"Total Inventory": lambda a: a.inventory,
                  "Path Length Random": lambda a: a.random_pl,
                  "Path Length Item": lambda a:a.random_pl_item,
                    "Path Length Target": lambda a:a.random_pl_target,
                  "DegreeCent": lambda a: a.dcentrality,
                  "BetweenCent": lambda a: a.bcentrality,
                  "ClosenessCen": lambda a: a.ccentrality,
                  "MaxScore": lambda a: a.score
                               })
        #Set up collector for export to CSV


        #Allow model to run on its own rather than step by step
        self.running = True

        #Place agents on each node
        nodes_list = list(self.G.nodes())
        nodes_list.sort()
        for i in range(self.num_agents):
             a = Traders(i, self)
             self.schedule.add(a)
             self.grid.place_agent(a, nodes_list[i])

        #list_of_random_nodes = self.random.sample(list(self.G.nodes()), self.num_agents)

        #For each agent in our node list, add the appropriately numbered agent to the appropriate node
        #for i in range(self.num_agents):
            #Find agent i in list of all agents (Traders)
         #   a = Traders(i, self)
           # self.schedule.add(a)
          #  self.grid.place_agent(a, list_of_random_nodes[i])

    def topologyavg(self):
        cluster = nx.average_clustering(self.G)
        self.clustering = np.append(self.clustering, cluster)
        self.avgclustering = np.average(self.clustering)
        self.efficiency = nx.global_efficiency(self.G)
        if nx.is_connected(self.G):
            self.pathlength = nx.average_shortest_path_length(self.G)
            self.diameter = nx.diameter(self.G)

  #Define function for dynamic link changes (agents change neighbors)
    def change_connections(self):
        #For each node in the graph:
        for i in list(self.G.nodes()):
            #Set a random value betwee 0,1
            self.chance = random.uniform(0, 1)
            #If value is larger than change_link, change neighbors
            if self.chance > self.change_link:
                #Find your non-neighbors
                self.notneighbors = list((self.G.nodes) - (self.G.neighbors(i)))
                #Remove yourself from non-neighbor list
                self.notneighbors.remove(i)
                #If non-neighbors is non-zero
                if len(self.notneighbors) > 0:
                    #If number of edges are non-zero
                    if len(self.G.edges(i)) > 0:
                        #Remove an edge
                        self.removedge = (random.choice(list(self.G.edges(i))))
                        self.G.remove_edge(*self.removedge)
                        #Add a new one
                        self.newpartner = random.choice(self.notneighbors)
                        self.G.add_edge(i,self.newpartner)

    def wipe(self):
        for agent in self.schedule.agents:
            agent.partner = []
            agent.diffusedpotions = []
            agent.newpot = []
            agent.mi_partner = []
            agent.red_partner = []
            agent.cmi_partner = []
            agent.mi_random = []
            agent.red_random = []
            agent.cmi_random = []
            agent.mi_partner_item = []
            agent.red_partner_item = []
            agent.cmi_partner_item = []
            agent.mi_random_item = []
            agent.red_random_item = []
            agent.cmi_random_item = []
            agent.mi_targ1 = []
            agent.cmi_targ1 = []
            agent.red_targ1 = []
            agent.mi_targ2 = []
            agent.cmi_targ2 = []
            agent.red_targ2 = []
            agent.random_pl = []
            agent.random_pl_target = []
            agent.random_pl_item = []
            agent.hamc1_partner = []
            agent.hamc2_partner = []
            agent.hamc3_partner = []
            agent.hamc1_random = []
            agent.hamc2_random = []
            agent.hamc3_random = []
            agent.rand_pos = []
            agent.part = []


    #Define steps in the model
    def step(self):
        if self.stage == 0:
#            If graph is not connected, end simulation
            if nx.is_connected(self.G) == False:
                self.incomplete = 1
                self.running = False
            else:
                self.topologyavg()
                self.incomplete = 0
        if self.num_agents > 50 or self.num_agents < 50:
            self.running = False
        self.stage = self.stage + 1
        self.wipe()
        if self.stage % 10 == 0:
            random_agent = self.schedule.agents[self.random.randrange(len(self.schedule.agents))]
            random_agent.newpot = "RAND"
            random_agent.info_calc_constant()
            random_agent.collectdata()
            self.datacollector.collect(self)
        self.schedule.step()
        for agent in self.schedule.agents:
            agent.potions = agent.inventory[:, 0].tolist()
        if nx.is_connected(self.G) and any(agent.success == 1 for agent in self.schedule.agents):
            self.datacollector.collect(self)
        #End simulation when crossover is obtained
        if self.crossover == 1:
            self.running = False

class Traders(Agent):
    #Create agents
    def __init__(self, unique_id, model):
        #Create agents with a unique ID
        super().__init__(unique_id, model)
        #Give each agent the initial inventory consisting of: Potion, Trajectory, Value, and Score
        self.inventory = np.array([
            ['a1','a', 6, 0, 1],
            ['a2','a', 8, 0, 1],
            ['a3','a', 10, 0, 1],
            ['b1','b', 6, 0, 1],
            ['b2','b', 8, 0, 1],
            ['b3','b', 10, 0, 1]
        ])
        self.redscore = [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0]
        #Set initial score to 0
        self.score= 0
        self.diffusedpotions = []
        self.mi_partner = []
        self.red_partner = []
        self.cmi_partner = []
        self.mi_random = []
        self.red_random = []
        self.cmi_random = []
        self.mi_partner_item = []
        self.red_partner_item = []
        self.cmi_partner_item = []
        self.mi_random_item = []
        self.red_random_item = []
        self.cmi_random_item = []
        self.mi_targ1 = []
        self.cmi_targ1 = []
        self.red_targ1 = []
        self.mi_targ2 = []
        self.cmi_targ2 = []
        self.red_targ2 = []
        self.random_pl = []
        self.random_pl_item = []
        self.random_pl_target = []
        self.Final = []
        self.hamc1_partner = []
        self.hamc2_partner = []
        self.hamc3_partner = []
        self.hamc1_random = []
        self.hamc2_random = []
        self.hamc3_random = []
        self.rand_pos = []

        # Calculate centrality measures
        if nx.is_connected(self.model.G):
            self.dcentrality = self.model.Gdcentrality[self.unique_id]
            self.bcentrality = self.model.Gbcentrality[self.unique_id]
            self.ccentrality = self.model.Gccentrality[self.unique_id]
            self.nsize = self.model.G.degree(self.unique_id)
        else:
            self.dcentrality = 'NA'
            self.bcentrality = 'NA'
            self.ccentrality = 'NA'
            self.nsize = 'NA'

    #Function for agents to find neighbors
    def get_neighborhood(self):
        self.neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        self.neighbors = self.model.grid.get_cell_list_contents(self.neighbors_nodes)


    #Function for agents to select a partner
    def pick_partner(self):
        self.partner = np.random.choice(self.neighbors)
        self.part = self.partner.pos
        #print(self.part)
        #print(self.partner.unique_id)
        self.partner_nodes = self.model.grid.get_neighbors(self.partner.pos, include_center=False)
        self.partner_neighbors = self.model.grid.get_cell_list_contents(self.partner_nodes)

    def info_calc_partner(self):
        mutual_items = [max(x, y) for x, y in zip(self.redscore, self.partner.redscore)]
        mutual_items_synergy = mutual_items.copy()
        mutual_items_synergy[self.potpos] = 1
        # create function to add what new potions are possible
        self.mi_partner, self.red_partner, self.cmi_partner = redundancy(self.redscore, self.partner.redscore,
                                                                         mutual_items)
        self.hamc1_partner, self.hamc2_partner, self.hamc3_partner = hamming(self.redscore, self.partner.redscore,
                                                                             mutual_items,
                                                                             mutual_items_synergy, 14)
        new_item = self.potvec
        self.mi_partner_item, self.red_partner_item, self.cmi_partner_item = redundancy(self.redscore,
                                                                                        self.partner.redscore, new_item)

    def info_calc_random(self):
        self.comparison_random = np.random.choice(self.model.schedule.agents)
        self.rand_pos = self.comparison_random.pos
        if self.comparison_random.unique_id == self.unique_id or self.partner.unique_id:
            self.comparison_random = np.random.choice(self.model.schedule.agents)
        mutual_items = [max(x, y) for x, y in zip(self.redscore, self.comparison_random.redscore)]
        mutual_items_synergy = mutual_items.copy()
        mutual_items_synergy[self.potpos] = 1
        if nx.has_path(self.model.G, source=self.unique_id, target=self.comparison_random.unique_id):
            self.random_pl = len(
                nx.shortest_path(self.model.G, source=self.unique_id, target=self.comparison_random.unique_id))
        # create function to add what new potions are possible
        self.mi_random, self.red_random, self.cmi_random = redundancy(self.redscore, self.comparison_random.redscore,
                                                                      mutual_items)
        self.hamc1_random, self.hamc2_random, self.hamc3_random = hamming(self.redscore,
                                                                          self.comparison_random.redscore,
                                                                          mutual_items, mutual_items_synergy, 14)

        new_item = self.potvec
        self.mi_random_item, self.red_random_item, self.cmi_random_item = redundancy(self.redscore,
                                                                                     self.comparison_random.redscore,
                                                                                     new_item)

    def info_calc_constant(self):
        self.comparison_const = np.random.choice(self.model.schedule.agents)
        self.rand_pos = self.comparison_const.pos
        if self.comparison_const.unique_id == self.unique_id:
            self.comparison_const = np.random.choice(self.model.schedule.agents)
        targ1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        targ2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        if nx.has_path(self.model.G, source=self.unique_id, target=self.comparison_const.unique_id):
            self.random_pl_target = len(
                nx.shortest_path(self.model.G, source=self.unique_id, target=self.comparison_const.unique_id))
        self.mi_targ1, self.red_targ1, self.cmi_targ1 = redundancy(self.redscore, self.comparison_const.redscore,
                                                                   targ1)
        self.mi_targ2, self.red_targ2, self.cmi_targ2 = redundancy(self.redscore, self.comparison_const.redscore,
                                                                   targ2)

        mutual_items = [max(x, y) for x, y in zip(self.redscore, self.comparison_const.redscore)]
        if nx.has_path(self.model.G, source=self.unique_id, target=self.comparison_const.unique_id):
            self.random_pl = len(
                nx.shortest_path(self.model.G, source=self.unique_id, target=self.comparison_const.unique_id))
        # create function to add what new potions are possible
        self.mi_random, self.red_random, self.cmi_random = redundancy(self.redscore, self.comparison_const.redscore,
                                                                      mutual_items)


    def trade(self):
        #Pick a random number of ingredients, 1 or 2, to trade with neighbor
        self_ingredients = random.randint(1, 2)
        #Weigh the probability of each item in your inventory for trading with your partner
        self.weights = np.divide(self.inventory[:, 2].astype(float),self.inventory[:, 2].astype(float).sum())
        #Select items to trade with your neighbor of size self_ingredients, and with probability weights
        items_1 = (np.random.choice(self.inventory[:, 0], size=self_ingredients, replace=False, p=self.weights)).tolist()
        #Do the same for your partner
        partner_weights = np.divide(self.partner.inventory[:, 2].astype(float),self.partner.inventory[:, 2].astype(float).sum())
        items_2 = (np.random.choice(self.partner.inventory[:, 0], size=(3 - self_ingredients), replace=False,p=partner_weights)).tolist()
        #Combine your items
        self.item_set = items_1 + items_2

    #Based on the combinations, check if they add up to new innovation tiers
    def combine(self):
        if all(x in self.item_set for x in ['a1', 'a2', 'a3']):
            ingredient_1a = ['1a','a', 48, 48, 1]#15
            #add ingredient 1a to inventory
            if '1a' not in self.inventory:
                self.potpos = 6
                self.newpot = '1a'
                self.potvec = [1,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0]
                self.info_calc_partner()
                self.info_calc_random()
                self.inventory = np.vstack([self.inventory,ingredient_1a])
                self.redscore[6] = 1
                self.partner.newpot = '1a'
                #Loop through your neighbors' inventory, check if they have the item, if not, give them the ingredient
            for x in range (len(self.neighbors)):
                if '1a' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].redscore[6] = 1
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_1a])
                        self.neighbors[x].diffused = 1
                        self.neighbors[x].diffusedpotions.append('1a')
            else:
                matching_row = np.where(self.inventory[:, 0] == '1a')[0]
                for index in matching_row:
                    self.inventory[index, -1] = int(self.inventory[index, -1]) + 1

                #Do the same for your partner
            if '1a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_1a])
                self.partner.redscore[6] = 1
            for x in range (len(self.partner_neighbors)):
                    if '1a' not in self.partner_neighbors[x].inventory:
                        if random.uniform(0, 1) > self.model.prob_diff:
                            self.partner_neighbors[x].redscore[6] = 1
                            self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_1a])
                            self.partner_neighbors[x].diffused = 1
                            self.partner_neighbors[x].diffusedpotions.append('1a')
            else:
                matching_row = np.where(self.partner.inventory[:, 0] == '1a')[0]
                for index in matching_row:
                    self.partner.inventory[index, -1] = int(self.partner.inventory[index, -1]) + 1

        elif all(x in self.item_set for x in ['b1', 'b2', 'b3']):
            ingredient_1b = ['1b','b', 48, 48, 1]
            if '1b' not in self.inventory:
                self.potpos = 16
                self.newpot = '1b'
                self.partner.newpot = '1b'
                self.potvec = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
                self.info_calc_partner()
                self.info_calc_random()
                self.inventory = np.vstack([self.inventory,ingredient_1b])
                self.redscore[16] = 1
            for x in range (len(self.neighbors)):
                if '1b' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].redscore[16] = 1
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_1b])
                        self.neighbors[x].diffused = 1
                        self.neighbors[x].diffusedpotions.append('1b')
            else:
                matching_row = np.where(self.inventory[:, 0] == '1b')[0]
                for index in matching_row:
                    self.inventory[index, -1] = int(self.inventory[index, -1]) + 1

            if '1b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_1b])
                self.partner.redscore[16] = 1
            for x in range (len(self.partner_neighbors)):
                if '1b' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].redscore[16] = 1
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_1b])
                        self.partner_neighbors[x].diffused = 1
                        self.partner_neighbors[x].diffusedpotions.append('1b')
            else:
                matching_row = np.where(self.partner.inventory[:, 0] == '1b')[0]
                for index in matching_row:
                    self.partner.inventory[index, -1] = int(self.partner.inventory[index, -1]) + 1

        elif all(x in self.item_set for x in ['1a', 'a1', 'b2']):
            ingredient_2a = ['2a','a', 109, 109, 1]#20
            if '2a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_2a])
                self.potpos = 7
                self.newpot = '2a'
                self.partner.newpot = '2a'
                self.potvec = [1,1,1,0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,0,0]
                self.info_calc_partner()
                self.info_calc_random()
                self.redscore[7] = 1
            for x in range (len(self.neighbors)):
                if random.uniform(0, 1) > self.model.prob_diff:
                    if '2a' not in self.neighbors[x].inventory:
                        self.neighbors[x].redscore[7] = 1
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_2a])
                        self.neighbors[x].diffused = 1
                        self.neighbors[x].diffusedpotions.append('2a')
            else:
                matching_row = np.where(self.inventory[:, 0] == '2a')[0]
                for index in matching_row:
                    self.inventory[index, -1] = int(self.inventory[index, -1]) + 1

            if '2a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_2a])
                self.partner.redscore[7] = 1
            for x in range (len(self.partner_neighbors)):
                if random.uniform(0, 1) > self.model.prob_diff:
                    if '2a' not in self.partner_neighbors[x].inventory:
                        self.partner_neighbors[x].redscore[7] = 1
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_2a])
                        self.partner_neighbors[x].diffused = 1
                        self.partner_neighbors[x].diffusedpotions.append('2a')
            else:
                matching_row = np.where(self.partner.inventory[:, 0] == '2a')[0]
                for index in matching_row:
                    self.partner.inventory[index, -1] = int(self.partner.inventory[index, -1]) + 1

        elif all(x in self.item_set for x in['1b', 'a2', 'a3']):
            ingredient_2b = ['2b','b', 109, 109, 1]
            if '2b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_2b])
                self.potpos = 17
                self.newpot = '2b'
                self.partner.newpot = '2b'
                self.potvec = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
                self.info_calc_partner()
                self.info_calc_random()
                self.redscore[17] = 1
            for x in range (len(self.neighbors)):
                if '2b' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].redscore[17] = 1
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_2b])
                        self.neighbors[x].diffused = 1
                        self.neighbors[x].diffusedpotions.append('2b')
            else:
                matching_row = np.where(self.inventory[:, 0] == '2b')[0]
                for index in matching_row:
                    self.inventory[index, -1] = int(self.inventory[index, -1]) + 1

            if '2b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_2b])
                self.partner.redscore[17] = 1
            for x in range (len(self.partner_neighbors)):
                if '2b' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].redscore[17] = 1
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_2b])
                        self.partner_neighbors[x].diffused = 1
                        self.partner_neighbors[x].diffusedpotions.append('2b')
            else:
                matching_row = np.where(self.partner.inventory[:, 0] == '2b')[0]
                for index in matching_row:
                    self.partner.inventory[index, -1] = int(self.partner.inventory[index, -1]) + 1

        elif all(x in self.item_set for x in['2a', 'b2', 'a3']):
            ingredient_3a = ['3a','a', 188, 188, 1]#25
            if '3a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_3a])
                self.potpos = 8
                self.newpot = '3a'
                self.partner.newpot = '3a'
                self.potvec = [1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,0,0,0,0]
                self.info_calc_partner()
                self.info_calc_random()
                self.redscore[8] = 1
            for x in range (len(self.neighbors)):
                if '3a' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].redscore[8] = 1
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_3a])
                        self.neighbors[x].diffused = 1
                        self.neighbors[x].diffusedpotions.append('3a')
            else:
                matching_row = np.where(self.inventory[:, 0] == '3a')[0]
                for index in matching_row:
                    self.inventory[index, -1] = int(self.inventory[index, -1]) + 1

            if '3a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_3a])
                self.partner.redscore[8] = 1
            for x in range (len(self.partner_neighbors)):
                if '3a' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].redscore[8] = 1
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_3a])
                        self.partner_neighbors[x].diffused = 1
                        self.partner_neighbors[x].diffusedpotions.append('3a')
            else:
                matching_row = np.where(self.partner.inventory[:, 0] == '3a')[0]
                for index in matching_row:
                    self.partner.inventory[index, -1] = int(self.partner.inventory[index, -1]) + 1

        elif all(x in self.item_set for x in['2b', 'b1', 'a2']):
            ingredient_3b = ['3b','b', 188, 188, 1]
            if '3b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_3b])
                self.potpos = 18
                self.newpot = '3b'
                self.partner.newpot = '3b'
                self.potvec = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                self.info_calc_partner()
                self.info_calc_random()
                self.redscore[18] = 1
            for x in range (len(self.neighbors)):
                if '3b' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].redscore[18] = 1
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_3b])
                        self.neighbors[x].diffused = 1
                        self.neighbors[x].diffusedpotions.append('3b')
            else:
                matching_row = np.where(self.inventory[:, 0] == '2b')[0]
                for index in matching_row:
                    self.inventory[index, -1] = int(self.inventory[index, -1]) + 1

            if '3b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_3b])
                self.partner.redscore[18] = 1
            for x in range (len(self.partner_neighbors)):
                if '3b' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].redscore[18] = 1
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_3b])
                        self.partner_neighbors[x].diffused = 1
                        self.partner_neighbors[x].diffusedpotions.append('3b')
            else:
                matching_row = np.where(self.partner.inventory[:, 0] == '2b')[0]
                for index in matching_row:
                    self.partner.inventory[index, -1] = int(self.partner.inventory[index, -1]) + 1

        elif all(x in self.item_set for x in['3a', '3b', '2a']):
            ingredient_4a = ['4a','a', 358, 358, 1]#30
            if '4a' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_4a])
                self.Final = 1
                self.partner.Final = 1
                self.newpot = '4a'
                self.partner.newpot = '4a'
                self.potpos = 9
                self.potvec = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
                self.info_calc_partner()
                self.info_calc_random()
                self.redscore[9] = 1
            for x in range (len(self.neighbors)):
                if '4a' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].redscore[9] = 1
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_4a])
                        self.neighbors[x].diffused = 1
                        self.neighbors[x].diffusedpotions.append('4a')
            else:
                matching_row = np.where(self.inventory[:, 0] == '4a')[0]
                for index in matching_row:
                    self.inventory[index, -1] = int(self.inventory[index, -1]) + 1

            if '4a' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_4a])
                self.partner.redscore[9] = 1
            for x in range (len(self.partner_neighbors)):
                if '4a' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].redscore[9] = 1
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_4a])
                        self.partner_neighbors[x].diffused = 1
                        self.partner_neighbors[x].diffusedpotions.append('4a')
            else:
                matching_row = np.where(self.partner.inventory[:, 0] == '4a')[0]
                for index in matching_row:
                    self.partner.inventory[index, -1] = int(self.partner.inventory[index, -1]) + 1

        elif all(x in self.item_set for x in['3b', '3a', '2b']):
            ingredient_4b = ['4b','b', 358, 358, 1]
            if '4b' not in self.inventory:
                self.inventory = np.vstack([self.inventory,ingredient_4b])
                self.Final = 1
                self.partner.Final = 1
                self.newpot = '4b'
                self.partner.newpot = '4b'
                self.potpos = 19
                self.potvec = [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1]
                self.info_calc_partner()
                self.info_calc_random()
                self.redscore[19] = 1
            for x in range (len(self.neighbors)):
                if '4b' not in self.neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.neighbors[x].redscore[19] = 1
                        self.neighbors[x].inventory = np.vstack([self.neighbors[x].inventory,ingredient_4b])
                        self.neighbors[x].diffused = 1
                        self.neighbors[x].diffusedpotions.append('4b')
            else:
                matching_row = np.where(self.inventory[:, 0] == '4b')[0]
                for index in matching_row:
                    self.inventory[index, -1] = int(self.inventory[index, -1]) + 1

            if '4b' not in self.partner.inventory:
                self.partner.inventory = np.vstack([self.partner.inventory,ingredient_4b])
                self.partner.redscore[19] = 1
            for x in range (len(self.partner_neighbors)):
                if '4b' not in self.partner_neighbors[x].inventory:
                    if random.uniform(0, 1) > self.model.prob_diff:
                        self.partner_neighbors[x].redscore[19] = 1
                        self.partner_neighbors[x].inventory = np.vstack([self.partner_neighbors[x].inventory, ingredient_4b])
                        self.partner_neighbors[x].diffused = 1
                        self.partner_neighbors[x].diffusedpotions.append('4b')
            else:
                matching_row = np.where(self.partner.inventory[:, 0] == '4b')[0]
                for index in matching_row:
                    self.partner.inventory[index, -1] = int(self.partner.inventory[index, -1]) + 1

    #Get your score
    def collectdata(self):
        #If everything is discovered, score = 716
        if all(x in self.inventory[:,0] for x in ['4a', '4b']):
            self.score = 716
        else:
        #Otherwise, score is equal to maximum item score in your inventory
            self.score = (self.inventory[:, 3].astype(float).max())
        if self.score >= 358:
            self.model.crossover = 1

    def step(self):
        self.initinventory = self.inventory
        self.get_neighborhood()
        if len(self.neighbors) >= 1:
            self.pick_partner()
            self.partnerinventory = self.partner.inventory
            self.trade()
            self.combine()
            self.tradeinventory = self.inventory
            self.partnertradeinventory = self.partner.inventory
            if len(self.initinventory) != len(self.tradeinventory):
                self.success = 1
                self.tradeinventory = self.inventory
                self.partnertradeinventory = self.partner.inventory
                self.inventedpotion = self.inventory[-1,:]
                self.collectdata()
            else:
                self.success = 0
                self.inventedpotion = 0
                self.tradeinventory = 0
                self.partnertradeinventory = 0