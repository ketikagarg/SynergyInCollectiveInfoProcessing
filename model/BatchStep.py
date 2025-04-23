# Model Batchrunner for Real World Networks
# Script written by Cody Moser
# This script will take your Real World networks and batch run them.
# Specify whether networks are weighted or unweighted
# Not optimized for multiprocessing due to incompatibilities between Windows and Mesa
# See documentation for multiprocessing batchruns: https://mesa.readthedocs.io/en/main/apis/batchrunner.html

#Import Mesa batchrunner
from mesa.batchrunner import BatchRunner
import pandas as pd

#Load network and score keeping functions
from Model_Step_Binary import NetworkModel, average_mi, compute_gini, average_red, average_cmi, average_score

#Select batch parameters
variable_params = {"num_agents": [50],
                   "theta": [0,1,3,5,10,15,25,50,75,100],
                   "prob_edge":[0],
                   "cliques": [0],
                    "cliqueK": [0],
                   "prob_diff": [100],
                   "change_link": [0]}

#Set up batch run parameters
batch_run = BatchRunner(NetworkModel,
                          variable_parameters= variable_params,
                          iterations=1000,
                          max_steps=1000,
                            model_reporters={"NumAgents": lambda m: m.num_agents,
                           #"CliqueSize": lambda m:m.cliqueK,
                            # "CliqueNum": lambda m:m.cliques,
                             "ProbEdge": lambda m: m.prob_edge,
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
                                             #"IncompleteGraph": lambda m: m.incomplete,
                                            # "MI": average_mi,
                                             #"Red": average_red,
                                             #"CMI": average_cmi
                                             },
                            agent_reporters={"Agent": "unique_id",
                            "Partner": "part",
                            "Neighbors": "neighbors_nodes",
                                             "NewPot": "newpot",
                                             "Mutual Info Partner": "mi_partner",
                                         "Redundancy Partner": "red_partner",
                                         "Conditional MI Partner": "cmi_partner",
                            "Random": "rand_pos",
                                         "Mutual Info Random": "mi_random",
                                         "Redundancy Random": "red_random",
                                         "Conditional MI Random": "cmi_random",
                                             "Item Mutual Info Partner": "mi_partner_item",
                                             "Item Redundancy Partner": "red_partner_item",
                                             "Item Conditiional MI Partner": "cmi_partner_item",
                                             "Item Mutual Info Random": "mi_random_item",
                                             "Item Redundancy Random": "red_random_item",
                                             "Item Conditiional MI Random": "cmi_random_item",

                                             "Mutual Info Targ1": "mi_targ1",
                                             "Redundancy Targ1": "red_targ1",
                                             "Conditional MI Targ1": "cmi_targ1",

                                             "Mutual Info Targ 2": "mi_targ2",
                                             "Redundancy Targ 2": "red_targ2",
                                             "Conditional MI Targ2": "cmi_targ2",

                                             "C1 Partner": "hamc1_partner",
                                             "C2 Partner": "hamc2_partner",
                                             "C3 Partner": "hamc3_partner",
                                             "C1 Random": "hamc1_random",
                                             "C2 Random": "hamc2_random",
                                             "C3 Random": "hamc3_random",
                                             "Path Length": "random_pl",
                                             "Final": "Final",
                            #"Total Inventory": "inventory",
                            "DegreeCent": "dcentrality",
                            "BetweenCent": "bcentrality",
                            "ClosenessCen": "ccentrality",
                            "MaxScore": "score"
                                             },
                            display_progress=True)

#Run batches and collect data
batch_run.run_all()
# modelvars_df = batch_run.get_model_vars_dataframe()
# modelvars_df.to_csv(r"K:\git\CollectiveInfoProcessing\modeltest2.csv")
modelvars_df = batch_run.get_collector_model()
modelvars_list = list(modelvars_df.values())
for i in range(len(modelvars_list)):
    modelvars_list[i]['Iteration'] = i + 1
pd.concat(modelvars_list).to_csv(r"model_smallw.csv") #do 5, 25, 50, 75
    #agentvars_list[i] = agentvars_list[i].tail((agentvars_list[i].index[-1][1])+ 1)
agentvars_df = batch_run.get_collector_agents()
agentvars_list = list(agentvars_df.values())
for i in range(len(agentvars_list)):
    agentvars_list[i]['Iteration'] = i + 1
    condition = (agentvars_list[i][agentvars_list[i].columns[4]].apply(lambda x: x == [])) & (agentvars_list[i][agentvars_list[i].columns[5]].apply(lambda x: x == []))
    #print(agentvars_list[i].columns[4])
    #print(agentvars_list[i].columns[5])
    agentvars_list[i] = agentvars_list[i].loc[~condition]
    #agentvars_list[i] = agentvars_list[i].tail((agentvars_list[i].index[-1][1])+ 1)
pd.concat(agentvars_list).to_csv(r"agent_smallw.csv")