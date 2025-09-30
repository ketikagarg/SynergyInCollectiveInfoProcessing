# Synergy In Collective Information Processing

This repository contains an agent-based model of the Potions Task and code to perform information theory analyses on its results.

- Model/Model_Step_Binary.py: Code to run Potions task model using Mesa
- Model/redundancy_functions.py: Functions containing information theory calculations.
- Model/hamming.py: Functions to calculate Hamming distance between potion sets.
- processNetworks.py: The data from the model can be processed using this file to calculate redundancy, complementarity and synergy at each time step, for eahc pair in the network.

The folder "data" contains average values of the information theory and hamming distance measures for the diffeent network structures.

- data/pairwise_smallw_basic.csv contains average values for each iteration and network structure, averaged over all pairs of nodes and each time step.
- data/means_pathlength.csv contains average values for each iteration, network structure and path length (distance) between each pair of nodes, averaged over time.
- data/means_timeseries.csv contains average values for each iteration, network structure and time step, averaged over all pairs in each network.

The file plots.ipynb calls these data files to generate the figures in the corresponding paper.

The folder "networkGIFS" showcases an example of various networks solving the task and the distribution of knowledge. We show an example of fast, medium and slow networks (based on the time taken by the network to complete the task) for each network structure (based on P(edge_rewiring)). The files are named as P(edge_rewiring)\_Success. For example, "0.0_Fast" indicates that this example shows a very efficient iteration of a network with P(rewiring) == 0.0.
