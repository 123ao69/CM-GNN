# GRAPH NEURAL NETWORKS FOR SIMULATING COLLECTIVE MOTION

## Introduction

Self-organisation is the process by which local interactions between some individuals in an initially disordered system of collective motion lead to the forming of some form of overall order. In this paper, we construct a lightweight Collective Motion Graph Neural Network (CM-GNN) by simulating the self-organisation process of collective motion. The network
simulates group multilevel leadership through node in-scaling and constructs a migratory environment by mapping raw graph features directly into the classification space. The message passing
mechanism describes the convergence rate of node movement within the environment. Each node is given the notion of ’infinity’ on its field of motion, forcing other nodes close to the node’s
personal space to reduce the convergence rate.CM-GNN achieves state-of-the-art performance and lowest boundary ambiguity on several real-world homogeneous and heterogeneous graph node classification datasets. We theoretically demonstrate that the dynamic system constituted by CM-GNN is asymptotically stable and verify that the system can evolve the population from ordered to disordered by constructing order parameters. Therefore, this dynamical system can avoid the common excessive smoothing problem of GNN under certain conditions.
