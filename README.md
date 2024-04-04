# Benchmark instances for the quadratic knapsack problem

This repository contains the code to generate benchmark instances for the quadratic knapsack problem. 

# Instructions on how to generate the instances

1) Clone the repository
2) Create a virtual environment and install the required packages (see requirements.txt)
3) Run the script `generate_instances_from_raw_data.py` 
4) Run the script `generate_synthetic_instances.py`

# Overview of instances

The folder collections contains five collections of instances.

* Standard-QKP: well-known standard qkp instances from https://cedric.cnam.fr/~soutif/QKP/QKP.html
* New-QKP: new instances generated with the same procedure as the standard qkp instances
* Dispersion-QKP: instances adapted from dispersion problem instances
* TeamFormation-QKP-1: instances adapted from team formation problem instances with low density
* TeamFormation-QKP-2: instances adapted from team formation problem instances with high density

# Description of format

Each txt file contain the information of one graph and one or multiple budgets:

First line: n m type
* n: number of nodes
* m: number of edges
* type: data type of edge weights (int or float)

lines 2-m+1: i j u_ij
* i: first node of edge 
* j: second node of edge 
* u_ij: weights of edge

line m+2: q_j
* q_j: weight of node j

line m+3: B_1 B_2 ...
* B_1, B_2, ...: budgets

Here is an example of a file that describes a graph with 5 nodes, 15 edges. The last line lists two budgets (25 and 75).

```
5 15 int
0 0 35
0 1 18
0 2 83
0 3 19
0 4 29
1 1 2
1 2 12
1 3 8
1 4 1
2 2 100
2 3 26
2 4 13
3 3 36
3 4 96
4 4 34
40 5 4 44 8 
25 75 
```


