# Benchmark instances for the quadratic knapsack problem

This repository contains the code to generate benchmark instances for the quadratic knapsack problem. 

# Instructions on how to generate the instances

1) Clone the repository
2) Create a virtual environment and install the required packages (see requirements.txt)
3) On Linux OS: run "sudo apt install unrar" in terminal to install the unrar package 
4) Run the script `generate_instances_from_raw_data.py` 
5) Run the script `generate_synthetic_instances.py`

# Overview of instances

The folder collections contains seven collections of instances.

* Standard-QKP: well-known standard qkp instances from https://cedric.cnam.fr/~soutif/QKP/QKP.html
* QKPGroupII: well-known larger qkp instances from https://leria-info.univ-angers.fr/~jinkao.hao/QKP.html
* QKPGroupIII: well-known larger qkp instances from https://leria-info.univ-angers.fr/~jinkao.hao/QKP.html
* Large-QKP: new instances generated with similar procedure as the standard qkp instances
* Dispersion-QKP: instances adapted from dispersion problem instances
* TeamFormation-QKP-1: instances adapted from team formation problem instances with low density
* TeamFormation-QKP-2: instances adapted from team formation problem instances with moderate density

# Description of format

Each txt file contains the information of one graph and one or multiple budgets:

First line: n m type
* n: number of nodes
* m: number of edges
* type: data type of edge weights (int or float)

lines 2-m+1: i j u_ij
* i: first node of edge 
* j: second node of edge 
* u_ij: weight of edge [i, j]

line m+2: q_0, q_1, ...
* q_j: weight of node j

line m+3: B_0 B_1 ...
* B_0, B_1, ...: budgets

Here is an example of a file that describes a graph with 5 nodes and 15 edges. 
The last line lists two budgets (25 and 75). Note that this graph contains 
self-loops (singleton utilities u_ii > 0).    

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
## Reference

Please cite the following paper if you use this repository.

**Hochbaum, D. S., Baumann, P., Goldschmidt O., Zhang Y.** (2024): A Fast and Effective Breakpoints Algorithm for the Quadratic Knapsack Problem. under review.

Bibtex:
```
@article{hochbaum2024fast,
	author={Hochbaum, Dorit S., Baumann, Philipp, Goldschmidt Olivier and Zhang Yiqing},
	title = {A Fast and Effective Breakpoints Algorithm for the Quadratic Knapsack Problem},
	year={2024},
	journal = {under review},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

