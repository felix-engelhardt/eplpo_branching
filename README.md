# Branch and Bound for Capacitated Facility Location under Strict Preferences (CFLP-SP) [![DOI](https://zenodo.org/badge/1089589145.svg)](https://doi.org/10.5281/zenodo.17524470)

## Description

This repository contains code relating to the paper "A Combinatorial Branch-and-Bound Algorithm for the Capacitated Facility Location Problem under Strict Customer Preferences" by [Christina Büsing](https://combi.rwth-aachen.de/en/team/christina-buesing), [Felix Engelhardt](https://combi.rwth-aachen.de/en/team/felix-engelhardt)  and [Sophia Wrede](https://combi.rwth-aachen.de/de/team/sophia-wrede), all from the Combinatorial Optimization group at RWTH Aachen.

The implementation contains a combinatorial branch-and-bound algorithm, as well as an integer programming formulation of the CFLP-SP. Both can be evaluated under different parameter configurations. At the time of this writing, the formulation + the given preprocessing are the best ILP approach known for the CFLP-SP.

## Using the repository

Python is required to run the code, we use the os, sys, time, json, math, random, helper, copy and psutil packages, as well as numpy and networkx. For executing the integer linear program, you also have to have Gurobi + gurobipy installed. Check their website for current installation guidelines. 

Then, clone the repository and use a terminal to navigate to the branching folder. You will find five files

- *branching.py* This is the main algorithm. It contains a tree and a node class each with multiple methods. These are called repeatedly during the algorithm's execution. The other functions are mostly used to provide preprocessing for the ILP solver.
- *helper.py* This contains some useful routines that can check whether solutions are feasible and whether they match previous solutions on record.
- *initialisation.py* has three functions that deal with reading in data from different formats. 
- *ip.py* Contains the ILP formulation. This is excecutable as a main file.
- *main.py* Contains the B&B algorithm. This is excecutable as a main file.

Thus, to generate results you need to either execute *ip.py* or *main.py*. Either logs the algorithm's progress in the terminal and produces a results file that is automatically saved in the *results* folder. That folder already contains all results from our computaitonal study, which can be used for verification.

To execute *ip.py* or *main.py*  you will have to enter something like this

`python3 main.py 0 1 300 X 20_20 "Lagrangian"`

The meaning of all possible parameters is given in comments in either file. 

Here, the first parameter, 0, indicates th instance index, so this is instance 0 being read in, the range is generally 0-19. The second parameter is either 0 for closest assignment or 1 for perturbed closest assignment. The third parameter indcates the typ/size of instances, this means here we read in an instance of size 300, for alternatives see the respective data files.
The fourth parameter allows to indicate a letter code for the second instance type outlined below, here it is just left aat the default X, since we are not using this type of instance. 
Then, the fifth parameter allows slicing of instances, i.e. here we only consider a 20x20 subset of the 300x300 instance. This is monstly a feature for debugging and experimenting - often it is helpful to be able to test arbitrary instance sizes an the fly. 
Finally, the sixth paramter tells the solver whether it has to use Lagrangian relaxation (as above) or not, if anything else is inputted. For the ip, the parameters are the same with the main difference being that you indicate whether or not to use preprocessing with the final argument instead, e.g.

`python ip.py 7 0 300 X 50_50 "preprocessing"`

## Data

A total of 4·20 + 2·12 = 104 instances from two instance families were considered in the computational study, theses are also part of the repository. The data provided here is based on previous research. Their size and origin are summarised below.

| Instance-families' size (# Locations, # Customers) | # Instances per family | Source |
|----------------------------------------------------|------------------------|---------|
| (300,300), (500,500), (700,700), (1000,1000)      | 20                     | [1] |
| (50,75), (75,100)                                 | 12                     | Taken from [2], modified according to [3] |

We thank the authors for freely providing this data. Please cite their works if you reuse it:

[1] P. Avella and M. Boccia. A cutting plane algorithm for the capacitated facility location problem. Computational Optimization and Applications, 43:39–65, 2009. doi: 10.1007/s10589-007-9125-x. URL: [https://doi.org/10.1007/s10589-007-9125-x](https://doi.org/10.1007/s10589-007-9125-x).

[2] J. Beasley. An algorithm for solving large capacitated warehouse location problems. European Journal of
Operational Research, 33(3):314–325, 1988. doi: 10.1016/0377-2217(88)90175-0. URL: [https://www.sciencedirect.com/science/article/pii/0377221788901750](https://www.sciencedirect.com/science/article/pii/0377221788901750)

[3] L. Cánovas, S. García, M. Labbé, and A. Marín. A strengthened formulation for the simple plant location problem with order. Operations Research Letters, 35(2):141–150, 2007. ISSN 0167-6377. doi: https://doi.org/10.1016/j.orl.2006.01.012. URL [https://www.sciencedirect.com/science/article/pii/S0167637706000320](https://www.sciencedirect.com/science/article/pii/S0167637706000320).

## Questions
Feel free to write to the corresponding author Ms. Wrede [Email: wrede@combi.rwth-aachen.de](mailto:wrede@combi.rwth-aachen.de) if you have any questions.

## Acknowledgments
We thank our colleague Timo Gersing for many fruitful discussions and the occasional inspirational chocolate bar. We also thank the organising team of CO@Work 2024, i.e. Timo Berthold, Ralf Borndörfer, Ambros Gleixner, Thorsten Koch, and Milena Petkovic. Their workshop motivated and inspired us to do this research.

## License
All code is under a GNU Affero General Public License v3.0 only.
