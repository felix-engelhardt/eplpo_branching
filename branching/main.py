############
### main ###
############

import os, sys, time, json
from initialisation import read_instance
from branching import BnB_node,BnB_tree,compute_implied_demands # type: ignore
from math import inf
from random import seed
from helper import check, validate
import psutil

seed(0)

dirname = os.path.dirname(__file__)

def main(i:int,insType,insTypeLetter,setting,pref_,do_Lagrangian,ties):
    
    start_time = time.time()
    node_selection_time, implied_demands_time, bounds_time, othertime, variable_selection_time, greedy_time, memory_usage = 0, 0, 0, 0, 0, 0, 0

    # 1. Define important sets and parameters
    capacities, demands, fixedCost, assignment_costs, n_cust, n_facs, pref_ordering = read_instance(i,insType,insTypeLetter,setting,pref_)
    openFacs,cannot_assign = {j:j for j in range(n_facs)},{j:set() for j in range(n_facs)}
    position_of_least_preferred_facility = {i:len(pref_ordering[i])-1 for i in range(n_cust)}    # tells us the position of the least preferred facility in pref_ordering of each customer

    # 2. Set up the root node  
    BnB_node.capacities = capacities # These values are global for all nodes
    BnB_node.demands = demands
    BnB_node.pref_ordering = pref_ordering
    BnB_node.n_cust = len(demands)
    BnB_node.assignment_costs = assignment_costs
    BnB_node.fixedCost = fixedCost
    BnB_node.totaldemand = sum(demands)
    BnB_node.primalBound = sum(fixedCost[j] + sum(assignment_costs[i][j] for i in range(n_cust)) for j in openFacs)

    # More initialisation of the root node
    former_branching_candidate = -1

    # Initialise Lagrangian multipliers
    former_LR_lambda = {i:min([BnB_node.assignment_costs[i][j] for j in openFacs]) for i in range(n_cust)} 
    former_LR_mu = {i:{j:0 for j in openFacs} for i in range(n_cust)}   
    former_LR_rho = {j:0 for j in openFacs} 

    # Now write the actual root node
    rootnode = BnB_node(openFacs,position_of_least_preferred_facility,cannot_assign,former_branching_candidate,former_LR_lambda,former_LR_mu,former_LR_rho,[0,0]) # Facilities have to be set for each node        
    temporary_pref_ordering = rootnode.make_feasible()

    try: # Initialise temporary dictionary local_pref_ordering
        rootnode.local_position_of_least_preferred_facility, rootnode.local_cannot_assign = compute_implied_demands(demands=rootnode.demands,capacities=rootnode.capacities,temporary_pref_ordering=temporary_pref_ordering,position_of_least_preferred_facility=rootnode.local_position_of_least_preferred_facility,cannot_assign=rootnode.local_cannot_assign, facilities=rootnode.facilities,required_facilities=rootnode.required_facilities) 
    except: print("ERROR: Root node infeasible.")
    temporary_pref_ordering = {i:[facility for facility in BnB_node.pref_ordering[i][:1+rootnode.local_position_of_least_preferred_facility[i]]
                    if facility in rootnode.facilities] for i in range(n_cust)}
    process = psutil.Process(os.getpid())   
    
    if rootnode.status == "Infeasible":
        print("UPDATE/WARNING: Root node infeasible")
        name = "_".join([str(i) for i in sys.argv[1:]])
        results = {
            "input parameters": sys.argv[1:],
            "instance": {
                "facilities": n_facs,
                "customers": BnB_node.n_cust,
            },
            "times": {
                "total_time in s": round(time.time() - start_time, 2),
                "initialisation_time in s": round(time.time() - start_time,2),
                "node_selection_time in s": round(node_selection_time,2),
                "implied_demands_time in s": round(implied_demands_time,2),
                "bounds_time in s": round(bounds_time,2),
                "variable_selection in s": round(variable_selection_time,2),
                "greedy_heuristics_time in s": round(greedy_time,2),
                "other_time in s": round(time.time() - start_time - (time.time() - start_time) - node_selection_time - implied_demands_time - bounds_time - variable_selection_time -greedy_time,2)
            },
            "nodes": {
                "total": 1,
                "pruned": 1,
                "infeasible": 1,
                "finished": 1,
            },
            "solution": {
                "LB": "infeasible",
                "UB": "infeasible",
                "gap in %": -1,
                "incumbent": [],
                "memory in MB": process.memory_info().rss / (1024 * 1024),
            },
            "bounds": {
                "lower_bounds": "infeasible",
                "upper_bounds": "infeasible",
                "Currently needed memory": process.memory_info().rss / (1024 * 1024)
            },
        }

        # Finally, write results to json file
        filepath = os.path.join("results",name) + ".json"
        with open(filepath,"w") as f:
            json.dump(results,f)
        return False
    else: 
        rootnode.set_bounds(temporary_pref_ordering,do_Lagrangian)        

    # 3. If rootnode feasible (see before), set up a branch and bound tree consisting of the root node and its children  
    tree = BnB_tree(rootnode.facilities,0,inf)
    tree.nodes[""] = rootnode
    tree.update_UB()
    tree.update_LB()
    timelimit = 60*60    # 60 seconds, 60 times (=1 h), 24 times (=24 h)
    rootnode.set_branching_variable(temporary_pref_ordering)

    # 4. Start B&B    
    # We log times for all major parts of the algorithm, as well as times of new bounds
    iterations = 0
    time_for_initialisation = round(time.time()-start_time,2)
    lower_bounds, upper_bounds = [(tree.LB,round(time.time() - start_time,2))], [(tree.UB,round(time.time() - start_time,2))]
    while time.time() - start_time < timelimit and process.memory_info().rss / (1024 * 1024) <= 10000:
        process = psutil.Process(os.getpid())   
        beginning_of_iteration = time.time() # For various logging purposes
        iterations += 1

        # 4.1 Pick child to start branching on
        try: minimum_LB_among_nodes_for_branching = min([tree.nodes[node].LB for node in tree.nodes if tree.nodes[node].status == "Evaluated"])
        except: minimum_LB_among_nodes_for_branching = BnB_node.primalBound
        potential_nodes_for_branching = [node for node in tree.nodes if tree.nodes[node].status == "Evaluated" if tree.nodes[node].LB <= minimum_LB_among_nodes_for_branching]
        if potential_nodes_for_branching == []: break # == we are at the end of the tree
        node = potential_nodes_for_branching[-1]
        NODE = tree.nodes[node]

        node_selection_time += time.time() - beginning_of_iteration

        # Branch whenever branching candidate is found          
        timer = time.time() # For logging purposes      
        l_child = node + "-" + str(NODE.branching_candidate)  # Node IDs
        r_child = node + "+" + str(NODE.branching_candidate)

        # Left child #######################################
        # Copy assignments from parent node for LHS, update temporary pref ordering for left child    
        tree.nodes[l_child] = BnB_node(NODE.facilities, NODE.local_position_of_least_preferred_facility, NODE.local_cannot_assign, NODE.branching_candidate,NODE.LR_lambda,NODE.LR_mu,NODE.LR_rho, [a+b for a,b in zip(NODE.depth,[1,0])],NODE.required_facilities)              
        temporary_pref_ordering_l_child = tree.nodes[l_child].close_facility(NODE.branching_candidate)
        othertime += time.time() - timer
        timer = time.time() # For logging purposes, LHS implied demands
        try: tree.nodes[l_child].local_position_of_least_preferred_facility, tree.nodes[l_child].local_cannot_assign = compute_implied_demands(demands=tree.nodes[l_child].demands,capacities=tree.nodes[l_child].capacities,temporary_pref_ordering=temporary_pref_ordering_l_child,position_of_least_preferred_facility=tree.nodes[l_child].local_position_of_least_preferred_facility,cannot_assign=tree.nodes[l_child].local_cannot_assign, facilities=tree.nodes[l_child].facilities,required_facilities=tree.nodes[l_child].required_facilities)
        except: 
            tree.nodes[l_child].status = "Infeasible"
        implied_demands_time += time.time() - timer

        temporary_pref_ordering_l_child = {i:[facility for facility in BnB_node.pref_ordering[i][:1+tree.nodes[l_child].local_position_of_least_preferred_facility[i]]
                    if facility in tree.nodes[l_child].facilities] for i in range(n_cust)}
        
        timer = time.time() # For logging purposes, LHS make feasible
        temporary_pref_ordering_l_child = tree.nodes[l_child].make_feasible()        
        # Left child #######################################

        # Right child #######################################
        # Copy assignments from parent node for RHS, update temporary pref ordering for right child            
        tree.nodes[r_child] = BnB_node(NODE.facilities, NODE.local_position_of_least_preferred_facility, NODE.local_cannot_assign, NODE.branching_candidate,NODE.LR_lambda,NODE.LR_mu,NODE.LR_rho, [a+b for a,b in zip(NODE.depth,[0,1])], NODE.required_facilities)        
        temporary_pref_ordering_r_child = tree.nodes[r_child].open_facility(NODE.branching_candidate)
        othertime += time.time() - timer
        # Right child #######################################
        
        # Do modified strong branching before computing bounds
        timer = time.time() # For logging purposes
        if tree.nodes[l_child].status != "Infeasible": tree.nodes[l_child].set_branching_variable(temporary_pref_ordering_l_child)
        if tree.nodes[r_child].status != "Infeasible": tree.nodes[r_child].set_branching_variable(temporary_pref_ordering_r_child)
        variable_selection_time += time.time() - timer

        # Set bounds for both new nodes, this also detect infeasibility
        timer = time.time() # For logging purposes
        if tree.nodes[l_child].status != "Infeasible": tree.nodes[l_child].set_bounds(temporary_pref_ordering_l_child,do_Lagrangian,tree.LB,tree.UB)
        if tree.nodes[r_child].status != "Infeasible": tree.nodes[r_child].set_bounds(temporary_pref_ordering_r_child,do_Lagrangian,tree.LB,tree.UB)
        bounds_time += time.time() - timer
        
        # Optionally, do some primal heuristics here
        timer = time.time() # For logging purposes
        if time.time() - start_time < timelimit:
            if tree.nodes[node].status != "Infeasible":
                if tree.primal_GRASP_probing(node,iterations=2,shortlist=2): 
                    upper_bounds.append((tree.UB,round(time.time() - start_time,2)))
            if tree.update_LB(): lower_bounds.append((tree.LB,round(time.time() - start_time,2),tree.LB_type))
            tree.update_UB()
            greedy_time += time.time() - timer
        else: break

        # Flag facilities that no longer allow for branching as finished
        timer = time.time() # For logging purposes
        if tree.nodes[l_child].branching_candidate == None:
            tree.nodes[l_child].status = "Finished" 
        if tree.nodes[r_child].branching_candidate == None:
            tree.nodes[r_child].status = "Finished" 

        # Update global bounds on all open nodes
        tree.nodes[node].status = "Finished"
        if tree.UB < BnB_node.primalBound:
            BnB_node.primalBound = tree.UB

        # Intermittently report progress        
        current_time = time.time()
        if (iterations) % 10 == 0: 
            print(iterations,"LB/UB Tree",int(tree.LB),int(tree.UB),"  Node",int(tree.nodes[node].LB),int(tree.nodes[node].UB), 
            "  Open nodes:",len([node for node in tree.nodes if tree.nodes[node].status == "Evaluated"]),
            "  Gap:",round((tree.UB - tree.LB)/tree.UB*100,4),
            "  Current depth -/+:",node.count("-"),node.count("+"),
            "  Time:",round(current_time - start_time,2), flush=True)

        # Finally prune the tree in each step
        tree.prune()
        othertime += time.time() - timer

        process = psutil.Process(os.getpid())         
        memory_usage = round(process.memory_info().rss / (1024 * 1024),2)                        
        name = "_".join([str(i) for i in sys.argv[1:]])
        results = {
            "input parameters": sys.argv[1:],
            "instance": {
                "facilities": n_facs,
                "customers": BnB_node.n_cust,
            },
            "times": {
                "total_time in s": round(time.time() - start_time, 2),
                "initialisation_time in s": round(time_for_initialisation,2),
                "node_selection_time in s": round(node_selection_time,2),
                "implied_demands_time in s": round(implied_demands_time,2),
                "bounds_time in s": round(bounds_time,2),
                "variable_selection in s": round(variable_selection_time,2),
                "greedy_heuristics_time in s": round(greedy_time,2),
                "other_time in s": round(time.time() - start_time - time_for_initialisation - node_selection_time - implied_demands_time - bounds_time - variable_selection_time -greedy_time,2)
            },
            "nodes": {
                "total": len(tree.nodes) +tree.evaluated + tree.infeasible,
                "pruned": tree.evaluated,
                "infeasible": tree.infeasible,
                "finished": len([node for node in tree.nodes if tree.nodes[node].status == "Finished"]),
            },
            "solution": {
                "LB": round(tree.LB,4),
                "UB": round(tree.UB,4),
                "gap in %": round((tree.UB - tree.LB)/tree.UB*100,2),
                "incumbent": [int(i) for i in tree.incumbent_facilities],
                "memory in MB": round(memory_usage,4),
            },
            "bounds": {
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds
            },
        }

        # Ensure that results are indeed correct
        validate(results['solution'],tree.facilities,list(range(n_cust)),pref_ordering,demands,capacities,fixedCost,assignment_costs)
        check(results["solution"],"_".join([str(i) for i in sys.argv[1:-1]])+"_no_preprocessing")

        # Finally, write results to json file
        filepath = os.path.join("results",name) + ".json"
        with open(filepath,"w") as f:
            json.dump(results,f)
        
    print("Finished after",iterations,"iterations")
    print("Total time:",round(time.time() - start_time,2),"seconds")
    print(iterations,"Tree LB/UB",int(tree.LB),int(tree.UB)," Currently open nodes:",len([node for node in tree.nodes if tree.nodes[node].status == "Evaluated"]),
                "  Gap:",round((tree.UB - tree.LB)/tree.UB*100,4))
    process = psutil.Process(os.getpid())         
    memory_usage = round(process.memory_info().rss / (1024 * 1024),2)                        
    name = "_".join([str(i) for i in sys.argv[1:]])
    results = {
        "input parameters": sys.argv[1:],
        "instance": {
            "facilities": n_facs,
            "customers": BnB_node.n_cust,
        },
        "times": {
            "total_time in s": round(time.time() - start_time, 2),
            "initialisation_time in s": round(time_for_initialisation,2),
            "node_selection_time in s": round(node_selection_time,2),
            "implied_demands_time in s": round(implied_demands_time,2),
            "bounds_time in s": round(bounds_time,2),
            "variable_selection in s": round(variable_selection_time,2),
            "greedy_heuristics_time in s": round(greedy_time,2),
            "other_time in s": round(time.time() - start_time - time_for_initialisation - node_selection_time - implied_demands_time - bounds_time - variable_selection_time -greedy_time,2)
        },
        "nodes": {
            "total": len(tree.nodes) +tree.evaluated + tree.infeasible,
            "pruned": tree.evaluated,
            "infeasible": tree.infeasible,
            "finished": len([node for node in tree.nodes if tree.nodes[node].status == "Finished"]),
        },
        "solution": {
            "LB": round(tree.LB,4),
            "UB": round(tree.UB,4),
            "gap in %": round((tree.UB - tree.LB)/tree.UB*100,2),
            "incumbent": [int(i) for i in tree.incumbent_facilities],
            "memory in MB": round(memory_usage,4),
        },
        "bounds": {
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds
        },
    }

    # Ensure that results are indeed correct
    validate(results['solution'],tree.facilities,list(range(n_cust)),pref_ordering,demands,capacities,fixedCost,assignment_costs)
    check(results["solution"],"_".join([str(i) for i in sys.argv[1:-1]])+"_no_preprocessing")

    # Finally, write results to json file
    filepath = os.path.join("results",name) + ".json"
    with open(filepath,"w") as f:
        json.dump(results,f)

if __name__ == "__main__":
    i = int(sys.argv[1])
    pref_ = int(sys.argv[2]) # Preferences = Distances: 0, Preferences = perturbed Distances: 1
    print("Considering instance "+str(i+1)+"\nConsidering Preference type "+str(pref_))
    if pref_ == 0: print("Preferences correspond to Closest Assignments.\n")
    elif pref_ == 1: print("Preferences according to Cánovas et al. (2007).\n")
    elif pref_ == 2: print("Preferences are strict closest assignments")
    elif pref_ == 3: print("Preferences according to perturbed closest assignments with indifferences")
    elif pref_ == 4: print("Preferences according to reversed closest assignments")
    elif pref_ == 5: print("Preferences according to strict reversed closest assignments")

    # Choice of instance; big: 300x300, medium: 100x75, small: 75x50
    insType = sys.argv[3]   # "300" für i300, cap for capABC, p
    insTypeLetter = "a" # "i", "a", "b", "c", "p"

    try: insTypeLetter = sys.argv[4] # "i", "a", "b", "c", "p"
    except: insTypeLetter = ""

    try: setting = sys.argv[5] # "10075075", "75500375"    
    except: setting = ""

    if sys.argv[6] != "Lagrangian":
        do_Lagrangian = False
    else: do_Lagrangian = True

    # if preproc_assignToMostPref or preproc_coincidingSets or preproc_maximallyContained:
    # 300x300 instances: python3 main.py instanz 1 300 -> mit instanz in {0,...19}
    # medium instances: python3 main.py instanz 1 cap a/b/c setting -> mit instanz in {1,2,3,4} und setting in {"10075075", "75500375"}
    # small instances: python3 main.py instanz 1 cap d setting -> mit instanz in {131,132,133,134} und setting in {"10075075", "75500375"}
    main(i,insType,insTypeLetter,setting,pref_,do_Lagrangian,ties=1)
