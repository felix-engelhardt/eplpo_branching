#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created October 2022
Last updated August 2025

@author: Sophia, refactored by Felix
"""

import sys, time, os, json
from initialisation import read_instance
from gurobipy import GRB, quicksum 
from branching import compute_implied_demands_in_IP, make_feasible
import gurobipy as gp
from helper import validate
import psutil

def mainMIP(i:int,insType,insTypeLetter,pref,setting,do_preprocessing = True):
    start_time = time.time()

    # 1. Define important sets and parameters                                           
    capacities, demands, fixedCost, assignment_costs, n_cust, n_facs, pref_ordering = read_instance(i,insType,insTypeLetter,setting,pref)
    facilities, customers = list(range(n_facs)), list(range(n_cust))
    lower_bounds, upper_bounds = [],[] # Used for logging progress so that primal-dual integral can be computed
    cannot_assign = {j:set() for j in facilities}
    
    if do_preprocessing: # Optionally: Turn off preprocessing for computational study
        print("Do preprocessing.")
        # Whats the minimum number of facilities I need in order to serve all demand? => k then use this to cut off the preflists            
        try:
            position_of_least_preferred_facility = {i:len(pref_ordering[i])-1 for i in range(n_cust)}    # tells us the position of the least preferred facility in pref_ordering of each customer
            facilities, pref_ordering = make_feasible(facilities,customers,pref_ordering,demands,capacities)
            position_of_least_preferred_facility, cannot_assign = compute_implied_demands_in_IP(demands,capacities,pref_ordering,position_of_least_preferred_facility,cannot_assign, facilities,required_facilities=[])                
            local_pref_ordering = [pref_ordering[i][:1+position_of_least_preferred_facility[i]] for i in customers]
        except:
            print("Instance is infeasible!")
            local_pref_ordering = pref_ordering
    else: local_pref_ordering = pref_ordering

    # 2. Set up model  
    m = gp.Model("EPLPO")        
    m.setParam("Threads", 1) # For benchmarking purposes
    m.setParam("SoftMemLimit",10)
    m._lastbound = float("-inf") # Used for logging dual bound improvements in callback

    # 3. Variables  
    x = {} # Customer => facility allocation, y represents facility openening
    for i in customers:  x[i] = m.addVars(local_pref_ordering[i],vtype = GRB.BINARY, lb=0, ub=1, name = ["x_" + str(i) + "_" + str(j) for j in local_pref_ordering[i]])                
    y = m.addVars(facilities,vtype = GRB.BINARY, lb=0, ub=1, name = ["y_" + str(j) for j in facilities])
            
    # 4. Constraints     
    m.addConstrs(quicksum(x[i][j] for j in x[i]) == 1 for i in customers) # Each customer i gets exactly one facility    
    m.addConstrs(quicksum(demands[i]*x[i][j] for i in customers if j in x[i]) <= capacities[j] * y[j] for j in facilities) # Capacity-Linking constraints                       
    m.addConstrs(x[i][j]  <= y[j] for i in customers for j in x[i]) # Additional linking constraints to strengthen the LP-relaxation  
    m.addConstrs(quicksum(x[i][j] for j in local_pref_ordering[i][j_index+1:] if j in x[i]) + y[local_pref_ordering[i][j_index]] <= 1 for i in customers for j_index in range(len(x[i])))  # Wagner and Falkson's preference constraints

    # 5. Objective    
    m.setObjective(quicksum(fixedCost[j]*y[j] for j in facilities) + quicksum(assignment_costs[i][j]*x[i][j] for i in customers for j in x[i]), GRB.MINIMIZE)            
    model_building_time = time.time()-start_time
    m.update()  

    # 6. Callback definition for logging
    def callback(model,where):
        if where == GRB.Callback.MIPSOL: upper_bounds.append([model.cbGet(GRB.Callback.MIPSOL_OBJ),model.cbGet(GRB.Callback.RUNTIME)])
        elif where == GRB.Callback.MIP: # General MIP progress: can query current best boun
            if model.cbGet(GRB.Callback.MIP_OBJBND) > model._lastbound + 1e-6: # Only note (non-trivally) improving dual bounds
                model._lastbound = model.cbGet(GRB.Callback.MIP_OBJBND)
                lower_bounds.append([model.cbGet(GRB.Callback.MIP_OBJBND),model.cbGet(GRB.Callback.RUNTIME)])
    
    # 7. Optimisation
    build_up_time = time.time() - start_time
    m.setParam("TimeLimit", max([60*60 - build_up_time,0])) # Hardcoded for the computational study: 12h max.    
    m.optimize(callback)
    model_solving_time = time.time()-start_time-model_building_time
    
    # 8. Logging of results
    print("Finished after",m.NodeCount,"iterations")
    print("Total time:",round(time.time() - start_time,2),"seconds")

    process = psutil.Process(os.getpid())         
    memory_usage = round(process.memory_info().rss / (1024 * 1024),2)                    
    name = "_".join([str(i) for i in sys.argv[1:]])
    results = {
        "input parameters": sys.argv[1:],
        "instance": {
            "facilities": n_facs,
            "customers": n_cust,
        },
        "times": {
            "total_time in s": round(time.time() - start_time, 2),
            "model_building_time in s": round(model_building_time,2),
            "model_solving_time in s": round(model_solving_time,2),
        },
        "nodes": {
            "total": m.NodeCount
        },
        "solution": {
            "LB": round(m.ObjBound,4) if m.solCount > 0 else "infeasible",
            "UB": round(m.ObjVal,4) if m.solCount > 0 else "infeasible",
            "gap in %": round(m.MIPGap,2) if m.solCount > 0 else "infeasible",
            "incumbent": "infeasible" if m.solCount == 0 else [j for j in facilities if y[j].x >= 0.99],
            "memory in MB": memory_usage,
        },
        "bounds": {
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds
        },
    }

    # 9. Validate that solution is indeed correct
    validate(results['solution'],facilities,customers,local_pref_ordering,demands,capacities,fixedCost,assignment_costs)

    # 10. Finally, write results to json file
    filepath = os.path.join("results",name) + "_IPsolution.json"
    with open(filepath,"w") as f:
        json.dump(results,f,indent=4)
    
if __name__ == "__main__":
    # 300x300 instances: python3 main.py instanz 1 300 -> mit instanz in {0,...19}
    # medium instances: python3 main.py instanz 1 cap a/b/c setting -> mit instanz in {1,2,3,4} und setting in {"10075075", "75500375"}
    # small instances: python3 main.py instanz 1 cap d setting -> mit instanz in {131,132,133,134} und setting in {"10075075", "75500375"}

    i = int(sys.argv[1])
    pref = int(sys.argv[2]) # Preferences = Distances: 0, Preferences = perturbed Distances: 1

    # Choice of instance; big: 300x300, medium: 100x75, small: 75x50
    insType = sys.argv[3]   # "300" für i300, cap for capABC, p
    insTypeLetter = "a" # "i", "a", "b", "c", "p"
    
    try: insTypeLetter = sys.argv[4] # "i", "a", "b", "c", "p"
    except: insTypeLetter = ""

    try: setting = sys.argv[5] # "75_100", "50_75"    
    except: setting = ""

    if sys.argv[6] != "preprocessing":
        do_preprocessing = False
    else: do_preprocessing = True

    if insType != "cap": print("Considering instance "+str(i+1)+"\nConsidering Preference type "+str(pref))
    else: print("Considering instance "+str(i)+"\nConsidering Preference type "+str(pref))

    if pref == 0: print("Preferences are defined by assignment cost.")
    elif pref == 1: print("Preferences according to Cánovas et al. (2007).")

    print("NN:,i,insType,insTypeLetter,pref,setting,preprocessing")
    print("NN:",i,insType,insTypeLetter,pref,setting,do_preprocessing)
    mainMIP(i,insType,insTypeLetter,pref,setting,do_preprocessing=do_preprocessing)
