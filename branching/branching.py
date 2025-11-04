from copy import deepcopy
from random import choice
import networkx as nx
from math import floor
from time import time
from math import inf
import numpy as np

class BnB_node:
    """ Class representing a node in a B&B tree."""

    # Global variables shared across all B&B nodes
    capacities = None # of facilities
    demands = None # of customers
    pref_ordering = None # of each customer
    n_cust = None # = len(demands)
    assignment_costs = None # of customers and facilities
    fixedCost = None # of opening facilities
    totaldemand = None # of all customers combined
    primalBound = None

    # Initialisation only requires a set of facilities
    def __init__(self,facilities,local_position_of_least_preferred_facility,local_cannot_assign,former_branching_candidate,former_LR_lambda,former_LR_mu,former_LR_rho,depth,required_facilities=set()):
        self.facilities = deepcopy(facilities)
        self.customers = np.array(range(self.n_cust))
        self.LB = None
        self.UB = None
        self.LB_type = None
        self.status = 'Open' # ''Evaluated' after determining bounds, 'Infeasible' if infeasibility was detected
        self.n_facs = len(self.facilities)
        self.local_position_of_least_preferred_facility = deepcopy(local_position_of_least_preferred_facility)
        self.required_facilities = deepcopy(required_facilities)
        self.last_branching_decision = deepcopy(former_branching_candidate)
        self.LR_lambda = deepcopy(former_LR_lambda)
        self.LR_mu = deepcopy(former_LR_mu)
        self.LR_rho = deepcopy(former_LR_rho)
        self.depth = depth
        self.branching_candidate = None
        self.incumbent = None
        self.local_cannot_assign = deepcopy(local_cannot_assign)    # contains a list for each facility with customers who can not be assigned to it   

    def remove_from_preflist(self,remove=[]) -> bool:
        """ Removes a list of facilities or a single facility from the preference list """
        temporary_pref_ordering = {i:[facility for facility in BnB_node.pref_ordering[i][:1+self.local_position_of_least_preferred_facility[i]]
                    if facility in self.facilities] for i in self.customers}
        
        if type(remove) == int or type(remove) == np.int32 or type(remove) == np.int64: remove = [remove] # For compatibility, so both j and [j] work
        for i in range(self.n_cust):
            for j in remove:
                try: 
                    temporary_pref_ordering[i].remove(j)                    
                except: pass
                if temporary_pref_ordering[i] == []: 
                    self.status = 'Infeasible'
                    return False
            self.local_position_of_least_preferred_facility[i] = BnB_node.pref_ordering[i].index(temporary_pref_ordering[i][-1])
            for i in [i for i in range(self.n_cust) if len(temporary_pref_ordering[i]) == 1]:
                if self.status != 'Infeasible': # Enforce customers_with_one_facility_in_preference_list
                    if temporary_pref_ordering[i][0] not in self.required_facilities: 
                        temporary_pref_ordering = self.open_facility(temporary_pref_ordering[i][0]) 
                else:
                    return False
        return temporary_pref_ordering
        
    def close_facility(self,facility:int,make_feasible=True):
        """ Removes the facility corresponding to the facility from the dics of facilities and the preflists.
            The "make_feasible" parameter is for usage in the make_feasible function to turn off recursive calls."""
        temporary_pref_ordering = self.remove_from_preflist(facility)
        if facility in self.required_facilities:            
            self.status = "Infeasible"
            return False    
        if self.status == 'Infeasible': return False
        else: 
            del self.facilities[facility]
            if make_feasible: 
                try: return self.make_feasible()
                except: return False
            return temporary_pref_ordering
        
    def open_facility(self,facility:int):
        """ Open the facility corresponding to the facility from the dict of facilities and updates the preflists.
            Note that opening does not call any other functions, it never requires a follow up, just reducing preflists."""
        temporary_pref_ordering = {i:[facility for facility in BnB_node.pref_ordering[i][:1+self.local_position_of_least_preferred_facility[i]]
                    if facility in self.facilities] for i in self.customers}
        
        if facility not in self.facilities: 
            self.status = 'Infeasible'
            return False
        else:             
            for i in self.customers: # cut off pref orderings 
                if len(temporary_pref_ordering[i]) == 0: 
                    self.status = 'Infeasible'
                    return False
                index,index_old = len(temporary_pref_ordering[i]),-1
                # If facility in local pref ordering of i, delete all facilities in i's local pref ordering until facility
                if facility in temporary_pref_ordering[i]:
                    while index > 0:
                        if temporary_pref_ordering[i][-1] == facility: 
                            break
                        else:
                            del temporary_pref_ordering[i][-1]
                        index -= 1   
                # Delete last facility from local pref ordering of i if it cannot serve i's implied demand after decreasing L_i
                while index_old != index:
                    index_old = index
                    if i in self.local_cannot_assign[temporary_pref_ordering[i][-1]]:
                        del temporary_pref_ordering[i][-1]
                        index -= 1 
                    if len(temporary_pref_ordering[i]) == 0:
                        self.status = 'Infeasible'
                        return False
                self.local_position_of_least_preferred_facility[i] = BnB_node.pref_ordering[i].index(temporary_pref_ordering[i][-1])                                
            self.required_facilities.add(facility)
        return temporary_pref_ordering

    def make_feasible(self):
        """ For a given set openFacs, calculate the largest subset of open facilities that results in a feasible assignment.
            Assumes a correct preflist and returns a correct preflist. 
            Note: If you change this, also change the corresponding function in ip.py. """
        temporary_pref_ordering = {i:[facility for facility in BnB_node.pref_ordering[i][:1+self.local_position_of_least_preferred_facility[i]]
                    if facility in self.facilities] for i in self.customers}
        
        # 1. Assign demand
        demandAtFac = {j:[0,[]] for j in self.facilities} # Demand at at facility and assigned customers
        for i in range(BnB_node.n_cust):
            try:
                demandAtFac[temporary_pref_ordering[i][0]][0] += BnB_node.demands[i]
                demandAtFac[temporary_pref_ordering[i][0]][1].append(i)
            except:      
                self.status = 'Infeasible'
                return False

        while len(self.facilities) > 0:

            # 2. Close overloaded facilities
            reassign = []
            j_index = 0
            for j in self.facilities.copy():
                if demandAtFac[j][0] > BnB_node.capacities[j]: # If facilities are overloaded, close them or terminate
                    reassign += demandAtFac[j][1]
                    try: 
                        temporary_pref_ordering = self.close_facility(j,make_feasible=False)
                        del demandAtFac[j] 
                    except: 
                        pass
                else:
                    j_index +=1
            if reassign == [] or self.status == "Infeasible": break # If no facilities were overloaded or infeasibility was detected, we are done

            # 3. Reassign customers whose facilities are overloaded
            for i in reassign:
                try:
                    demandAtFac[temporary_pref_ordering[i][0]][0] += BnB_node.demands[i]
                    demandAtFac[temporary_pref_ordering[i][0]][1].append(i)   
                except: 
                    self.status = 'Infeasible'
                    return False
        
        if not self.status == "Infeasible":
            # Enforce opening of facilities that are the last option for a customer
            for i in self.customers:
                if len(temporary_pref_ordering[i]) == 1 and temporary_pref_ordering[i][0] not in self.required_facilities:
                    temporary_pref_ordering = self.open_facility(temporary_pref_ordering[i][0])
            self.n_facs = len(self.facilities)
            cost_per_capacity = [BnB_node.fixedCost[j]/BnB_node.capacities[j] for j in self.facilities]
            self.facilities_sorted_by_relative_cost_per_capacity = [x for _, x in sorted(zip(cost_per_capacity, self.facilities), key=lambda pair: pair[0])]
            return temporary_pref_ordering
        else: 
            self.status = 'Infeasible'
            return False       
    
    def greedy_ub(self,initially_open_facilities,temporary_pref_ordering,flowDict):     
        """Compute an upper bound based on the solution of a min-cost flow solution"""
        # Initialisation: define already-open facilities by required facilities + facilities, which are used in a min-cost flow solution
        if len(initially_open_facilities) > 0: open_facilities = set(initially_open_facilities).union(set(j for j in self.facilities if flowDict["Facility"+str(j)]["sink"] > 0))
        else: open_facilities = set(j for j in self.facilities if flowDict["Facility"+str(j)]["sink"] > 0)
        infeasible = False
        
        # 1. Determine whether set open_facilities induces a feasible solution
        # 1.1 Construct feasible customer assignment
        assigned_customers_per_facility = {j:[] for j in self.facilities}        
        for i in range(BnB_node.n_cust):
            for fac in temporary_pref_ordering[i]:
                if fac in open_facilities:
                    assigned_customers_per_facility[fac].append(i)
                    break
        # 1.2 Feasibility check
        sum_of_demands_at_j = {j:sum(BnB_node.demands[i] for i in assigned_customers_per_facility[j]) 
                               if sum(BnB_node.demands[i] for i in assigned_customers_per_facility[j]) <= BnB_node.capacities[j] 
                               else - self.totaldemand   # The capacity of a facility in open_facilities is violated -> setting is infeasible. We catch this in the next if.
                               for j in open_facilities}
        if sum(sum_of_demands_at_j[j] for j in open_facilities) != self.totaldemand:
            infeasible = True       # There are unassigned customers
        
        if infeasible:  # Add a new facility to open_facilities according to greedy procedure.
            facilities_sorted_for_greedy = [(BnB_node.fixedCost[j] + sum(BnB_node.assignment_costs[i][j] for i in range(self.n_cust) if temporary_pref_ordering[i][0] == j))/BnB_node.capacities[j] for j in self.facilities]
            fac = [x for _, x in sorted(zip(facilities_sorted_for_greedy,
                                            [j for j in self.facilities]), key=lambda pair: pair[0]) if x not in open_facilities][0]
            open_facilities.add(fac)   

        # 2. Create feasible solution by first solving another network flow instance and, if infeasible, adding new facilities to open_facilities
        index = 0
        while infeasible:
            
            index += 1             
            infeasible = False
            
            # 2.1 Compute lower bound on solution via mcf; network under the assumption that all facilities in open_facilities are open in the solution.
            resolution = 10
            G = nx.DiGraph()
            sinkdemand = 0 
            for i in range(BnB_node.n_cust): 
                sinkdemand += int(BnB_node.demands[i]) # Use exactly the inverse of the node demands to ensure consistency
                G.add_node("Customer"+str(i), demand=-int(BnB_node.demands[i]))                
                G.add_edge("Customer"+str(i),"Facility"+str(temporary_pref_ordering[i][0]),    # Using the arc to my most-preferred facility incurs no costs
                                weight=0,
                                capacity=int(BnB_node.demands[i])) 
                if temporary_pref_ordering[i][0] not in open_facilities: 
                    for j in temporary_pref_ordering[i][1:]:
                        if not (i in self.local_cannot_assign[j]):
                            G.add_edge("Customer"+str(i),"Facility"+str(j),
                                        weight=floor(resolution*(BnB_node.assignment_costs[i][j]/BnB_node.demands[i])),
                                        capacity=int(BnB_node.demands[i])) # Divide by total capacity of edge to get price per unit
                        if j in open_facilities: break  # Only consider arcs up to my most preferred facility in open_facilities.
            G.add_node("sink",demand=sinkdemand) # Cover all demand with a single sink
            for j in self.facilities: 
                if j in open_facilities: weight = 0 # Required facilities are already paid for and provide their remaining capacity for free
                else: 
                    weight = floor(resolution*(BnB_node.fixedCost[j] + sum(BnB_node.assignment_costs[i][j] for i in range(self.n_cust) if temporary_pref_ordering[i][0] == j))/BnB_node.capacities[j])
                G.add_edge("Facility"+str(j),"sink",capacity=int(BnB_node.capacities[j]),weight=weight)
            flowDict = nx.min_cost_flow(G)            
            
            # 2.2 Construct potentially feasible solution: add all facilities utilised in the mcf-solution to open_facilities
            assigned_customers_per_facility = {j:[] for j in self.facilities}        
            open_facilities = open_facilities.union(set(j for j in self.facilities if flowDict["Facility"+str(j)]["sink"] > 0))
            for i in range(BnB_node.n_cust):    # Reconstruct feasible customer assignment
                for fac in temporary_pref_ordering[i]:
                    if fac in open_facilities:
                        assigned_customers_per_facility[fac].append(i)
                        break
            
            # 2.3 Feasibility check
            sum_of_demands_at_j = {j:sum(BnB_node.demands[i] for i in assigned_customers_per_facility[j]) 
                               if sum(BnB_node.demands[i] for i in assigned_customers_per_facility[j]) <= BnB_node.capacities[j] 
                               else - self.totaldemand   # The capacity of a facility in open_facilities is violated -> setting is infeasible. We catch this in the next if.
                               for j in open_facilities}
            if sum(sum_of_demands_at_j[j] for j in open_facilities) != self.totaldemand:
                infeasible = True       # There are unassigned customers
                        
            # 2.4 If infeasible, add new facility to open_facilities
            if infeasible:
                fac = [x for _, x in sorted(zip(facilities_sorted_for_greedy,[j for j in self.facilities]), key=lambda pair: pair[0]) if x not in open_facilities][0]
                open_facilities.add(fac)        

        # 3. We probably overestimaded the set of needed open facilities. Can we close unnecessary facilities via Greedy? If infeasible, can we apply makefeasible?
        there_are_closable_facilities,index2 = True,0
        while there_are_closable_facilities:
            index2 += 1
            there_are_closable_facilities = False
                            
            # 3.1 Determine facility that we try to close next. Rule: preferably facilities with high fixed cost or facilities with a lot of leftover capacity
            saved_cost = [(BnB_node.capacities[j] - sum(BnB_node.demands[i] for i in assigned_customers_per_facility[j])) / 
                           (BnB_node.fixedCost[j] + sum(BnB_node.assignment_costs[i][j] for i in assigned_customers_per_facility[j])) for j in open_facilities]            
            facilities_ordered_by_cost = [x for _, x in sorted(zip(saved_cost,[j for j in open_facilities]), key=lambda pair: pair[0]) if x not in self.required_facilities]

            # 3.2 Check whether its both feasible and advantageous to close j
            upper_bound = sum(BnB_node.fixedCost[j] + sum(BnB_node.assignment_costs[i][j] for i in assigned_customers_per_facility[j]) for j in open_facilities)  
            for j in facilities_ordered_by_cost:            
                if j in open_facilities:                            
                    # Peek one step ahead: what happens if we close j?
                    whatIf = open_facilities.copy()
                    whatIf.remove(j)                     
                    
                    isfeasible, allReqFacsOpen, allCustServed = False, True, True   # is solution feasible? are all required facilities open? are all customers served?
                    demandAtFac = {k:[0,[]] for k in whatIf.copy()} # Demand and assigned customers at facility
                    
                    # 3.2.1 Test whether at least one fac in whatIf occurs in the pref list of each customer; assign each customer to their most-pref fac in whatIf
                    for i in range(BnB_node.n_cust):
                        i_is_assigned = False
                        for k in temporary_pref_ordering[i]: 
                            if k in demandAtFac:
                                demandAtFac[k][0] += BnB_node.demands[i]
                                demandAtFac[k][1].append(i)
                                i_is_assigned = True
                                break
                        if not i_is_assigned: # there is an unassigned customer                                
                            allCustServed = False
                            isfeasible = False
                    # 3.2.2 Test whether the capacity at each facility is met if all customers are assigned
                    if allCustServed:   
                        while not isfeasible:   # Close facilities until either all facilities are closed or a feasible solution is found.                                
                            isfeasible = True            
                            # Close overloaded facilities
                            reassign = []
                            for k in whatIf.copy():
                                if demandAtFac[k][0] > BnB_node.capacities[k]: # If facilities are overloaded, close them
                                    reassign += demandAtFac[k][1]
                                    del demandAtFac[k] 
                                    whatIf.remove(k)
                                    isfeasible = False 
                                    if k in self.required_facilities: 
                                        allReqFacsOpen = False
                                        break

                            # Reassign customers whose facilities are overloaded
                            if allReqFacsOpen and allCustServed:        
                                if len(reassign) > 0:
                                    for i in reassign:
                                        isCustServed = False                                
                                        for k in temporary_pref_ordering[i]: 
                                            if k in demandAtFac:
                                                demandAtFac[k][0] += BnB_node.demands[i]
                                                demandAtFac[k][1].append(i)
                                                isCustServed = True
                                                break
                                        if not isCustServed: # Closing the facility leads to an infeasible solution.
                                            allCustServed = False
                                            isfeasible = False
                                            break
                            if not (allReqFacsOpen and allCustServed) or len(whatIf) == 0: 
                                isfeasible = False
                                break 

                        if isfeasible:  # If the constructed solution is feasible, update open_facilities from before
                            there_are_closable_facilities = True
                            for k in demandAtFac:
                                    assigned_customers_per_facility[k] = demandAtFac[k][1]
                            open_facilities = whatIf.copy()   
                            
        # If a facility is not used, remove it.                                       
        for j in open_facilities.copy():
            if len(assigned_customers_per_facility[j]) == 0 and j not in self.required_facilities:
                open_facilities.remove(j)
                del assigned_customers_per_facility[j]
        # open_facilities induces a feasible solution. Update the value of the upper bound to the corresponding solution value.
        upper_bound = sum(BnB_node.fixedCost[j] + sum(BnB_node.assignment_costs[i][j] for i in assigned_customers_per_facility[j]) for j in open_facilities)        
        return upper_bound,open_facilities

    def set_bounds(self,temporary_pref_ordering,do_Lagrangian,treeLB=0,treeUB=inf): 
        """ UB is determined greedily. For LB two new LBs are compared with the current LB. """
        starttime = time()
        
        # 1. Set up list of LB candidates
        LB_candidates = [{'ID': 'incumbent','LB': treeLB}]        
                    
        # 2. Calculate combinatorial bound

        # Determine facility costs: fixed cost + cost of assigning all customers who prefer the facility most
        facility_costs = {j: BnB_node.fixedCost[j] 
                          + sum(BnB_node.assignment_costs[i][temporary_pref_ordering[i][0]] for i in range(self.n_cust) if temporary_pref_ordering[i][0] == j) for j in self.facilities}  # type: ignore
        
        # Assign remaining customers according to a min-cost-flow solution
        resolution = 1000
        G = nx.DiGraph()
        sinkdemand = 0 
        for i in range(self.n_cust): # type: ignore
            sinkdemand += int(BnB_node.demands[i]) # Use exactly the inverse of the node demands to ensure consistency, #type: ignore
            G.add_node("Customer"+str(i), demand=-int(BnB_node.demands[i])) # type: ignore
            G.add_edge("Customer"+str(i),"Facility"+str(temporary_pref_ordering[i][0]),
                            weight=floor(0), # Divide by total capacity of edge to get price per unit
                            capacity=int(BnB_node.demands[i]))   # type: ignore
            for j in temporary_pref_ordering[i][1:]:                
                if not (i in self.local_cannot_assign[j]):                        
                    G.add_edge("Customer"+str(i),"Facility"+str(j),
                            weight=floor(resolution*(BnB_node.assignment_costs[i][j]/BnB_node.demands[i])), # Divide by total capacity of edge to get price per unit
                            capacity=int(BnB_node.demands[i]))   # type: ignore               
        G.add_node("sink",demand=sinkdemand) # Cover all demand with a single sink
        for j in self.facilities: # But let each facility only send capacity up to its remaining capacity forward.
            if j in self.required_facilities:
                G.add_edge("Facility"+str(j),"sink",
                        weight=0,
                        capacity=int(BnB_node.capacities[j]))
            else:
                G.add_edge("Facility"+str(j),"sink",
                        weight=floor(resolution*facility_costs[j]/BnB_node.capacities[j]),
                        capacity=int(BnB_node.capacities[j]))
        flowCost = nx.min_cost_flow_cost(G)
        flowDict = nx.min_cost_flow(G)
        minimal_assignment_costs = flowCost/resolution

        # Update LB
        LB_candidates.append({'ID':'combined_combinatorial','LB': sum(facility_costs[j] for j in self.required_facilities) + minimal_assignment_costs})
        endtime = time()-starttime

        # 3. UB / Assign each customer to most preferred facility and open all used facilities
        ub,solution = self.greedy_ub(self.required_facilities,temporary_pref_ordering,flowDict)

        # 4. Lagrangian bound
        if do_Lagrangian == True:
            # Relax (1b), (1c), (1d)
            # 1. Solve flow problem in order to determine minimum number of needed open facilities analogously to before
            capacities_ = [BnB_node.capacities[j] for j in self.facilities if j not in self.required_facilities]  # type: ignore
            facilities_sorted_by_capacities = [x for _, x in sorted(zip(capacities_, [j for j in self.facilities if j not in self.required_facilities]), key=lambda pair: -pair[0])]
            k = len(self.required_facilities)
            covered_demand = sum(BnB_node.capacities[j] for j in self.required_facilities)  # type: ignore
            while covered_demand < self.totaldemand: # type: ignore
                j = facilities_sorted_by_capacities.pop(0)
                covered_demand += BnB_node.capacities[j]
                k += 1

            # Parameters for subgradient method
            iter, MAXiter, beta = 0,6,2
            kUnchanged,qParam = 4,0.5
            noChangeInObjective = 0      
            current_objective = -inf  

            LR_lambda_best =  self.LR_lambda.copy()
            LR_mu_best = self.LR_mu.copy()
            LR_rho_best = self.LR_rho.copy()
            
            # 2. Start with subgradient-method
            while iter < MAXiter:
                iter += 1                
                                
                # 2.0 Calcualte L^j_... parameters for each j in J
                fixed_cost_LR = {j: BnB_node.fixedCost[j] - self.LR_rho[j] + sum(self.LR_mu[i][j] for i in range(self.n_cust) if j in temporary_pref_ordering[i] if temporary_pref_ordering[i][0] not in self.required_facilities) for j in self.facilities}  # type: ignore
                
                LR_contribution = {j:{i: BnB_node.assignment_costs[i][j] - self.LR_lambda[i] + self.LR_rho[j]*BnB_node.demands[i]/BnB_node.capacities[j] 
                                    - sum(self.LR_mu[i][k] for k in temporary_pref_ordering[i][temporary_pref_ordering[i].index(j):] if j in temporary_pref_ordering[i])
                                    for i in range(self.n_cust) if j in temporary_pref_ordering[i] if temporary_pref_ordering[i][0] not in self.required_facilities} 
                                    for j in self.facilities}
                
                L_j = {j:[[i for i in range(self.n_cust) 
                        if temporary_pref_ordering[i][0] not in self.required_facilities 
                        if j in temporary_pref_ordering[i] 
                        if LR_contribution[j][i] <= 0],
                        fixed_cost_LR[j] + sum(LR_contribution[j][i] for i in range(self.n_cust) 
                                                if temporary_pref_ordering[i][0] not in self.required_facilities 
                                                if j in temporary_pref_ordering[i] 
                                                if LR_contribution[j][i] <= 0)] 
                        if j not in self.required_facilities else
                    [[i for i in range(self.n_cust) 
                        if temporary_pref_ordering[i][0] == j]
                        +[i for i in range(self.n_cust) 
                        if temporary_pref_ordering[i][0] not in self.required_facilities 
                        if j in temporary_pref_ordering[i] 
                        if LR_contribution[j][i] <= 0],
                        fixed_cost_LR[j] + sum(BnB_node.assignment_costs[i][temporary_pref_ordering[i][0]] + self.LR_rho[temporary_pref_ordering[i][0]]*BnB_node.demands[i]/BnB_node.capacities[temporary_pref_ordering[i][0]] 
                                            for i in range(self.n_cust) if temporary_pref_ordering[i][0] == j)
                                            + sum(LR_contribution[j][i] for i in range(self.n_cust) 
                                                    if temporary_pref_ordering[i][0] not in self.required_facilities 
                                                    if j in temporary_pref_ordering[i] 
                                                    if LR_contribution[j][i] <= 0)]
                        for j in self.facilities} # solution of subproblem for facility j

                # 2.1 Initialise solution values of LR
                y = {j: 1 if L_j[j][1] <= 0 else 1 if j in self.required_facilities else 0 for j in self.facilities}
                                    
                # 2.2 Update objective, capacity, number of opened facilities
                current_objective_new = sum(L_j[j][1] * y[j] for j in self.facilities) + sum(self.LR_lambda[i] for i in range(self.n_cust) if temporary_pref_ordering[i][0] not in self.required_facilities)
                current_capacity = sum(BnB_node.capacities[j] * y[j] for j in self.facilities)
                number_of_opened_facilities = sum(y[j] for j in self.facilities)
                
                # Then, add 
                remaining_facilities = [j for j in self.facilities if y[j] == 0]
                    
                # NOTE Two different approaches to solve the remaining problem: relax capacity or cardinalty constraints
                increment_capacity,increment_cardinality = 0,0 # used to compare the two approaches
                open_facilities_by_capacity,open_facilities_by_cardinality = [],[]
                # Approach A: Minimise the number of facilities opened to meet capacity (relax cardinalty)
                costs = [L_j[j][1]/BnB_node.capacities[j] for j in remaining_facilities]
                facilities_sorted_by_cost = [x for _, x in sorted(zip(costs,remaining_facilities), key=lambda pair: pair[0])]
                for index in range(len(remaining_facilities)):
                    if self.totaldemand > current_capacity:
                        current_facility = facilities_sorted_by_cost[index]

                        if current_capacity + BnB_node.capacities[current_facility] >= self.totaldemand: factor = (self.totaldemand - current_capacity)/BnB_node.capacities[current_facility]
                        else: factor = 1 # Use factor != 1 for the final facility

                        current_capacity = min(self.totaldemand,current_capacity + BnB_node.capacities[current_facility])
                        increment_capacity += factor * L_j[current_facility][1]                                         
                        open_facilities_by_capacity += [[current_facility,factor]] # y[current_facility] = factor                   
                    else: break
                
                # Approach B: Select cheapest facilities to meet cardinality (relax capacity)
                costs = [L_j[j][1] for j in remaining_facilities]
                facilities_sorted_by_cost = [x for _, x in sorted(zip(costs,remaining_facilities), key=lambda pair: pair[0])]
                required_facs = max(0,k - number_of_opened_facilities)
                for index in range(required_facs):
                    current_facility = facilities_sorted_by_cost[index]
                    increment_cardinality += L_j[current_facility][1]                                         
                    open_facilities_by_cardinality += [[current_facility,1]] # y[current_facility] = 1

                # Update y-variable values based on max(increment_capacity,increment_cardinality)
                if increment_capacity < increment_cardinality: 
                    current_objective_new += increment_cardinality
                    open_facilities_by_capacity = open_facilities_by_cardinality
                else: 
                    current_objective_new += increment_capacity
                for facility in open_facilities_by_capacity: 
                    y[facility[0]] = facility[1]

                # Assign allocation variable values
                x = {i:{j: y[j] if i in L_j[j][0] else 0 for j in self.facilities} for i in range(self.n_cust)}
                
                # Update lower bound
                LR_lambda_old = self.LR_lambda.copy()
                LR_mu_old = self.LR_mu.copy()
                LR_rho_old = self.LR_rho.copy()
                                
                if iter <= 1:
                    current_objective = current_objective_new
                else:
                    if current_objective_new > current_objective: 
                        current_objective = current_objective_new
                    
                        LR_lambda_best = self.LR_lambda.copy()
                        LR_mu_best = self.LR_mu.copy()
                        LR_rho_best = self.LR_rho.copy()

                    else:
                        noChangeInObjective += 1     
                        ### If best LB has not changed for kUnchanged, divide the step size by two and re-consider multipliers of the best solution found so far ###
                        if noChangeInObjective >= kUnchanged:
                            beta = qParam * beta                        
                            noChangeInObjective = 0

                            self.LR_lambda = LR_lambda_best.copy()
                            self.LR_mu = LR_mu_best.copy()
                            self.LR_rho = LR_rho_best.copy()                                           
                        
                ### Update multipliers ###
                # Find subgradient for current_objective
                sum_of_assignment_values = {i: sum(x[i][j] for j in self.facilities) for i in range(self.n_cust)}                
                s_iter = {"LR_lambda": {i: 1 - sum_of_assignment_values[i] for i in range(self.n_cust)}
                        ,"LR_mu": {i:{j: 0 if y[j] - sum_of_assignment_values[i] < 0 and self.LR_mu[i][j] == 0
                                    else y[j] - sum_of_assignment_values[i] for j in self.facilities} for i in range(self.n_cust)}
                        ,"LR_rho": {j: 0 if sum(BnB_node.demands[i]/BnB_node.capacities[j] * x[i][j] for i in range(self.n_cust)) - y[j] < 0 and self.LR_rho[j] == 0 
                                    else sum(BnB_node.demands[i]/BnB_node.capacities[j] * x[i][j] for i in range(self.n_cust)) - y[j]
                                    for j in self.facilities}}
                                
                s_iter_normed = (sum(s_iter["LR_lambda"][i]**2 for i in range(self.n_cust)) 
                                + sum(sum(s_iter["LR_mu"][i][j]**2 for j in self.facilities) for i in range(self.n_cust)) 
                                + sum(s_iter["LR_rho"][j]**2 for j in self.facilities))

                if s_iter_normed > 0.001:

                    alpha_iter = beta * (1.05*BnB_node.primalBound - current_objective_new) / s_iter_normed
                                        
                    if alpha_iter <= 0: break 

                    self.LR_lambda = {i: LR_lambda_old[i] + alpha_iter * s_iter["LR_lambda"][i]
                        for i in range(self.n_cust)}    # relax complete assignment constraints
                    self.LR_mu = {i:{j: max([0,LR_mu_old[i][j] + alpha_iter * s_iter["LR_mu"][i][j]])
                                for j in self.facilities} for i in range(self.n_cust)}   # relax preference constraints
                    self.LR_rho = {j: max([0,LR_rho_old[j] + alpha_iter * s_iter["LR_rho"][j]])
                            for j in self.facilities} # relax capacity constraints                                                   
                    
                else: break
            
            LB_candidates.append({'ID':'Lagrangian','LB': current_objective})

            self.LR_lambda = LR_lambda_best.copy()
            self.LR_mu = LR_mu_best.copy()
            self.LR_rho = LR_rho_best.copy()

            lagrangian_ub,lagrangian_solution = self.greedy_ub((self.required_facilities).union(set(j for j in self.facilities if j not in self.required_facilities if y[j] == 1)),temporary_pref_ordering,flowDict)
            if lagrangian_ub < ub: 
                ub = lagrangian_ub 
                solution = lagrangian_solution

        if ub < treeUB :
            self.UB = ub
            self.incumbent = solution
        else: self.UB = treeUB

        self.LB = max([item['LB'] for item in LB_candidates])
        self.LB_type = [item for item in LB_candidates if item['LB'] == self.LB][0]['ID'] # Get ID of the best LB candidate
        self.status = 'Evaluated'
        return True

    def set_branching_variable(self,temporary_pref_ordering):
        """ Uses a modified strong branching based on reduction of the search space.
            Also fixes variables to 1, if branching to 0 would lead to infeasibility. """
        branchable_facilities = np.array([j for j in self.facilities if j not in self.required_facilities])
        if len(branchable_facilities) >= 0:
            score = {facility:0 for facility in branchable_facilities}
            for facility in branchable_facilities: # "Strong branching"
                for customer in self.customers: # First check effect of opening the facility
                    try: score[facility] += temporary_pref_ordering[customer].index(facility)
                    except ValueError: pass

                test_facilityset,test_prefordering = list(deepcopy(self.facilities)),deepcopy(temporary_pref_ordering) # Then assess effect of closing the facility
                test_facilityset.remove(facility)
                for i in self.customers:
                    try: test_prefordering[i].remove(facility)
                    except: pass
                try: dummy, test_localprefordering = make_feasible(test_facilityset,self.customers,test_prefordering,self.demands,self.capacities)
                except: 
                    temporary_pref_ordering = self.open_facility(facility)
                    continue
                score[facility] += sum(len(test_localprefordering[i]) for i in self.customers) # If possible, compute closing score
            if len(score) > 0:
                self.branching_candidate = min(score, key=score.get) 
                return True
            else:
                return False
        else:
            return False
        
class BnB_tree:
    """ Class for B&B tree"""

    def __init__(self,facilities, LB, UB):
        self.facilities = facilities # This is all facilities considered during B&B, static
        self.LB = LB # Global LB, changes during execution
        self.UB = UB # Global UB, changes during execution
        self.LB_type = 'Initial' # Type of the global LB, changes during execution
        self.nodes = {} # Dict of nodes, changes during execution
        self.incumbent_facilities = facilities
        self.evaluated = 0
        self.infeasible = 0

    def update_LB(self):
        new_LB = min([self.nodes[node].LB for node in self.nodes if self.nodes[node].status == "Evaluated"])
        self.LB_type = ([self.nodes[node].LB_type for node in self.nodes if self.nodes[node].LB == new_LB][0])
        if new_LB > self.LB:
            self.LB = new_LB
            return True
        return False
         
    def update_UB(self):
        incumbents = {key:item.UB for key,item in self.nodes.items() if item.UB != None}
        incumbent_node = self.nodes[min(incumbents, key=incumbents.get)]
        new_UB = incumbent_node.UB
        if new_UB < self.UB:
            self.UB = new_UB
            self.incumbent_facilities = list(sorted(incumbent_node.incumbent))
            return True
        return False
        
    def get_incumbent(self):
        """ Returns the incumbent node, i.e. the node with the lowest upper bound."""
        incumbents = {key:item.UB for key,item in self.nodes.items() if item.UB != None}
        incumbent_node = self.nodes[min(incumbents, key=incumbents.get)]
        return incumbent_node.incumbent
    
    def prune(self,save_RAM=True):
        """ Updates global LB and removes all evaluated nodes whose lower bound is greater than the global upper bound on the objective."""
        remove_by_eval = [node for node in self.nodes if self.nodes[node].status == 'Evaluated' if self.nodes[node].LB > self.UB]
        remove_by_inf = [node for node in self.nodes if self.nodes[node].status == 'Infeasible']
        remove = remove_by_eval + remove_by_inf
        self.evaluated += len(remove_by_eval)
        self.infeasible += len(remove_by_inf)
        if save_RAM: remove += [node for node in self.nodes if self.nodes[node].status == 'Finished']
        for node in remove: del self.nodes[node]

    def primal_GRASP_probing(self,node,iterations=10,shortlist=5):
        """ Use GRASP to try to improve a new incumbent. """
        current_node = self.nodes[node]
        temporary_pref_ordering = {i:[facility for facility in BnB_node.pref_ordering[i][:1+current_node.local_position_of_least_preferred_facility[i]]
                    if facility in current_node.facilities] for i in current_node.customers}
        
        # Do interations many greedy rounds
        for iteration in range(iterations):

            # Initially assign demand based on node facilities
            demandAtFac = {j:[0,[]] for j in current_node.facilities} # Demand at a facility and assigned customers
            for i in range(BnB_node.n_cust):
                demandAtFac[temporary_pref_ordering[i][0]][0] += BnB_node.demands[i]
                demandAtFac[temporary_pref_ordering[i][0]][1].append(i)                

            temp_facilties = deepcopy(current_node.facilities) 
            temp_pref_ordering = deepcopy(temporary_pref_ordering)

            del_fac = set()
            for j in current_node.facilities: # Remove all unused facilities                            
                if demandAtFac[j][0] == 0 and j not in current_node.required_facilities:
                    del temp_facilties[j],demandAtFac[j]                      
                    del_fac.add(j)                    
            for i in range(BnB_node.n_cust):
                j = 0
                while j < len(temp_pref_ordering[i]):
                    if temp_pref_ordering[i][j] in del_fac: temp_pref_ordering[i].pop(j)
                    else: j += 1

            while True: # This is the main GREEDY loop                
                closing_candidates = set(temp_pref_ordering[i][0] # Currently active facilities that are not required to be in the solution
                                         for i in range(BnB_node.n_cust) 
                                         if temp_pref_ordering[i][0] not in current_node.required_facilities
                                         if temp_pref_ordering[i][0] in temp_facilties)
                
                fixed_cost = []
                for j in closing_candidates:
                   fixed_cost.append(BnB_node.fixedCost[j]/demandAtFac[j][0] + sum(BnB_node.assignment_costs[i][j] - BnB_node.assignment_costs[i][temp_pref_ordering[i][1]]
                                                             for i in range(BnB_node.n_cust) if len(temp_pref_ordering[i]) > 1 if temp_pref_ordering[i][0] == j))                             
                
                candidate_list = [x for _, x in sorted(zip(fixed_cost, closing_candidates), key=lambda pair: pair[0])][-shortlist:]
                if len(candidate_list) == 0: return False
                deletion_candidate = choice(candidate_list) # Select one candidate for model
                del temp_facilties[deletion_candidate] 

                for i in range(BnB_node.n_cust): # Remove facility from pref ordering
                    j = 0
                    while j < len(temp_pref_ordering[i]):
                        if temp_pref_ordering[i][j] == deletion_candidate:            
                            temp_pref_ordering[i].pop(j)
                            break
                        else: j += 1
                
                weAreInfeasible = False
                for i in temp_pref_ordering.values():
                    if len(i) == 0:
                        weAreInfeasible = True
                if weAreInfeasible: break
                
                for i in deepcopy(demandAtFac[deletion_candidate][1]): # Assign all customers from deletion_candidate to next favorite facility                     
                    demandAtFac[temp_pref_ordering[i][0]][0] += BnB_node.demands[i]
                    demandAtFac[temp_pref_ordering[i][0]][1].append(i)
                del demandAtFac[deletion_candidate] 

                temp_feasible = False
                while len(temp_facilties) > 0 and not temp_feasible:
                    temp_feasible = True            
                    dobreak = False

                    # Close overloaded facilities
                    reassign,removed = [],[]                                        
                    for j in deepcopy(temp_facilties):
                        if demandAtFac[j][0] > BnB_node.capacities[j]: # If facilities are overloaded, close them
                            reassign += demandAtFac[j][1]
                            del temp_facilties[j],demandAtFac[j]  
                            removed.append(j)
                            temp_feasible = False 
                            if j in current_node.required_facilities: dobreak = True
                    if dobreak: break

                    for i in range(BnB_node.n_cust): # Remove deleted facilities from pref ordering
                        j = 0
                        while j < len(temp_pref_ordering[i]):
                            if temp_pref_ordering[i][j] in removed: temp_pref_ordering[i].pop(j)
                            else: j += 1
                        if len(temp_pref_ordering[i]) == 0: 
                            dobreak= True
                            break
                    if dobreak: break

                    # Reassign customers whose facilities are overloaded
                    for i in reassign:
                        demandAtFac[temp_pref_ordering[i][0]][0] += BnB_node.demands[i]
                        demandAtFac[temp_pref_ordering[i][0]][1].append(i)

                if len(temp_facilties) == 0 or not temp_feasible: break
                if current_node.status != 'Infeasible': 
                    current_assignment_costs = sum(BnB_node.assignment_costs[i][temp_pref_ordering[i][0]] for i in range(current_node.n_cust)) # This is feasible by construction
                    active_facilities = set(temp_pref_ordering[i][0] for i in range(current_node.n_cust))
                    opening_costs = sum(BnB_node.fixedCost[j] for j in active_facilities) + sum(BnB_node.fixedCost[j] for j in current_node.required_facilities if j not in active_facilities)
                    if current_node.UB > opening_costs + current_assignment_costs: current_node.UB = opening_costs + current_assignment_costs
                    if opening_costs + current_assignment_costs < self.UB: 
                        print("UPDATE: Greedy found new incumbent:",round(opening_costs + current_assignment_costs,2),"using",len(active_facilities),"facilities:",sorted(active_facilities))
                        self.UB = opening_costs + current_assignment_costs 
                        self.incumbent_facilities = list(sorted(active_facilities))
                        return True
        return False # No improvement found

def compute_implied_demands(demands,capacities,temporary_pref_ordering,position_of_least_preferred_facility,cannot_assign,facilities,required_facilities=[]):
    """ Computes the implied demands for each facility and customer. """
    customers = list(range(len(demands)))
    if cannot_assign == None:
        cannot_assign = {j:set() for j in facilities}

    def close(facility, customers, required_facilities, local_demandAtFac, local_pref_ordering):
        """ Closes a facility and reassigns customers to their next preferred facility.
            Returns a list of facilities closed to enforce feasibility, updated local demands and pref ordering if closing the facilitiy is possible.
            Otherwise, returns False. """
        
        # Remove facility from preflists
        for i in customers:
            if facility in local_pref_ordering[i]:

                # Easy case: Remove leads to infeasibility
                if len(local_pref_ordering[i]) == 1 or facility in required_facilities:
                    return False

                # Normal case: Remove leads to reassignments
                if local_pref_ordering[i][0] == facility:
                    local_demandAtFac[local_pref_ordering[i][1]][0] += demands[i]
                    local_demandAtFac[local_pref_ordering[i][1]][1].append(i)   

                local_pref_ordering[i].remove(facility)
        
        # Check whether this leads to overloaded facilities
        active_facilities = list(set(local_pref_ordering[i][0] for i in customers)) # All facilities that are currently assigned to customers
        to_close = []
        for j in active_facilities:
            if local_demandAtFac[j][0] > capacities[j] and j != facility: # If facilities are overloaded, close them or terminate
                to_close.append(j)
        return to_close, local_demandAtFac, local_pref_ordering

    # Step 1: Compute demands at facilities and initialize local implied demands
    demandAtFac = {j:[0,[]] for j in facilities} # Demand at at facility and assigned customers
    for i in range(len(temporary_pref_ordering)):
        if temporary_pref_ordering[i] == []:
            return False
        else:
            demandAtFac[temporary_pref_ordering[i][0]][0] += demands[i]
            demandAtFac[temporary_pref_ordering[i][0]][1].append(i)
    
    # NOTE At this point, the implied demands for the current assigned open facility are correctly computed. 
    # Now, assess whether closing assigned_open_facility leads to closing more facilities. 

    # Step 2: Iterate over customers
    for i in customers: 
    
        # Make local copies, since customers generally have different preflists
        local_demandAtFac = deepcopy(demandAtFac) # For each customer, track the effect of closed facilities
        local_pref_ordering = deepcopy(temporary_pref_ordering) # For each customer, track the effect of closed facilities

        while len(local_pref_ordering[i]) > 1: # Iterate over the current list of customer preferences
            assigned_open_facility = local_pref_ordering[i][0] # currently most preferred facility
            to_close = [assigned_open_facility] # List of facilities to close, may be increased through reassignments

            while to_close != []: # While there are facilities to close, close them and reassign customers
                to_remove = to_close.pop(0) # Take first facility to close
                try: 
                    to_close, local_demandAtFac, local_pref_ordering = close(to_remove, customers, required_facilities, local_demandAtFac, local_pref_ordering)
                    if local_demandAtFac[to_remove][0] > capacities[to_remove]:
                        cannot_assign[to_remove].add(i)
                except:
                    to_close = []
                    position_of_least_preferred_facility[i] = BnB_node.pref_ordering[i].index(assigned_open_facility) # Keep only preferred facilities up to the assigned open facility
                    local_pref_ordering[i] = []
                    break
    return position_of_least_preferred_facility, cannot_assign

def compute_implied_demands_in_IP(demands,capacities,temporary_pref_ordering,position_of_least_preferred_facility,cannot_assign,facilities,required_facilities=[]):
    """ Computes the implied demands for each facility and customer. """
    customers = list(range(len(demands)))
    if cannot_assign == None:
        cannot_assign = {j:set() for j in facilities}
    
    pref_ordering = temporary_pref_ordering
    
    def close(facility, customers, required_facilities, local_demandAtFac, local_pref_ordering):
        """ Closes a facility and reassigns customers to their next preferred facility.
            Returns a list of facilities closed to enforce feasibility, updated local demands and pref ordering if closing the facilitiy is possible.
            Otherwise, returns False. """
        
        # Remove facility from preflists
        for i in customers:
            if facility in local_pref_ordering[i]:

                # Easy case: Remove leads to infeasibility
                if len(local_pref_ordering[i]) == 1 or facility in required_facilities:
                    return False

                # Normal case: Remove leads to reassignments
                if local_pref_ordering[i][0] == facility:
                    local_demandAtFac[local_pref_ordering[i][1]][0] += demands[i]
                    local_demandAtFac[local_pref_ordering[i][1]][1].append(i)   

                local_pref_ordering[i].remove(facility)
        
        # Check whether this leads to overloaded facilities
        active_facilities = list(set(local_pref_ordering[i][0] for i in customers)) # All facilities that are currently assigned to customers
        to_close = []
        for j in active_facilities:
            if local_demandAtFac[j][0] > capacities[j] and j != facility: # If facilities are overloaded, close them or terminate
                to_close.append(j)
        return to_close, local_demandAtFac, local_pref_ordering

    # Step 1: Compute demands at facilities and initialize local implied demands
    demandAtFac = {j:[0,[]] for j in facilities} # Demand at at facility and assigned customers
    for i in range(len(temporary_pref_ordering)):
        if temporary_pref_ordering[i] == []:
            return False
        else:
            demandAtFac[temporary_pref_ordering[i][0]][0] += demands[i]
            demandAtFac[temporary_pref_ordering[i][0]][1].append(i)
    
    # NOTE At this point, the implied demands for the current assigned open facility are correctly computed. 
    # Now, assess whether closing assigned_open_facility leads to closing more facilities. 

    # Step 2: Iterate over customers
    for i in customers: 
    
        # Make local copies, since customers generally have different preflists
        local_demandAtFac = deepcopy(demandAtFac) # For each customer, track the effect of closed facilities
        local_pref_ordering = deepcopy(temporary_pref_ordering) # For each customer, track the effect of closed facilities

        while len(local_pref_ordering[i]) > 1: # Iterate over the current list of customer preferences
            assigned_open_facility = local_pref_ordering[i][0] # currently most preferred facility
            to_close = [assigned_open_facility] # List of facilities to close, may be increased through reassignments

            while to_close != []: # While there are facilities to close, close them and reassign customers
                to_remove = to_close.pop(0) # Take first facility to close
                try: 
                    to_close, local_demandAtFac, local_pref_ordering = close(to_remove, customers, required_facilities, local_demandAtFac, local_pref_ordering)
                    if local_demandAtFac[to_remove][0] > capacities[to_remove]:
                        cannot_assign[to_remove].add(i)
                except:
                    to_close = []
                    position_of_least_preferred_facility[i] = pref_ordering[i].index(assigned_open_facility) # Keep only preferred facilities up to the assigned open facility
                    local_pref_ordering[i] = []
                    break    
    return position_of_least_preferred_facility, cannot_assign

def make_feasible(facilities,customers,local_pref_ordering,demands,capacities):
    """ For a given set openFacs, calculate the largest subset of open facilities that results in a feasible assignment.
        Assumes a correct preflist and returns a correct preflist. 
        Note: If you change this, also change the corresponding function in branching.py. """    
    
    # 1. Assign demand
    demandAtFac = {j:[0,[]] for j in facilities} # Demand at at facility and assigned customers
    for i in customers:
        try:
            demandAtFac[local_pref_ordering[i][0]][0] += demands[i]
            demandAtFac[local_pref_ordering[i][0]][1].append(i)
        except:
            return [] # If a customer cannot be assigned to any facility, return infeasibility

    while len(facilities) > 0:

        # 2. Close overloaded facilities
        reassign = []
        for j in facilities.copy():
            if demandAtFac[j][0] > capacities[j]: # If facilities are overloaded, close them or terminate
                reassign += demandAtFac[j][1]
                for i in customers:  # Remove facility from preflists
                    if j in local_pref_ordering[i]: 
                        if len(local_pref_ordering[i]) == 1: return [] # Remove leads to infeasibility
                        local_pref_ordering[i].remove(j)
                facilities.remove(j)
                del demandAtFac[j]  
                
        if reassign == []: break # If no facilities were overloaded

        # 3. Reassign customers whose facilities are overloaded
        for i in reassign:
            demandAtFac[local_pref_ordering[i][0]][0] += demands[i]
            demandAtFac[local_pref_ordering[i][0]][1].append(i)       

    return facilities, local_pref_ordering
