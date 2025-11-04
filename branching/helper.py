import os, json    

def check(solution,name):
    """ Checks whether a solution is consistent with the MILP result on record. """ 
    file_path = os.path.join('results',name) + '_IPsolution.json'

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("Could not find IP solution file.",file_path,os.path.dirname(__file__))
                return False
    else:
        print("Could not find IP solution file.", file_path,os.path.dirname(__file__))
        return False

    if int(data['solution']['UB']) == int(solution['UB']):
        print("Solution matches the recorded solution.")
        if data['solution']['incumbent'] == solution['incumbent']:
            print("Incumbent matches the recorded incumbent.")
        else:
            print("Different incumbents found: IP incumbent:", data['solution']['incumbent'], " vs. given incumbent:", solution['incumbent'])
        return True
    else:
        print("Different objective values found: IP UB:", data['solution']['UB'], " vs. given UB:", solution['UB'])
        return False    
    
def validate(solution,facilities,customers,local_pref_ordering,demands,capacities,fixedCost,assignment_costs):
    """ Validates a solution by checking whether it is feasible and whether its cost is correct. """ 

    print("\nValidating solution with UB",solution['UB'],"and incumbent",solution['incumbent'])

    if [j for j  in solution['incumbent'] if j not in facilities] != []:
        print("The incumbent contains facilities that are not in the given set of facilities:",[j for j  in solution['incumbent'] if j not in facilities],"\n")
        return False
    else:
        facilities = [j for j in facilities if j in solution['incumbent']] # Only consider incumbent facilities
        cost = sum(fixedCost[j] for j in facilities) # Fixed costs of open facilities
        localdemand = {j:0 for j in facilities} # Demand assigned to each facility

        for customer in customers:
            index = 0
            while local_pref_ordering[customer][index] not in facilities:
                index += 1
                continue
            localdemand[local_pref_ordering[customer][index]] += demands[customer]
            cost += assignment_costs[customer][local_pref_ordering[customer][index]]

        print("Total cost of solution is",int(cost))
        if int(cost) != int(solution['UB']):
            print("Cost of solution does not match reported UB of",solution['UB'],"\n")
            return False
        else:
            print("Cost of solution matches reported UB of",int(solution['UB']))

        for facility in facilities:
            if localdemand[facility] > capacities[facility]:
                print("Facility",facility,"is overloaded:",localdemand[facility],">",capacities[facility],"\n")
                return False
        print("All facilities respect their capacities in preference-based assignment.","\n")
        return True