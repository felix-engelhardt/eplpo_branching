#############
### input ###
#############

import os
import numpy as np

def read_instance(i,insType,insTypeLetter,setting,pref_):
    """ Reads in multiple instance variations. 
        i : instance index (natural number)
        insType : 'cap' or something else
        setting : facilities_customers can be used to manually set the number of facilties and customers to be read in
        pref : choose pref type from 0-5 """

    # 1. Paths to instances                
    dirname = os.path.dirname(__file__)
    if insType == "cap":
        if insTypeLetter in {"a","b","c"}:
            dataInstanz = os.path.join(dirname, os.pardir,'data_set_medium_small',str(insType)+str(insTypeLetter)+str(i))
        else: 
            dataInstanz = os.path.join(dirname, os.pardir,'data_set_medium_small',str(insType)+str(i))
        data = read_HRY(dataInstanz)
    else:
        dataInstanz = os.path.join(dirname, os.pardir,'data_set_' + str(insType),'i' + str(insType) + '_' + str(i+1) + ".plc")
        data = readABC(dataInstanz)

    # 2. Read basic input data
    print("Start Initializing \n")
    n_facilities = int(data[0][0]) # number of facilities J
    n_customers = int(data[0][1]) # number of customers I
    
    if insType in {"cap"} :
        demands = np.array([float(value) for value in data[1+n_facilities]][:n_customers])
        capacities = np.array([float(line[0]) for line in data[1:1+n_facilities]])
        fixedCost = np.array([float(line[1]) for line in data[1:1+n_facilities]])
        distances = np.array([[float(el) for el in line[:n_customers]] for line in data[2+n_facilities:]][:n_facilities])
    else:
        fixedCost = np.array([k for line in data[int(n_customers/50+3)+int(n_facilities/50)+1:int(n_customers/50+3)+2*int(n_facilities/50)+1] for k in line if type(k) == float])
        demands = np.array([k for line in data[2:int(n_customers/50+2)] for k in line if type(k) == float])
        capacities = np.array([k for line in data[int(n_customers/50+3):int(n_customers/50+3)+int(n_facilities/50)] for k in line if type(k) == float])
        dist = np.array([l for el in range(2+(int(n_customers/50)+1)+2*(int(n_facilities/50)+1),len(data)) for l in data[el] if type(l) == float])
        distances = np.array([[dist[cust] for cust in range(fac*n_customers,(fac+1)*n_customers)] for fac in range(n_facilities)])
    
    if setting != '': # Optionally, if instances are to be scaled, reduce the in size
        reduced_facilities,reduced_customers = setting.split("_")
        facility_scaling = round((float(reduced_facilities)/n_facilities)*(float(reduced_customers)/n_customers),8)
        n_facilities = min(n_facilities,int(reduced_facilities))
        n_customers = min(n_customers,int(reduced_customers))
        feasibility_factor = 2 if insTypeLetter == "a" else 3 if insTypeLetter == "b" else 4 if insTypeLetter == "c" else 1
        average_demand = sum(d for d in demands)/(feasibility_factor*len(demands)) if insTypeLetter in {"a","b","c"} else 1
        
        demands = demands[:n_customers]
        capacities = [np.ceil(d/average_demand) for d in capacities[:n_facilities]]
        fixedCost = [cost*facility_scaling for cost in fixedCost[:n_facilities]]
        distances = [distances[j][:n_customers] for j in range(0,n_facilities)] # Also consider dist of cust
    
    distances = np.transpose(distances)
    
    # 3. Set up preferences
    preferences = []
    np.random.seed(0)
    if pref_ == 0: # Strict closest assignments
        for i in range(n_customers):
            preferences.append([])
            aux_vec = []
            for j in range(n_facilities):
                u = np.random.uniform(0,0.0001)
                distances[i][j] += u
                aux_vec.append([j,distances[i][j]])
            aux_vec = sorted(aux_vec, key=lambda x: x[1])
            for k in range(len(aux_vec)):
                aux_vec[k].append(k)
            aux_vec = sorted(aux_vec, key=lambda x: x[0])
            for k in aux_vec:
                preferences[i].append(k[1])
    elif pref_ == 1:
        # Strict closest assignment on slightly perturbed data
        for i in range(n_customers):
            preferences.append([])
            M_i = float(max(distances[i]))
            m_i = float(min(distances[i]))
            for j in range(n_facilities):
                u = np.random.uniform(0,1)
                d_ij = float(distances[i][j])
                if d_ij < M_i and m_i < d_ij:   
                    if 0 <= u and u <= (d_ij - m_i)/(M_i - m_i):                           
                        preferences[i].append(m_i + np.sqrt((M_i-m_i)*(d_ij - m_i) * u))
                    if 1 >= u and u >= (d_ij - m_i)/(M_i - m_i):                           
                        preferences[i].append(M_i - np.sqrt((M_i-m_i)*(M_i - d_ij) * (1- u)))
                if m_i == d_ij and m_i < M_i:
                    preferences[i].append(M_i - (M_i - m_i) * np.sqrt(1-u))
                if m_i < M_i and M_i == d_ij:
                    preferences[i].append(m_i + (M_i - m_i) * np.sqrt(u))
                if m_i == M_i and m_i == d_ij:
                    preferences[i].append(u)

    # 4. Check for ties and warn if any appear        
    containsTies = False
    for i in range(n_customers):
        for j in range(n_facilities):
            for k in range(n_facilities):
                if j != k:  
                    if preferences[i][j] == preferences[i][k]:
                        containsTies = True
                        break
            if containsTies:
                break
        if containsTies:
            break
        
    if not containsTies:
        print("There are no ties. Proceed with preference ordering. \n")
        pref_ordering = {i:[None for k in range(n_facilities)] for i in range(n_customers)}
        for i in range(n_customers):
            for j in range(n_facilities):
                aux_set = set()            
                for a in set(range(n_facilities)) - {j}:
                    if preferences[i][a] < preferences[i][j]:
                        aux_set.add(a)
                pref_ordering[i][len(aux_set)] = j
        return capacities, demands, fixedCost, distances, n_customers, n_facilities, pref_ordering
    else:
        print("There are ties. Consider another instance or preference pattern. \n")    
        return False    

def readABC(dataInstanz):
    # Einlesen
    data = []
    
    with open(dataInstanz) as file:
        for line in file:       
            newLine = line.rstrip('\n').split('\t')  
            if newLine != [''] and newLine != [" "]:
                for row in range(len(newLine)):   
                    if newLine[row] != '':
                        newLine[row] = float(newLine[row])
            data.append(newLine)
    return data

def read_HRY(dataInstanz):                        
    res = []
    i = 0

    with open(dataInstanz) as file:
        for line in file:       
            newLine = line.rstrip('\n').split(" ")  
            res.append([])
            for ele in newLine:
                if ele.strip():
                    res[i].append(ele)
            i = i+1

    return res