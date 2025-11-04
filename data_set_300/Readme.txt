These data files are set TBED1 in Guastaroba G. and Speranza M.G. A Heuristic for BILP Problems: The Single Source Capacitated Facility Location Problem, currently under review in the European Journal of Operational Research.
This set of instances was originally proposed for the Multi-Source Capacitated Facility Location Problem in Avella, P. and Boccia, M., ``A Cutting Plane Algorithm for the Capacitated Facility Location Problem'', Computational Optimization and Applications, 43(1), 2009, 39--65., and publicly available at http://www.ing.unisannio.it/boccia/CFLP.htm.

There are currently 100 data files. The format of these data files is:

|J| |I|
d_1 d_2 d_3 … d_|I|
s_1 s_2 ... s_|J|
f_1 f_2 ... f_|J|
c_{11} c_{12} c_{13} … c_{1|I|} 
c_{21} c_{22} c_{23} … c_{2|I|} 
   ...    ...    ...   ....
c_{|J|1} c_{|J|2} c_{|J|3} … c_{|J||I|} 

where:

|J| is the number of potential facility locations;
|I| is the number of customers;
d_i (i=1,...,|I|) is the demand of customer i;
s_j (j=1,...,|J|) is the capacity of facility j;
f_j (j=1,...,|J|) is the fixed cost of opening facility j;
c_{ji} (j=1,...,|J|), (i=1,...,|I|) is the cost of supplying one unit of demand of customer i from facility j. 