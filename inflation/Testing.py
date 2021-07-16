"""
TESTING
Open question: Why are there duplicate columns in the triangle scenario?
"""
from __future__ import absolute_import
import numpy as np
from igraph import Graph
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from inf_classes_w_numb_beta import *

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
# import CausInf.inflation.infgraph as old
# import CausInf.inflation.bora_specialized.classes as older
import timeit
if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property
from linear_program_options.moseklp import InfeasibilityCertificate
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
# from old_versions.function_based.inflationmatrix import InflationMatrixFromGraph, NumericalAndSymbolicVectorsFromGraph
from collections import Counter
import functools

def ListOfBitStringsToListOfIntegers(list_of_bitstrings):
    return list(map(lambda s: int(s,2),list_of_bitstrings))
def UniformDistributionFromSupport(list_of_bitstrings):
    numvar = max(map(len,list_of_bitstrings))
    numevents = len(list_of_bitstrings)
    data = np.zeros(2 ** numvar)
    data[ListOfBitStringsToListOfIntegers(list_of_bitstrings)] = 1/numevents
    return data

def oldest_code_solver(rawgraph, rawdata, card, inflation_order):
    
    inf = InflationMatrixFromGraph(rawgraph, inflation_order, card, extra_expressible=False)
    num_b, sym_b = NumericalAndSymbolicVectorsFromGraph(rawgraph, rawdata, inflation_order, card, extra_expressible=False)
    inequality = InfeasibilityCertificate(inf, num_b)
    
    return [num_b,sym_b,inf]
    
def older_code_solver(rawgraph, rawdata, card, inflation_order):
    
    inf = older.InflationProblem(rawgraph,rawdata,card,inflation_order)
    num_b,sym_b = inf.numeric_and_symbolic_b()
    mat = inf.InflationMatrix()
    inequality = older.InflationLP(rawgraph, rawdata, card, inflation_order,extra_ex = False , solver = 'moseklp').Inequality()
    
    return [num_b,sym_b,mat]


def old_code_solver(rawgraph, rawdata, card, inflation_order):

     start_mat = timeit.default_timer()
     inf = old.InflationProblem(rawgraph,rawdata,card,inflation_order)

     sym_b = inf.symbolic_b
     mat_shape = inf.inflation_matrix.shape
     stop_mat = timeit.default_timer()
     sym_b = inf.symbolic_b
     print(sym_b)
     print(mat_shape)

     start_lp = timeit.default_timer()
     inequality = old.InflationLP(rawgraph, rawdata, card, inflation_order, solver = 'moseklp').Inequality()
     stop_lp = timeit.default_timer()
     
     AMatrix = inf.expressible_sets[0].AMatrices
     
     print('It took ',stop_mat-start_mat,' seconds to compute the inflation matrix.')
     print('It took ',stop_lp-start_lp,' seconds to run the linear program.')

     return [inf.numeric_b,inequality,sym_b,inf.inflation_matrix, AMatrix]

def old_code_comparison1(rawgraph, rawdata, card, inflation_order):

     start_mat = timeit.default_timer()
     inf = old.InflationProblem(rawgraph,rawdata,card,inflation_order)

     sym_b = inf.symbolic_b
     mat_shape = inf.inflation_matrix.shape
     stop_mat = timeit.default_timer()
     sym_b = inf.symbolic_b

     start_lp = timeit.default_timer()
     num_b,sym_b = NumericalAndSymbolicVectorsFromGraph(rawgraph, rawdata, inflation_order, card, extra_expressible=False)
     temp = []
     for i in range(len(sym_b)):
         if sym_b[i][0] == '2':
             temp.append(i)
     num_b[temp] =  num_b[temp]/2
     inequality = InfeasibilityCertificate(inf.inflation_matrix, num_b)
     #print(np.dot(inequality['x'], num_b))
     inequality = old.InflationLP(rawgraph, rawdata, card, inflation_order, solver = 'moseklp').Inequality()
     stop_lp = timeit.default_timer()
     
     if Counter(num_b)==Counter(inf.numeric_b):
         print('The tallies of the numeric bs are THE SAME')
     else:
         print('The tallies of the numeric bs are NOT THE SAME')
         
     if np.array_equal(num_b,inf.numeric_b):
         print('The numeric bs are THE SAME')
     else:
         print('The numeric bs are NOT THE SAME')

     return [num_b,sym_b,inf.inflation_matrix]

def old_code_comparison2(rawgraph, rawdata, inflation_order, hypergraph, directed_structure, outcome_cardinalities,
                            private_setting_cardinalities):
     card = list(outcome_cardinalities)
     start_mat = timeit.default_timer()
     inf = old.InflationProblem(rawgraph,rawdata,card,inflation_order)

     sym_b = inf.symbolic_b
     mat_shape = inf.inflation_matrix.shape
     stop_mat = timeit.default_timer()

     start_lp = timeit.default_timer()
     
     solver = 'moseklp'
     num_b = inflation_LP(hypergraph, inflation_order, directed_structure, outcome_cardinalities,
                            private_setting_cardinalities,rawdata,solver).numeric_b
     
     #inequality = InfeasibilityCertificate(inf.inflation_matrix, num_b)
     stop_lp = timeit.default_timer()

     if Counter(num_b)==Counter(inf.numeric_b):
         print('The tallies of the numeric bs are THE SAME')
     else:
         print('The tallies of the numeric bs are NOT THE SAME')
         
     if np.array_equal(num_b,inf.numeric_b):
         print('The numeric bs are THE SAME')
     else:
         print('The numeric bs are NOT THE SAME')
         
     return [inf.numeric_b,sym_b,inf.inflation_matrix]

def comparison_of_infmats1(rawgraph, rawdata, card, inflation_order):
    
    infmat_old = old.InflationProblem(rawgraph,rawdata,card,inflation_order).inflation_matrix
    infmat_oldest = InflationMatrixFromGraph(rawgraph, inflation_order, card, extra_expressible=False)
    
    old_mat_as_array=infmat_old.transpose().toarray()
    oldest_mat_as_array=infmat_oldest.transpose().toarray()
    
    old_list=[]
    oldest_list=[]
    for i in range(len(old_mat_as_array)):
        old_list.append(str(old_mat_as_array[i]))
        oldest_list.append(str(oldest_mat_as_array[i]))

    comp = Counter(old_list) == Counter(oldest_list)
    
    if comp:
        print('The tallies of the two inf mats are THE SAME')
    else:
        print('The tallies of the two inf mats are NOT THE SAME')

def comparison_of_infmats2(rawgraph, rawdata, inflation_order, hypergraph, directed_structure, outcome_cardinalities,
                            private_setting_cardinalities):
    card = list(outcome_cardinalities)
    infmat_old = old.InflationProblem(rawgraph,rawdata,card,inflation_order).inflation_matrix
    infmat_new = inflation_problem(hypergraph, inflation_order, directed_structure, outcome_cardinalities,
                            private_setting_cardinalities).inflation_matrix
    
    old_mat_as_array=infmat_old.transpose().toarray()
    new_mat_as_array=infmat_new.transpose().toarray()
    
    old_list=[]
    new_list=[]
    for i in range(len(old_mat_as_array)):
        old_list.append(str(old_mat_as_array[i]))
        new_list.append(str(new_mat_as_array[i]))

    comp = Counter(old_list) == Counter(new_list)
    
    if comp:
        print('The tallies of the two inf mats are THE SAME')
    else:
        print('The tallies of the two inf mats are NOT THE SAME')

def new_code_solver(hypergraph, inflation_orders, directed_structure, outcome_cardinalities,
                                private_setting_cardinalities,rawdata):
    
    start_mat = timeit.default_timer()
    inf = inflation_problem(hypergraph, inflation_orders, directed_structure, outcome_cardinalities,
                            private_setting_cardinalities)
    
    mat_shape = inf.inflation_matrix.shape
    stop_mat = timeit.default_timer()
    sym_b = inf.symbolic_b
    print(sym_b)
    print(mat_shape)
    
    solver = 'moseklp'
    start_lp = timeit.default_timer()
    inequality = inflation_LP(hypergraph, inflation_orders, directed_structure, outcome_cardinalities,
                            private_setting_cardinalities,rawdata,solver)
    inequality.Inequality()
    stop_lp = timeit.default_timer()
    
    print('It took ',stop_mat-start_mat,' seconds to compute the inflation matrix.')
    print('It took ',stop_lp-start_lp,' seconds to run the linear program.')
    return [sym_b,inequality,inf.inflation_matrix, inf.AMatrix]

class testing:
    
    def __init__(self):
        
        """
        -----------------
        TRIANGLE SCENARIO
        -----------------
        """
        
        #NEW CODE
        
        self.triangle_hypergraph = np.array([[1,1,0],[0,1,1],[1,0,1]])
        self.triangle_directed_structure = np.array([[0,0,0],[0,0,0],[0,0,0]])
        self.triangle_outcome_cardinalities = (4,4,4)
        self.triangle_private_setting_cardinalities = (1,1,1)
        self.triangle_inflation_orders = [2,2,2]
        self.triangle_rawdata = np.array([0.12199995751046305, 0.0022969343799089472, 0.001748319476328954, 3.999015242496535e-05, 0.028907881434196828, 0.0005736087488455967, 0.0003924033706699725, 1.1247230369521505e-05, 0.0030142577390317635, 0.09234476010282468, 4.373922921480586e-05, 0.0014533921021948346, 0.0007798079722868244, 0.024091567451515063, 1.1247230369521505e-05, 0.0003849052170902915, 0.020774884184769502, 0.000396152447459813, 0.0003049249122403608, 4.998769053120669e-06, 0.10820335492385, 0.0020794879260981982, 0.0015546171755205281, 2.4993845265603346e-05, 0.0006260958239033638, 0.020273757587194154, 7.498153579681003e-06, 0.0003374169110856452, 0.0028942872817568676, 0.08976414557915113, 2.624353752888351e-05, 0.0012984302615480939, 0.002370666223442477, 4.7488306004646356e-05, 0.0999928767540993, 0.001957018084296742, 0.0006198473625869629, 8.747845842961171e-06, 0.02636975644747481, 0.0005198719815245496, 1.4996307159362007e-05, 0.000403650601039494, 0.0005498645958432735, 0.017359475229224805, 7.123245900696953e-05, 0.002346922070440154, 0.0033754188031197316, 0.10295964618712641, 0.00038740460161685187, 7.498153579681003e-06, 0.01608353942841575, 0.000306174604503641, 0.0021319750011559654, 4.248953695152569e-05, 0.09107007399427891, 0.001860791780024169, 5.998522863744803e-05, 0.0018395470115484063, 0.002570616985567304, 0.0766411271224461, 1.874538394920251e-05, 0.00048238121362614454, 0.0006410921310627258, 0.020223769896662948])
        
        #OLD CODE
        
        self.triangle_rawgraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
        self.triangle_card = 4
        self.triangle_inflation_order = 2
    
        """
        -----------------
        EVANS 14A SCENARIO
        -----------------
        """
        
        #NEW CODE
        
        self.evans14a_hypergraph = np.array([[1,0,1,0],[1,1,0,1],[0,1,1,1]])
        self.evans14a_directed_structure = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
        self.evans14a_outcome_cardinalities = (2,2,2,2)
        self.evans14a_private_setting_cardinalities = (1,1,1,1)
        self.evans14a_inflation_orders = [2,2,2]
        self.evans14a_rawdata = UniformDistributionFromSupport(['0000','1001','1111'])
        
        #OLD CODE

        self.evans14a_rawgraph = Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
        self.evans14a_card = 2
        self.evans14a_inflation_order = 2
    
        """
        -----------------
        EVANS 14B SCENARIO
        -----------------
        """
        
        #NEW CODE
        #TOO MANY COLUMNS AND ROWS
        self.evans14b_hypergraph = np.array([[1,0,1,0],[0,1,1,1],[1,0,0,1]])
        self.evans14b_directed_structure = np.array([[0,1,0,0],[0,0,0,1],[0,0,0,1],[0,0,0,0]])
        self.evans14b_outcome_cardinalities = (2,2,2,2)
        self.evans14b_private_setting_cardinalities = (1,1,1,1)
        self.evans14b_inflation_orders = [2,2,2]
        self.evans14b_rawdata = UniformDistributionFromSupport(['1000','1001','1111'])
        
        #OLD CODE

        self.evans14b_rawgraph = Graph.Formula("U1->A:C,U2->B:C:D,U3->A:D,A->B,B:C->D")
        self.evans14b_card = 2
        self.evans14b_inflation_order = 2
        
    @cached_property
    def new_code_triangle_scenario(self):
        return new_code_solver(self.triangle_hypergraph, self.triangle_inflation_orders, self.triangle_directed_structure, self.triangle_outcome_cardinalities,
                                self.triangle_private_setting_cardinalities,self.triangle_rawdata)

    @cached_property
    def new_infprob_triangle_scenario(self):
        return inflation_problem(self.triangle_hypergraph, self.triangle_inflation_orders, self.triangle_directed_structure, self.triangle_outcome_cardinalities,
                                self.triangle_private_setting_cardinalities)

    @cached_property
    def old_code_triangle_scenario(self):
        return old_code_solver(self.triangle_rawgraph, self.triangle_rawdata, self.triangle_card, self.triangle_inflation_order)
    
    @cached_property
    def old_code_comparison1_triangle_scenario(self):
        return old_code_comparison1(self.triangle_rawgraph, self.triangle_rawdata, self.triangle_card, self.triangle_inflation_order)
    
    @cached_property
    def old_code_comparison2_triangle_scenario(self):
        return old_code_comparison2(self.triangle_rawgraph, self.triangle_rawdata, self.triangle_inflation_orders,self.triangle_hypergraph, self.triangle_directed_structure, self.triangle_outcome_cardinalities,
                                self.triangle_private_setting_cardinalities )
    
    @cached_property
    def comparison_of_infmats1_triangle_scenario(self):
        return comparison_of_infmats1(self.triangle_rawgraph, self.triangle_rawdata, self.triangle_card, self.triangle_inflation_order)
    
    @cached_property
    def comparison_of_infmats2_triangle_scenario(self):
        return comparison_of_infmats2(self.triangle_rawgraph, self.triangle_rawdata, self.triangle_inflation_orders,self.triangle_hypergraph, self.triangle_directed_structure, self.triangle_outcome_cardinalities,
                                self.triangle_private_setting_cardinalities )
    
    @cached_property
    def oldest_code_triangle_scenario(self):
        return oldest_code_solver(self.triangle_rawgraph, self.triangle_rawdata, self.triangle_card, self.triangle_inflation_order)
    
    @cached_property
    def new_code_evans14a_scenario(self):
        return new_code_solver(self.evans14a_hypergraph, self.evans14a_inflation_orders, self.evans14a_directed_structure, self.evans14a_outcome_cardinalities,
                                self.evans14a_private_setting_cardinalities,self.evans14a_rawdata)

    @cached_property
    def new_code_evans14a_scenario_no_inflation(self):
        return inflation_problem(self.evans14a_hypergraph, [2,2,2], self.evans14a_directed_structure, self.evans14a_outcome_cardinalities,
                                self.evans14a_private_setting_cardinalities)

    @cached_property
    def new_infprob_evans14a_scenario(self):
        return inflation_problem(self.evans14a_hypergraph, self.evans14a_inflation_orders, self.evans14a_directed_structure, self.evans14a_outcome_cardinalities,
                                self.evans14a_private_setting_cardinalities)
    
    @cached_property
    def old_code_evans14a_scenario(self):
        return old_code_solver(self.evans14a_rawgraph, self.evans14a_rawdata, self.evans14a_card, self.evans14a_inflation_order)
    
    @cached_property
    def new_code_evans14b_scenario(self):
        return new_code_solver(self.evans14b_hypergraph, self.evans14b_inflation_orders, self.evans14b_directed_structure, self.evans14b_outcome_cardinalities,
                                self.evans14b_private_setting_cardinalities,self.evans14b_rawdata)
    
    @cached_property
    def old_code_evans14b_scenario(self):
        return old_code_solver(self.evans14b_rawgraph, self.evans14b_rawdata, self.evans14b_card, self.evans14b_inflation_order)


if __name__ == '__main__':
    """
    t0 = testing().new_code_evans14a_scenario_no_inflation
    amat = t0.AMatrix.copy()
    amat.sort(axis=0)
    #amat = amat[:,np.lexsort(amat)]
    #
    # # from itertools import chain
    # # from more_itertools import unique_everseen, unique_justseen
    # # start = timeit.default_timer()
    # # fmat1 = np.fromiter(chain.from_iterable(unique_everseen(amat.T, key=tuple)),int).reshape((-1,len(amat))).T;
    # # print(timeit.default_timer()-start)
    # # start = timeit.default_timer()
    # # fmat2 = np.fromiter(chain.from_iterable(unique_justseen(amat.T, key=tuple)),int).reshape((-1,len(amat))).T;
    # # print(timeit.default_timer()-start)
    start = timeit.default_timer()
    fmat3, idx, inv, counts = np.unique(amat, axis=1, return_index=True, return_inverse=True, return_counts=True)
    fmat3 = fmat3[:,np.lexsort(fmat3)]
    print(timeit.default_timer() - start)

    duplicate_marks = idx[counts>1]
    first_duplicate_set = np.flatnonzero(inv == inv[duplicate_marks[0]])
    print("Number of columns participating in duplicate sets: ",duplicate_marks.shape)
    print("Max number of duplicate columns in a set: ", counts.max())
    print("A set of duplicate orbits: ",first_duplicate_set)
    print(t0.AMatrix[:, first_duplicate_set])
    print(t0.column_orbits[:, first_duplicate_set])
    print("The meaning of these orbits: ", np.stack(np.unravel_index(t0.column_orbits[:, first_duplicate_set], t0.inflated_unpacked_cardinalities), axis=-1))
    # print("So we can see that column 5 and column 17 are being treated as identical with respect to rows, even though they are in different orbits.")
    # print("this is their meaning on the inflation graph")
    # print(np.stack(np.unravel_index([5, 80], t0.inflated_unpacked_cardinalities), axis=-1))
    # print(np.stack(np.unravel_index([17, 68], t0.inflated_unpacked_cardinalities), axis=-1))
    # print(t0.expressible_sets[0].partitioned_tuple_form)

    
    # print(testing().new_code_evans14a_scenario_no_inflation.AMatrix.shape)
    """
    t1 = testing().new_code_triangle_scenario
    t7 = testing().oldest_code_triangle_scenario
    # t2 = testing().old_code_triangle_scenario
    #t8 = testing().older_code_triangle_scenario
    #t9 = testing().old_code_comparison1_triangle_scenario
    #t10 = testing().old_code_comparison2_triangle_scenario
    #print(np.array_equal(t1[-2].toarray(),t2[-2].toarray()))
    
    """
    def sort_amat(rawamat):
        amat = rawamat.copy()
        amat.sort(axis=0)
        amat = amat[:, np.lexsort(amat)]
        
        return amat
    
    old_amat = sort_amat(t2[-1])
    new_amat = sort_amat(t1[-1])
    
    print(np.array_equal(old_amat,new_amat))
    """
    #t11 = testing().comparison_of_infmats2_triangle_scenario
    #t12 = testing().comparison_of_infmats1_triangle_scenario
    
    #t3 = testing().new_code_evans14a_scenario
    #t4 = testing().old_code_evans14a_scenario
    
    #t5 = testing().new_code_evans14b_scenario #MEMORY CRASH
    #t6 = testing().old_code_evans14b_scenario
"""
  Interior-point solution summary
  Problem status  : PRIMAL_AND_DUAL_FEASIBLE
  Solution status : OPTIMAL
  Primal.  obj: -3.0799707845e-05   nrm: 1e+00    Viol.  con: 5e-10    var: 0e+00  
  Dual.    obj: -3.1099286914e-05   nrm: 3e-03    Viol.  con: 2e-19    var: 5e-10  
  Problem status: prosta.prim_and_dual_feas
  Solution status: solsta.optimal

    Coefficient Range: [-0.0025598145716492215, 0.005119923417404859]
    Distribution Compatibility Status: INCOMPATIBLE
"""

"""
Interior-point solution summary
  Problem status  : PRIMAL_AND_DUAL_FEASIBLE
  Solution status : OPTIMAL
  Primal.  obj: -3.0799707845e-05   nrm: 1e+00    Viol.  con: 5e-10    var: 0e+00  
  Dual.    obj: -3.1099286914e-05   nrm: 3e-03    Viol.  con: 2e-19    var: 5e-10  
Problem status: prosta.prim_and_dual_feas
Solution status: solsta.optimal

Coefficient Range: [-0.0025598145716492215, 0.005119923417404859]
-3.079970784503949e-05 ------------
Distribution Compatibility Status: COMPATIBLE
"""


    
    