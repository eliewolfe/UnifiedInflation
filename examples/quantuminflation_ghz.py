import sys, os
# Locate the script in UnifiedInflation/examples and add to path one folder before, that is, UnifiedInflation/
# in order to be able to import quantuminflation
# ! Note: I found online that "__file__" sometimes can be problematic, So I'm using the solution provided in
# https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python?lq=1
from inspect import getsourcefile
from os.path import abspath
cws = abspath(getsourcefile(lambda:0))
cws = os.sep.join(cws.split(os.sep)[:-1])  # Remove the script filename to get the directory of the script
cws = cws + os.sep + os.pardir  # Go one folder above UnifiedInflation/examples -> UnifiedInflation/ 
sys.path.append(cws) 
###############################################################################

import numpy as np

import quantuminflation.useful_distributions
from quantuminflation.general_tools import generate_sdp_relaxation, helper_extract_constraints
from quantuminflation.sdp_utils import solveSDP

if __name__ == '__main__':  # Necessary for parallel computation, used in ncpol2sdpa
    settings_per_party = [1, 1, 1]
    outcomes_per_party = [2, 2, 2]
    hypergraph = np.array([[0, 1, 1],
                           [1, 1, 0],
                           [1, 0, 1]])  # Each line is the parties that are fed by a state
    inflation_level_per_source = [2, 2, 2]
    expectation_values = False  # Currently doesn't work

    col_structure = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    probability_function = quantuminflation.useful_distributions.P_GHZ
    prob_param = 0.828  # Noise
    filename_label = 'test'

    vars_dic = generate_sdp_relaxation(settings_per_party,
                                       outcomes_per_party,
                                       hypergraph,
                                       inflation_level_per_source,
                                       expectation_values,
                                       col_structure,
                                       verbose=1.0, filename_label='', calculate_semiknowns=True)

    final_positions_matrix, known_moments, semiknown_moments, symbolic_variables_to_be_given, variable_dict \
        = helper_extract_constraints(settings_per_party, outcomes_per_party, hypergraph, inflation_level_per_source,
                                     probability_function, prob_param, filename_label='')
    sol, lambdaval = solveSDP('inflationMATLAB_.mat', use_semiknown=True, verbosity=1)
    print(lambdaval)
