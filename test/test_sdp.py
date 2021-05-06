import unittest

# from sdp_utils import solveSDP
from quantuminflation.general_tools import *

""" THIS IS MISSING data/ DOES NOT WORK """

# Commented out because it takes long to test
'''
class TestSDPOutput(unittest.TestCase):
    def test_GHZ_known_semiknown(self):
        """
        Comparing with what I get when solving the SDP in MATLAB up to 4 decimal places.
        These lambda values are the same as in Alex's stable version 0.1 before any change to
        check hypergraphs instead of the completed connected graph.
        """

        settings_per_party = [1, 1, 1]
        outcomes_per_party = [2, 2, 2]
        hypergraph = np.array([[0, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])  # Each line is the parties that are fed by a state
        inflation_level_per_source = [2, 2, 2]
        expectation_values = False
        col_structure = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        probability_function = useful_distributions.P_GHZ
        prob_param = 0.828  # Noise
        filename_label = 'test'
        generate_sdp_relaxation(settings_per_party, outcomes_per_party, hypergraph,
                                inflation_level_per_source, expectation_values, col_structure,
                                verbose=1.0, filename_label='', calculate_semiknowns=True)
        final_positions_matrix, known_moments, semiknown_moments = \
                                    get_relaxation_constraints(settings_per_party, outcomes_per_party, hypergraph,
                                       inflation_level_per_source, probability_function, prob_param, filename_label=filename_label)
        #sol, lambdaval = solveSDP('inflationMATLAB_.mat', use_semiknown=True)
        #lambdaval = out[1]

        print("! Takes around 30s/14 steps per SDP. Solving with only fully-known constraints. ")
        # With seminknowns, comparing to MATLAB
        #sol, lambdaval = solveSDP('inflationMATLAB_momentmatrix_and_constraints.mat', use_semiknown=True)
        sol, lambdaval = solveSDP('inflationMATLAB_.mat', use_semiknown=True, verbosity=0)
        print("lambda=", lambdaval, "assert compares to", -0.2078, "to 4 decimal places.")
        assert abs(lambdaval - -0.2078) < 1e-4  # for reference, in MATLAB it is -0.207820

        print("! Takes around 30s/14 steps per SDP. Solving with known and semi-known constraints.")
        # Without semiknowns
        #sol, lambdaval = solveSDP('inflationMATLAB_momentmatrix_and_constraints.mat', use_semiknown=False)
        sol, lambdaval = solveSDP('inflationMATLAB_.mat', use_semiknown=False, verbosity=0)
        print("lambda=", lambdaval, "assert compares to", -0.1755, "to 4 decimal places.")
        assert abs(lambdaval - -0.1755) < 1e-4  # For refernce, in MATLAB it is -0.175523

    def test_GHZ_inf3(self):
        """
        Takes too long to solve the SDP. We will generate the SDP knowing that the result of the maximization of
        the minimum eigenvalue is deterministic, thus we will only check that the final SDP is the same as one of
        we know to be correct.
        """
        print("\nStarting to test GHZ with inflation level 3 with noise v=0.429. \n" +
              "This test calculates the SDP to solve with the constraints, and \n" +
              "checks that it is equal to a reference SDP with the correct solution.\n")
        print("Estimated time: 5-10 minutes.")
        settings_per_party = [1, 1, 1]
        outcomes_per_party = [2, 2, 2]
        hypergraph = np.array([[0, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])  # Each line is the parties that are fed by a state
        inflation_level_per_source = [3, 3, 3]
        expectation_values = False
        col_structure = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        probability_function = useful_distributions.P_GHZ
        prob_param = 0.429  # Noise
        filename_label = 'test'
        generate_sdp_relaxation(settings_per_party, outcomes_per_party, hypergraph,
                                inflation_level_per_source, expectation_values, col_structure,
                                verbose=1.0, filename_label='', calculate_semiknowns=True)
        test_final_positions_matrix, test_known_moments, test_semiknown_moments = \
            get_relaxation_constraints(settings_per_party, outcomes_per_party, hypergraph,
                                       inflation_level_per_source, probability_function, prob_param, filename_label=filename_label)

        # Import correct ones
        correct_final_positions_matrix = loadmat("test/test_data/inf3_out222_local1_P_GHZ_v0429/inflationMomentMat.mat")['G']
        correct_known_moments = loadmat("test/test_data/inf3_out222_local1_P_GHZ_v0429/inflationKnownMoments.mat")['known_moments']
        correct_semiknown_moments = loadmat("test/test_data/inf3_out222_local1_P_GHZ_v0429/inflationProptoMoments.mat")['propto']

        assert np.allclose(test_final_positions_matrix, correct_final_positions_matrix), "Not the same as the reference."
        assert np.allclose(test_known_moments, correct_known_moments), "Not the same as the reference."
        assert np.allclose(test_semiknown_moments, correct_semiknown_moments), "Not the same as the reference."

    def test_W_inf2(self):
        """
        Takes too long to solve the SDP. We will generate the SDP knowing that the result of the maximization of
        the minimum eigenvalue is deterministic, thus we will only check that the final SDP is the same as one of
        we know to be correct.
        """
        print("\nStarting to test W distrib. with inf. level 2 with noise v=0.81 and \n"
              "local level 2 truncated to operators with containing at most 4 factors. \n" +
              "This test calculates the SDP to solve with the constraints, and \n" +
              "checks that it is equal in formulation to a reference SDP which we know \n"
              "to give the correct solution.\n")
        print("Estimated time: around 5 minutes.")
        settings_per_party = [1, 1, 1]
        outcomes_per_party = [2, 2, 2]
        hypergraph = np.array([[0, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])  # Each line is the parties that are fed by a state
        inflation_level_per_source = [2, 2, 2]
        expectation_values = False

        # local leve 2 star: only up to products of 4 terms
        col_structure = [[],
                         [0], [1], [2],
                         [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2],
                         [0, 1, 2],
                         [0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 2, 2]]
        probability_function = useful_distributions.P_W
        prob_param = 0.81  # Noise
        filename_label = 'test'
        generate_sdp_relaxation(settings_per_party, outcomes_per_party, hypergraph,
                                inflation_level_per_source, expectation_values, col_structure,
                                verbose=1.0, filename_label='', calculate_semiknowns=True)
        test_final_positions_matrix, test_known_moments, test_semiknown_moments = \
            get_relaxation_constraints(settings_per_party, outcomes_per_party, hypergraph,
                                       inflation_level_per_source, probability_function, prob_param, filename_label=filename_label)

        # Import correct ones
        correct_final_positions_matrix = loadmat("test/test_data/inf2_out222_local2_star_P_W_v081/inflationMomentMat.mat")['G']
        correct_known_moments = loadmat("test/test_data/inf2_out222_local2_star_P_W_v081/inflationKnownMoments.mat")['known_moments']
        correct_semiknown_moments = loadmat("test/test_data/inf2_out222_local2_star_P_W_v081/inflationProptoMoments.mat")['propto']

        assert np.allclose(test_final_positions_matrix, correct_final_positions_matrix), "Not the same as the reference."
        assert np.allclose(test_known_moments, correct_known_moments), "Not the same as the reference."
        assert np.allclose(test_semiknown_moments, correct_semiknown_moments), "Not the same as the reference."


    def test_Mermin_inf2(self):
        """
        Takes too long to solve the SDP. We will generate the SDP knowing that the result of the maximization of
        the minimum eigenvalue is deterministic, thus we will only check that the final SDP is the same as one of
        we know to be correct.
        """
        print("\nStarting to test Mermin distrib. with inf. level 2 with noise v=0.81 and \n"
              "local level 2 truncated to operators with containing at most 4 factors. \n" +
              "This test calculates the SDP to solve with the constraints, and \n" +
              "checks that it is equal in formulation to a reference SDP which we know \n"
              "to give the correct solution.\n")
        print("Estimated time: 5-10 minutes.")
        settings_per_party = [2, 2, 2]
        outcomes_per_party = [2, 2, 2]
        hypergraph = np.array([[0, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])  # Each line is the parties that are fed by a state
        inflation_level_per_source = [2, 2, 2]
        expectation_values = False

        # local leve 2 star: only up to products of 4 terms
        # Union of S2 and Local 1
        col_structure = [[], [0], [1], [2], [0,0], [0,1], [0,2], [1,1], [1,2], [2,2], [0,1,2]]
        probability_function = useful_distributions.P_Mermin
        prob_param = 0.51  # Noise
        filename_label = 'test'
        generate_sdp_relaxation(settings_per_party, outcomes_per_party, hypergraph,
                                inflation_level_per_source, expectation_values, col_structure,
                                verbose=1.0, filename_label='', calculate_semiknowns=True)
        test_final_positions_matrix, test_known_moments, test_semiknown_moments = \
            get_relaxation_constraints(settings_per_party, outcomes_per_party, hypergraph,
                                       inflation_level_per_source, probability_function, prob_param, filename_label=filename_label)

        # Import correct ones
        correct_final_positions_matrix = loadmat("test/test_data/inf3_out222_in222_local1_S2_P_mermin_v051/inflationMomentMat.mat")['G']
        correct_known_moments = loadmat("test/test_data/inf3_out222_in222_local1_S2_P_mermin_v051/inflationKnownMoments.mat")['known_moments']
        correct_semiknown_moments = loadmat("test/test_data/inf3_out222_in222_local1_S2_P_mermin_v051/inflationProptoMoments.mat")['propto']

        assert np.allclose(test_final_positions_matrix, correct_final_positions_matrix), "Not the same as the reference."
        assert np.allclose(test_known_moments, correct_known_moments), "Not the same as the reference."
        assert np.allclose(test_semiknown_moments, correct_semiknown_moments), "Not the same as the reference."


    def test_Salman_u2_085(self):
        """
        Takes too long to solve the SDP. We will generate the SDP knowing that the result of the maximization of
        the minimum eigenvalue is deterministic, thus we will only check that the final SDP is the same as one of
        we know to be correct.
        """
        print("\nStarting to test W distrib. with inf. level 2 with noise v=0.81 and \n"
              "local level 2 truncated to operators with containing at most 4 factors. \n" +
              "This test calculates the SDP to solve with the constraints, and \n" +
              "checks that it is equal in formulation to a reference SDP which we know \n"
              "to give the correct solution.\n")
        print("Estimated time: ~30 minutes.")
        settings_per_party = [1, 1, 1]
        outcomes_per_party = [4, 4, 4]
        hypergraph = np.array([[0, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])  # Each line is the parties that are fed by a state
        inflation_level_per_source = [2, 2, 2]
        expectation_values = False

        # local leve 2 star: only up to products of 4 terms
        # Union of S2 and Local 1
        col_structure = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        probability_function = P_Salman
        prob_param = 0.85  # u2 param
        filename_label = 'GHZ_inf3'
        generate_sdp_relaxation(settings_per_party, outcomes_per_party, hypergraph,
                                inflation_level_per_source, expectation_values, col_structure,
                                verbose=1.0, filename_label='', calculate_semiknowns=True)
        test_final_positions_matrix, test_known_moments, test_semiknown_moments = \
            get_relaxation_constraints(settings_per_party, outcomes_per_party, hypergraph,
                                       inflation_level_per_source, probability_function, prob_param, filename_label='')

        # Import correct ones
        correct_final_positions_matrix = \
        loadmat("test/test_data/inf2_out444_local1_P_Salman_u2_085/inflationMomentMat.mat")['G']
        correct_known_moments = \
        loadmat("test/test_data/inf2_out444_local1_P_Salman_u2_085/inflationKnownMoments.mat")['known_moments']
        correct_semiknown_moments = \
        loadmat("test/test_data/inf2_out444_local1_P_Salman_u2_085/inflationProptoMoments.mat")['propto']

        assert np.allclose(test_final_positions_matrix,
                           correct_final_positions_matrix), "Not the same as the reference."
        assert np.allclose(test_known_moments, correct_known_moments), "Not the same as the reference."
        assert np.allclose(test_semiknown_moments, correct_semiknown_moments), "Not the same as the reference."

'''
