import copy
import pickle
from itertools import product, permutations
from time import time

from typing import Dict, List, Callable, Tuple
import sympy

import networkx as nx
import numpy as np
import sympy as sp
from ncpol2sdpa import generate_variables, generate_operators, projective_measurement_constraints, flatten, \
    SdpRelaxation
from scipy.io import savemat
from sympy import S
from tqdm import tqdm


def generate_sdp_relaxation(settings_per_party: List[int],
                            outcomes_per_party: List[int],
                            hypergraph: np.ndarray,
                            inflation_level_per_source: List[int],
                            expectation_values: bool,
                            col_structure: List[List],
                            verbose: int = 1,
                            filename_label: str = '',
                            calculate_semiknowns: bool = True) -> Dict:
    """
    Generate the SDP for the scenario and save it to file. Output many intermediate variables useful for debugging.

    :param settings_per_party: Number of inputs of each party.
    :param outcomes_per_party: Number of outcomes of each party.
    :param hypergraph: Causal structure hypothesis.
    :param inflation_level_per_source: Number of copies of each source.
    :param expectation_values: (Unused flag)
    :param col_structure: Choice of operators for the moment matrix. Manual way to set the "NPA level".
    :param verbose: How much info to show.
    :param filename_label: Name of the filename where we
    :param calculate_semiknowns: If True, use non-certificate constraints.
    :return: Dictionary with many intermediate variables which are useful for debugging. The relevant info is saved to file.
    """
    n_sources = hypergraph.shape[0]
    n_parties = len(settings_per_party)

    ## Get measurement operators
    measurements, substitutions, names = generate_parties(hypergraph,
                                                          settings_per_party,
                                                          outcomes_per_party,
                                                          inflation_level_per_source,
                                                          expectation_values,
                                                          True)

    ## Define moment matrix columns
    ordered_cols_num = build_columns(col_structure, measurements, names)

    ## Get moment matrix
    filename_momentmatrix = 'momentmatrix+' + filename_label + '.dat-s'
    filename_monomials = 'monomials' + filename_label + '.txt'
    ordered_cols = from_coord_to_sym(ordered_cols_num, names, n_sources, measurements)
    get_relaxation_wrap(measurements, substitutions, ordered_cols, filename_momentmatrix, filename_monomials,
                        verbosity=verbose)
    problem_arr, monomials_list = read_problem_from_file(filename_momentmatrix, filename_monomials)

    ## Get the symmetry group
    inflation_symmetries = calculate_inflation_symmetries_coord(ordered_cols_num, inflation_level_per_source, n_sources)

    ## Symmetrize the moment matrix
    symmetric_arr, orbits, remaining_variables, remaining_monomials \
        = symmetrize_momentmatrix(problem_arr, monomials_list, inflation_symmetries)  ##

    ## Factorize the symmetrized monomials
    monomials_as_numbers = monomials_str_to_numbers(remaining_monomials, names)
    monomials_factors = factorize_monomials(monomials_as_numbers)
    monomials_factors_names = monomials_factors_from_num_to_name(monomials_factors, names)

    ## Label the different factors as known, semiknown and unknown and reorder
    monomials_factors_knowable = label_knowable_and_unknowable(monomials_factors, hypergraph)
    monomials_factors_names = reorder_according_to_known_semiknown_unknown(monomials_factors_names,
                                                                           monomials_factors_knowable)

    if verbose >= 1:
        print("Number of known, semi-known and unknown variables =",
              sum(monomials_factors_knowable[:, 1] == 'Yes'),
              sum(monomials_factors_knowable[:, 1] == 'Semi'),
              sum(monomials_factors_knowable[:, 1] == 'No'))

    n_known = sum(monomials_factors_knowable[:, 1] == 'Yes')
    n_something_known = sum(monomials_factors_knowable[:, 1] != 'No')
    # For only constraints on known
    # stop_counting = n_known
    # For also factorization constraints
    stop_counting = n_something_known

    ## !! Be careful when doing scalar extension. Here we join together all products of unknown variables
    ## back into its original variable.
    monomials_factors_names = combine_products_of_unknowns(measurements, monomials_factors_names,
                                                           monomials_factors_knowable, names, hypergraph,
                                                           monomials_list)  ##

    ## Go from string or numeric tuple representation of the monomials to a single integer representation
    # ! warning, in the following "monomials_factors_reps" only covers monomials with known parts or semiknown parts
    monomials_factors_vars = monomial_to_var_repr(monomials_factors_names, monomials_factors_knowable, monomials_list,
                                                  flag_use_semiknowns=calculate_semiknowns)  ##
    monomials_factors_reps = change_to_representative_variables(monomials_factors_vars, orbits)

    ####### TODO Make this intermediate save point nicer with simpler integer enumeration
    ## Save the current matrix and variables
    ## TODO SAVE this so that we don't need to recompute it for different probability distributions
    ### TODO find a nice way to save avoiding pickle, for compatibility with other programs

    mdict = {'symmetric_arr': symmetric_arr,
             'monomials_factors_names': monomials_factors_names,
             'monomials_factors_reps': monomials_factors_reps,
             'remaining_monomials': remaining_monomials,
             'monomials_list': monomials_list,
             'n_known': n_known,
             'n_something_known': n_something_known,
             'stop_counting': stop_counting,
             'measurements': measurements,
             'substitutions': substitutions,
             'names': names,
             'ordered_cols_num': ordered_cols_num,
             'ordered_cols': ordered_cols,
             'inflation_symmetries': inflation_symmetries}
    '''
    savemat('scenario_relaxation' + filename_label + '.mat', mdict)
    '''

    obj_to_save = [symmetric_arr, monomials_factors_names, monomials_factors_reps, remaining_monomials, monomials_list,
                   n_known, n_something_known, stop_counting]
    with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(obj_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    return mdict


def helper_extract_constraints(settings_per_party: List[int], outcomes_per_party: List[int], hypergraph: List[List],
                               inflation_level_per_source: List[int],
                               probability_function: Callable, prob_param, filename_label: str = ''):
    """
    Intermediate helper function that simply loads the SDP saved in get_sdp_relaxation in 'objs.pkl' and does some
    processing to get known and seminknown moments.
    Used for debugging and seeing what values the variables have.
    TODO: Remove this at some point.

    :param settings_per_party: Number of inputs for each party.
    :param outcomes_per_party: Number of outputs for each party.
    :param hypergraph: Causal hypothesis.
    :param inflation_level_per_source: Number of copies of each source.
    :param probability_function: A callable function which takes a tuple of outcomes (a1,a2,...) and of settings (x1,x2,...) and outputs the probability p(a1,a2,...|x1,x2,...).
    :param prob_param: Optional parameters passable to probability_function.
    :param filename_label: Label for filenames.
    :return: Tuple of variables of interest
    """
    '''
    # Load the sdp problem
    relaxation_data = loadmat('scenario_relaxation' + filename_label + '.mat')
    symmetric_arr = relaxation_data['symmetric_arr']
    monomials_factors_names = relaxation_data['monomials_factors_names']
    monomials_factors_reps = relaxation_data['monomials_factors_reps']
    remaining_monomials = relaxation_data['remaining_monomials']
    monomials_list = relaxation_data['monomials_list']
    n_known = relaxation_data['n_known']
    n_something_known = relaxation_data['n_something_known']
    stop_counting = relaxation_data['stop_counting']
    '''
    with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        symmetric_arr, monomials_factors_names, monomials_factors_reps, remaining_monomials, monomials_list, \
        n_known, n_something_known, stop_counting = pickle.load(f)

    '''
    # clean format from import
    for i in range(len(monomials_factors_names)):
        monomials_factors_names[i][0] = monomials_factors_names[i][0][0][0]
        monomials_factors_names[i][1] = list(monomials_factors_names[i][1])
    '''

    ## Extract variables whose value we can get from the probability distribution in a symbolic form
    variables_to_be_given = get_variables_the_user_can_specify(monomials_factors_reps, monomials_list)
    symbolic_variables_to_be_given = transform_vars_to_symb(copy.deepcopy(variables_to_be_given),
                                                            max_nr_of_parties=len(settings_per_party))
    print("symbolic_variables_to_be_given =", symbolic_variables_to_be_given)

    ## Substitute the list of known variables with symbolic values with numerical values
    variables_values = substitute_sym_with_numbers(copy.deepcopy(symbolic_variables_to_be_given), settings_per_party,
                                                   outcomes_per_party, probability_function, prob_param)

    assert (np.array(variables_values)[:, 0].astype(int).tolist()
            == np.array(variables_to_be_given)[:, 0].astype(int).tolist())
    print("variables_values =", variables_values)
    final_monomials_list = substitute_variable_values_in_monlist(variables_values, monomials_factors_reps,
                                                                 monomials_factors_names, stop_counting)

    final_positions_matrix, known_moments, semiknown_moments, variable_dict = export_to_MATLAB(final_monomials_list,
                                                                                               symmetric_arr,
                                                                                               'inflationMATLAB_' + filename_label + '.mat',
                                                                                               n_known,
                                                                                               n_something_known)
    return final_positions_matrix, known_moments, semiknown_moments, symbolic_variables_to_be_given, variable_dict


def export_to_MATLAB(final_monomials_list: np.ndarray,
                     symmetric_arr: np.ndarray,
                     matlab_filename: str,
                     n_known: int,
                     n_something_known: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Writes in a .mat file various variables of interest passed as input.

    Returns the same variables plus one more for debugging.
    TODO: make this not return anything as it doesn't make sense. Change whatever depends on the output from this function.
    """
    # Variables 1 and 2 are reserved for the numerical values 0 and 1, respectively
    variable_dict = {**{1: 1, 2: 2},
                     **{init: fin for init, fin in zip(final_monomials_list[:, 0] + 2,
                                                       range(3, len(final_monomials_list) + 3))}}
    for idx in range(len(final_monomials_list)):
        final_monomials_list[idx][0] = variable_dict[final_monomials_list[idx][0] + 2]
        if isinstance(final_monomials_list[idx][1][-1], int):  ## !! Careful
            final_monomials_list[idx][1][-1] = variable_dict[final_monomials_list[idx][1][-1] + 2]

    values_matrix = symmetric_arr[:, :, 1]
    positions_matrix = symmetric_arr[:, :, 0].astype(int) + 1
    # Use variable 0 for the numerical value 0
    positions_matrix[values_matrix == 0] = 0
    # MATLAB does not like 0 indices
    positions_matrix += 1

    final_positions_matrix = positions_matrix.copy()
    for i, row in enumerate(final_positions_matrix):
        for j, col in enumerate(row):
            final_positions_matrix[i, j] = variable_dict[col]

    known_moments = [0, 1] + [mul(factors) for _, factors in final_monomials_list[:n_known]]
    semiknown_moments = [[var, mul(val[:-1]), val[-1]]
                         for var, val in final_monomials_list[n_known:n_something_known]]

    savemat(matlab_filename,
            mdict={'G': final_positions_matrix,
                   'known_moments': known_moments,
                   'propto': semiknown_moments
                   }
            )
    return final_positions_matrix, known_moments, semiknown_moments, variable_dict


def substitute_variable_values_in_monlist(variables_values: np.ndarray,
                                          monomials_factors_reps: np.ndarray,
                                          monomials_factors_names: np.ndarray,
                                          stop_counting: int):
    """


    :param variables_values:
    :param monomials_factors_reps:
    :param monomials_factors_names:
    :param stop_counting:
    :return:
    """
    vars_numeric_dict = {var: val for var, val in variables_values}
    monomials_factors_numeric = monomials_factors_reps.copy()
    for idx, [_, monomial_factors] in enumerate(monomials_factors_numeric):
        factors_nums_list = []
        for factor in monomial_factors:
            try:
                factors_nums_list.append(vars_numeric_dict[factor])
            except KeyError:
                factors_nums_list.append(factor)
        monomials_factors_numeric[idx][1] = sorted(factors_nums_list)

    final_monomials_list = np.concatenate([monomials_factors_numeric,
                                           monomials_factors_names[stop_counting:]])
    return final_monomials_list


def generate_commuting_measurements(party, label):
    """Generates the list of symbolic variables representing the measurements
    for a given party. The variables are treated as commuting.

    :param party: configuration indicating the configuration of number m
                  of measurements and outcomes d for each measurement. It is a
                  list with m integers, each of them representing the number of
                  outcomes of the corresponding  measurement.
    :type party: list of int
    :param label: label to represent the given party
    :type label: str

    :returns: list of sympy.core.symbol.Symbol
    """
    measurements = []
    for i, p in enumerate(party):
        measurements.append(generate_variables(label + '_%s_' % i, p - 1,
                                               hermitian=True))
    return measurements


def generate_noncommuting_measurements(party, label):
    """Generates the list of symbolic variables representing the measurements
    for a given party. The variables are treated as noncommuting.

    :param party: configuration indicating the configuration of number m
                  of measurements and outcomes d for each measurement. It is a
                  list with m integers, each of them representing the number of
                  outcomes of the corresponding  measurement.
    :type party: list of int
    :param label: label to represent the given party
    :type label: str

    :returns: list of sympy.core.symbol.Symbol
    """
    measurements = []
    for i, p in enumerate(party):
        measurements.append(generate_operators(label + '_%s_' % i, p - 1,
                                               hermitian=True))
    return measurements


def generate_parties(hypergraph, settings, outcomes, inflation_level_per_source,
                     expectation_values=False, return_names=False, noncommuting=False):
    '''Generates all the party operators and substitution rules in an quantum
       inflation setup on a network given by a hypergraph.

       Args:
       :param hypergraph: Matrix encoding the network structure. Each row
                          represents a shared state, and each column contains a
                          0 if the corresponding party is not fed by the state
                          and 1 otherwise
       :type hypergraph: np.ndarray
       :param settings: Number of measurement settings available to each party
       :type settings: list of int
       :param outcomes: Number of possible outcomes per setting for each party
       :type outcomes: list of int > 1
       :param inflation_level_per_source: Number of copies of each state, can be different
       :type inflation_level_per_source: list of int
       :param expectation_values: Whether the binary-outcome operators are
                                  written in projector or Collins-Gisin form
       :type expectation_values: bool
       :param return_names: Whether the names assigned to parties are returned
       :type return_names: bool

       Returns:
       :measurements: array of measurement operators. measurements[p][c][i][o]
                      is the operator for party p, copies c, input i, output o
       :substitutions: dict containing commutation, orthogonality and square
                       constraints
       :parties: list of names assigned to parties
    '''
    assert len(settings) == len(outcomes), 'There\'s a different number of settings and outcomes'
    assert len(settings) == hypergraph.shape[1], 'The hypergraph does not have as many columns as parties'
    substitutions = {}
    measurements = []
    parties = [chr(i) for i in range(ord('A'), ord('A') + hypergraph.shape[1])]
    n_states = hypergraph.shape[0]
    for pos, [party, ins, outs] in enumerate(zip(parties, settings, outcomes)):
        party_meas = []
        party_states = sum(hypergraph[:, pos])
        # Generate all possible copy indices for a party
        # for roll back:all_inflation_indices = product(range(inflation_level), repeat=party_states)
        all_inflation_indices = product(
            *[list(range(inflation_level_per_source[p_idx])) for p_idx in np.nonzero(hypergraph[:, pos])[0]])
        # Include zeros in the positions corresponding to states not feeding the party
        all_indices = []
        for inflation_indices in all_inflation_indices:
            indices = []
            i = 0
            for idx in range(n_states):
                if hypergraph[idx, pos] == 0:
                    indices.append('0')
                elif hypergraph[idx, pos] == 1:
                    indices.append(str(inflation_indices[i] + 1))  # The +1 is just to begin at 1
                    i += 1
                else:
                    raise Error('You don\'t have a proper hypergraph')
            all_indices.append(indices)

        # Generate measurements for every combination of indices
        for indices in all_indices:
            if outs == 2:
                if noncommuting:
                    meas = generate_noncommuting_measurements([outs + 1 for _ in range(ins)],
                                                              party + '_' + '_'.join(indices))
                else:
                    meas = generate_commuting_measurements([outs + 1 for _ in range(ins)],
                                                           party + '_' + '_'.join(indices))
                for i in range(ins):
                    meas[i].pop(-1)
            else:
                if noncommuting:
                    meas = generate_noncommuting_measurements([outs for _ in range(ins)],
                                                              party + '_' + '_'.join(indices))
                else:
                    meas = generate_commuting_measurements([outs for _ in range(ins)],
                                                           party + '_' + '_'.join(indices))

            subs = projective_measurement_constraints(meas)
            substitutions = {**substitutions, **subs}
            party_meas.append(meas)
        measurements.append(party_meas)

    if expectation_values:
        for party, outs in enumerate(outcomes):
            if outs == 2:
                for operator in flatten(measurements[party]):
                    substitutions[operator ** 2] = S.One

    if return_names:
        return measurements, substitutions, parties
    else:
        return measurements, substitutions


def build_columns(party_structure: List[List],
                  measurements: List[List[List[sympy.core.symbol.Symbol]]],
                  names: List[str]) -> List[List[List[int]]]:

    """
    Helper function which builds a the list of operators for building the SDP. The operators are encoded numerically
    labeled by the party, inflation copies on which it acts, and inputs and outputs.py

    For the output, suppose we want to output {Id, A_ij_io, B_kl_i'o' * C_mn_i''o''}. This is enconded as [[], [[1,0,i,j,i,o]], [[2,k,l,0,i',o'],[3,m,0,n,i'',o'']]].
    The first element of the list of numbers denotes the parties starting from 1, then 2, etc.
    The last two numbers denotes the input and output for the measurement.
    The remaining numbers in between are equal to the number of sources.
    If there is a 0, then this operator doesn't act on that source (the first source is the number right after the party number).
    If there is any other number, this denotes on which copy of the source this operators acts.
    A list of lists of such numbers represents a product of operators.
    There is another helper function which converts this list of lists of numbers to products of symbolic variables.

    :param party_structure: Specified what parties and products of parties to include in the moments for the SDP.
    :param measurements: Sympy symbols denoting the measurement operators of each party. Called as measurements[party][input][output]
    :param names: List of party labels.
    :return: A list of monomials encoded numerically.
    """
    # party_structure is something like:
    # only the first element is special, and it determines whether we include 1 or not
    res = []
    for block in party_structure:
        if block == []:
            res.append([0])
        else:
            meas_ops = []
            for party in block:
                meas_ops.append(flatten(measurements[party]))
            for slicee in product(*meas_ops):
                coords = []
                for factor in slicee:
                    coords.append(*to_numbers(factor, names))
                res.append(coords)
    return res



def from_coord_to_sym(ordered_cols_coord: List[List[List[int]]],
                      names: str,
                      n_sources: int,
                      measurements: List[List[List[sympy.core.symbol.Symbol]]]) -> List[sympy.core.symbol.Symbol]:
    """
    Go from the output of build_columns to a list of symbolic operators
    """
    flatmeas = np.array(flatten(measurements))
    measnames = np.array([str(meas) for meas in flatmeas])

    res = []
    for elements in ordered_cols_coord:
        if elements == [0]:
            res.append(S.One)
        else:
            producto = S.One
            for element in elements:
                party = element[0]
                name = names[party - 1] + '_'
                for s in element[1:1 + n_sources]:
                    name += str(s) + '_'
                name += str(element[-2]) + '_' + str(element[-1])
                term = flatmeas[measnames == name][0]
                producto = producto * term
            res.append(producto)
    return res


def find_permutation(list1: List, list2: List) -> List:
    '''Returns the permutation that transforms list2 in list1, by saying
       for each element in list2, the element that is in list1 in
       that position
    '''
    if (len(list1) != len(list2)) or (set(list1) != set(list2)):
        raise Exception('The two lists are not permutations of one another')
    else:
        original_dict = {element: num for element, num in zip(list1, range(len(list1)))}
        return [original_dict[element] for element in list2]


def mul(lst: List):
    result = 1
    for element in lst:
        result *= element
    return result


'''
def apply_source_permutation(columns, source, permutation, measurements):
    flatmeas = np.array(flatten(measurements))
    measnames = np.array([str(meas) for meas in flatmeas])
    permuted_op_list = []
    for monomial in columns:
        if monomial == S.One:
            permuted_op_list.append(monomial)
        else:
            factors = monomial.as_ordered_factors()
            relevant_info = []
            for i, factor in enumerate(factors):
                factor_split = factor.name.split('_')
                inflation_indices = [int(idx) for idx in factor_split[1:-1]]
                if inflation_indices[source] > 0:
                    inflation_indices[source] = ((inflation_indices[source]-1)^permutation)+1 # Python starts counting at 0
                    new_factor = flatmeas[measnames == '_'.join(factor_split[:1]
                                                               + [str(idx) for idx in inflation_indices]
                                                               + factor_split[-1:])][0]
                else:
                    continue
                factors[i] = new_factor
            permuted_op_list.append(mul(factors))
    return permuted_op_list
'''


def calculate_inflation_symmetries_coord(ordered_cols_coord: List[List[List[int]]],
                                         inflation_level_per_party: List[int],
                                         n_sources: int) -> List[List]:
    """
    Calculates all the symmetries and applies them to the set of operators used to define the moment matrix. The new
    set of operators is a permutation of the old. The function outputs a list of all permutations.
    """
    inflation_symmetries = [list(range(len(ordered_cols_coord)))]  # Start with the identity permutation
    for source, permutation in tqdm(sorted(
            [(source, permutation) for source in list(range(n_sources)) for permutation in
             permutations(range(inflation_level_per_party[source]))])):
        # if list(permutation) == list(range(inflation_level_per_party[source])):
        #    # Skip the identity permutation of a source which is the identity perm of ordered_cols_coord,
        #    # the first element of inflation_symmetries
        #    continue
        # else:
        permuted_cols_ind = apply_source_permutation_coord_input(ordered_cols_coord, source, permutation)
        list_original = from_numbers_to_flat_tuples(ordered_cols_coord)
        list_permuted = from_numbers_to_flat_tuples(permuted_cols_ind)
        total_perm = find_permutation(list_permuted, list_original)
        inflation_symmetries.append(total_perm)

    return inflation_symmetries


def apply_source_permutation_coord_input(columns, source, permutation):
    """
    Applies a specific source permutation to the list of operators used to define the moment matrix. Outputs
    the permuted list of operators. The operators are enconded as lists of numbers denoting [party,source_1_copy,source_2_copy,...,input,output]
    """
    permuted_op_list = []
    for monomial in columns:
        if monomial == [0]:
            permuted_op_list.append(monomial)
        else:
            new_factors = copy.deepcopy(monomial)
            # print("old",new_factors)
            for i in range(len(monomial)):
                # print("source", source, "permutation", list(permutation), "i",i, "old_factors", new_factors[i])
                if new_factors[i][1 + source] > 0:
                    new_factors[i][1 + source] = permutation[
                                                     new_factors[i][1 + source] - 1] + 1  # Python starts counting at 0
                else:
                    continue
                # print("!ource", source, "permutation", list(permutation), "i",i,"new_factors", new_factors[i])
            permuted_op_list.append(new_factors)
            # print("new",new_factors)
            # print("\n")
    # print(list(zip(columns, permuted_op_list)))
    return permuted_op_list


### Todo MOVE TO SDP utils
def read_from_sdpa(filename):
    with open(filename, 'r') as file:
        problem = file.read()
    _, nvars, nblocs, blocstructure, obj = problem.split('\n')[:5]
    mat = [list(map(float, row.split('\t'))) for row in tqdm(problem.split('\n')[5:-1],
                                                             desc='Reading problem')]
    return mat, obj


def create_array_from_sdpa(sdpa_mat):
    sdpa_mat = np.array(sdpa_mat)
    size = int(max(sdpa_mat[:, 3]))
    mat = np.zeros((size, size, 2))
    for var, _, i, j, val in tqdm(sdpa_mat):
        mat[int(i - 1), int(j - 1)] = np.array([var, val])
        mat[int(j - 1), int(i - 1)] = np.array([var, val])
    return mat


def as_ordered_factors_for_powers(monomial):
    # this is for treating cases like A**2, where we want factors = [A, A] and this behaviour
    # doesn't work with .as_ordered_factors()
    factors = monomial.as_ordered_factors()
    factors_expanded = []
    for f_temp in factors:
        base, exp = f_temp.as_base_exp()
        if exp == 1:
            factors_expanded.append(base)
        elif exp > 1:
            for i_idx in range(exp):
                factors_expanded.append(base)
    factors = factors_expanded
    return factors


def to_numbers(monomial, parties_names):
    '''monomial can be given as a string or as a product of symbols'''
    parties_names_dict = {name: i + 1 for i, name in enumerate(parties_names)}

    # monomial_parts = str(monomial).split('*')
    if isinstance(monomial, str):
        monomial_parts = monomial.split('*')
    else:
        factors = as_ordered_factors_for_powers(monomial)
        monomial_parts = [str(factor) for factor in factors]

    monomial_parts_indices = []
    for part in monomial_parts:
        atoms = part.split('_')
        indices = ([parties_names_dict[atoms[0]]]
                   + [int(j) for j in atoms[1:-2]]
                   + [int(atoms[-2]), int(atoms[-1])])
        monomial_parts_indices.append(indices)
    return monomial_parts_indices


def to_name(monomial_numbers, parties_names):
    components = []
    for monomial in monomial_numbers:
        components.append('_'.join([parties_names[monomial[0] - 1]]  # party
                                   + [str(i) for i in monomial[1:]]))  # input output
    return '*'.join(components)


def from_numbers_to_flat_tuples(list1):
    tuples = []
    for element in list1:
        if element == [0]:
            tuples.append(tuple([0]))
        else:
            tuples.append(tuple(flatten(element)))
    return tuples


def is_knowable(factors_numbers, hypergraph_scenario):
    # After inflation and factorization, a monomial is known if it just contains
    # at most one operator per party, and in the case of having one operator per
    # node in the network, if the corresponding graph is the same as the scenario
    # hypergraph
    n_parties = hypergraph_scenario.shape[1]
    factors_states = np.array(factors_numbers)[:, 0]
    if len(factors_numbers) < n_parties:
        return len(set(factors_states)) == len(factors_states)
    else:
        if len(set(factors_states)) != len(factors_states):
            return False
        else:
            '''
            print("Case 3")
            G = nx.Graph()
            G.add_nodes_from(range(len(factors_numbers)))
            inflation_indices = np.array(factors_numbers)[:,1:-2]
            for i, component in enumerate(inflation_indices):
                for j, index in enumerate(component):
                    if index > 0:
                        connected_monomials = np.where(inflation_indices[:, j]==index)[0]
                        for k in connected_monomials:
                            if k != i:
                                G.add_edge(i, k)
            return nx.is_isomorphic(G, nx.complete_graph(n_parties))
            '''
            return hypergraphs_are_equal(hypergraph_of_a_factor(factors_numbers), hypergraph_scenario)


def transform_vars_to_symb(variables_to_be_given, max_nr_of_parties=2):
    sym_variables_to_be_given = copy.deepcopy(variables_to_be_given)
    for idx, [var, term] in enumerate(variables_to_be_given):
        factors = term.split('*')
        nr_terms = len(factors)
        factors = [list(factor.split('_')) for factor in factors]
        '''
        for i in range(nr_terms):
            # Split ['A',...,'3','io'] into ['A',...,'3','i', 'o']
            setting =  factors[i][-2]
            output = factors[i][-1]
            factors[i].pop()
            factors[i].append(setting)
            factors[i].append(output)
        '''
        factors = np.array(factors)
        parties = factors[:, 0]
        inputs = factors[:, -2]
        outputs = factors[:, -1]
        name = 'p'
        # add parties if we are marginalizing over a distribution
        if len(parties) < max_nr_of_parties:
            for p in parties:
                name += p
        name += '('
        for o in outputs:
            name += o
        name += '|'
        for i in inputs:
            name += i
        name += ')'
        sym_variables_to_be_given[idx][1] = sp.symbols(name)

    return sym_variables_to_be_given


def substitute_sym_with_value(syminput, settings_per_party, outcomes_per_party, prob_function, *argv, **kwargs):
    '''
    prob_function is a function which works as follows
    prob_function(a1,a2,...,an,x1,x2,...,xn)
    in "settings_per_party" we have a list [max(x1) max(x2) ... max(xn)]
    and in "outcomes_per_party" another one [max(a1) max(a2) ... max(an)]
    outlining the upper bounds of the inputs and outputs
    *argv **kwargs containts possible argumetns that might be passed to prob_function
    '''
    # extract the parties 
    nrparties = len(settings_per_party)
    name = syminput.name
    charelement = name[0]  # should be 'p'
    assert charelement == 'p', "The names of the symbolic variables are not correct."
    parties = []  # Parties over which to NOT marginalize.
    idx = 1
    if name[1] == '(':
        parties = [chr(ord('A') + i) for i in range(nrparties)]
    else:
        while name[idx] != '(':
            parties.append(name[idx])
            idx += 1
    assert parties == sorted(parties), "The symbolic variables should have the parties in the correct order."
    idx += 1
    outcomes = []
    while name[idx] != '|':
        outcomes.append(int(name[idx]))
        idx += 1
    idx += 1
    inputs = []
    while name[idx] != ')':
        inputs.append(int(name[idx]))
        idx += 1

    # assume parties are in order 'A'->0, 'B'->1, 'C'->2, etc.
    parties_idx = [ord(p) - ord('A') for p in parties]
    if parties_idx:  # if not empty
        aux = list(range(nrparties))
        for p in parties_idx:
            aux.remove(p)
        over_which_to_marginalize = aux
    else:
        over_which_to_marginalize = []

    # here we want to fix the settings we know from 'inputs' and choose a fixed one for the 
    # settings we marginalize over BECAUSE WE ASSUME NO SIGNALING, important assumption to 
    # bear in mind
    # by default if we sum over a party, we use input==0
    # if we dont assume this we need to specify all the inputs in the symbolic 
    # probabilities, eg pA(a|x)->pA(a|xyz)
    settings_aux = [[0] for x in range(nrparties)]
    i_idx = 0
    for p in parties_idx:
        settings_aux[p] = [inputs[i_idx]]
        i_idx += 1
    # for the outcomes we will define a list of lists where where the outcomes that are 
    # not marginalized over we give a fixed value and for the others we give all possible values
    # because I want to to itertools.product
    # example: pAC(0,1|1,2) --> [[0],[0,1,2],[1]]
    # we have [0,1,2] because Bob is being marginalized over so we put all outcome values,
    # but we only put 0 and 1 for Alice and Charlie.
    outcomes_aux = []
    for p in range(nrparties):
        if p in parties_idx:
            outcomes_aux.append([outcomes[parties_idx.index(p)]])  # i use .index in case the parties are disordered
        else:
            outcomes_aux.append(list(range(outcomes_per_party[p])))
    summ = 0

    settings_combination = flatten(settings_aux)
    for outcomes_combination in product(*outcomes_aux):
        summ += prob_function(*outcomes_combination, *settings_combination, *argv, **kwargs)
    return summ


def hypergraphs_are_equal(hypergraph1, hypergraph2, up_to_source_permutation=True):
    """
    I take a hypergraph to be written as a List[List], a list of hyperlinks.
    A hyperlink will be a list of 0s and 1s where there is a 1 in the i-th
    index only if the i-th party is connected by this hyperlink.
    Therefore to see if two hypergraphs are equal I will need to remove duplicate
    hyperlinks and see if two lists of hyperlinks are the same up to permutations.
    For this I will translate the hyperlinks to tuples and translate the list
    to a set. This will both remove duplicates and allow me to compare and see
    if the two sets are equal.

    !! We allow the hypergraphs to be different up to permutation of the hyperlinks,
    but within the description of a hyperlink we assume both hypergraphs use the same
    ordering for the parties.
    """
    if up_to_source_permutation:
        tuple_hypergraph1 = []
        for hyperlink in hypergraph1:
            tuple_hypergraph1.append(tuple(hyperlink))

        tuple_hypergraph2 = []
        for hyperlink in hypergraph2:
            tuple_hypergraph2.append(tuple(hyperlink))

        return set(tuple_hypergraph2) == set(tuple_hypergraph1)
    else:
        if hypergraph1.shape == hypergraph2.shape:
            return np.allclose(hypergraph1, hypergraph2)
        else:
            return False


def hypergraph_of_a_factor(factors):
    n_sources = len(factors[0][1:-2])
    party_slice = np.array(factors)[:, 0]
    n_parties = len(np.unique(party_slice))

    hypergraph = []  # The representation of the hypergraph will be a List of hyperlinks,
    # Hyperlinks are lists of 0s and 1s. If element i of the list if the value 1,
    # then this means party i is connected by this hyperlink.
    for source in range(n_sources):
        inf_indices = np.array(factors)[:, source + 1]
        for inflation_index in set(inf_indices):
            parties_that_are_connected = party_slice[np.where(inf_indices == inflation_index)]
            if len(parties_that_are_connected) > 1:  # we only have a hyperlink if there is more than one party affected
                hyperlink = np.zeros(n_parties).astype(int)
                hyperlink[parties_that_are_connected - 1] = 1
                hypergraph.append(hyperlink)
    hypergraph = np.array(hypergraph)
    return hypergraph


def get_relaxation_wrap(measurements, substitutions, extramonomials, filename_momentmatrix, filename_monomials,
                        verbosity=1):
    time0 = time()
    sdp = SdpRelaxation(flatten(measurements), verbose=verbosity, parallel=True)
    sdp.get_relaxation(level=-1, extramonomials=extramonomials,
                       substitutions=substitutions)  # We don't put the moments straight away
    print("SDP relaxation was generated in " + str(time() - time0) + " seconds.\n")
    print("Saving as '" + filename_momentmatrix + "' and '" + filename_monomials + "'")
    sdp.write_to_file(filename_momentmatrix)
    sdp.save_monomial_index(filename_monomials)


def read_problem_from_file(filename_momentmatrix, filename_monomials):
    problem, _ = read_from_sdpa(filename_momentmatrix)
    problem_arr = create_array_from_sdpa(problem)
    monomials_list = np.genfromtxt(filename_monomials, dtype=str, skip_header=1).astype(object)
    print(len(monomials_list))
    return problem_arr, monomials_list


def symmetrize_momentmatrix(momentmatrix, monomials_list, inflation_symmetries):
    symmetric_arr = copy.deepcopy(momentmatrix)
    indices_to_delete = []
    orbits = {i: i for i in range(len(monomials_list))}
    for permutation in tqdm(inflation_symmetries):
        for i, ip in enumerate(permutation):
            for j, jp in enumerate(permutation):
                if symmetric_arr[i, j, 0] < symmetric_arr[ip, jp, 0]:
                    indices_to_delete.append(int(symmetric_arr[ip, jp, 0]))
                    orbits[symmetric_arr[ip, jp, 0]] = symmetric_arr[i, j, 0]
                    symmetric_arr[ip, jp, :] = symmetric_arr[i, j, :]

    # Make the orbits go until the representative
    for key, val in orbits.items():
        previous = 0
        changed = True
        while changed:
            val = orbits[val]
            if val == previous:
                changed = False
            else:
                previous = val
        orbits[key] = val

    # Remove from monomials_list all those that have disappeared
    remaining_variables = set(range(len(monomials_list))) - set(np.array(indices_to_delete) - 1)
    remaining_monomials = monomials_list[sorted(list(remaining_variables))]

    return symmetric_arr.astype(int), orbits, remaining_variables, remaining_monomials


def monomials_str_to_numbers(remaining_monomials, names):
    monomials_numbers = remaining_monomials.copy()
    for i, line in enumerate(tqdm(remaining_monomials)):
        monomials_numbers[i][1] = to_numbers(line[1], names)
    monomials_numbers[:, 0] = monomials_numbers[:, 0].astype(int)
    return monomials_numbers


def factorize_monomials(monomials_as_numbers):
    monomials_factors = monomials_as_numbers.copy()
    for idx, [test, monomial] in enumerate(monomials_factors):
        inflation_indices = np.array(monomial)[:, 1:-2]
        G = nx.Graph()
        G.add_nodes_from(range(len(monomial)))
        # For each factor, and each state that reaches it, connect its vertex with all the vertices which have the same state
        for i, component in enumerate(inflation_indices):
            for j, index in enumerate(component):
                if index > 0:
                    connected_monomials = np.where(inflation_indices[:, j] == index)[0]
                    for k in connected_monomials:
                        if k != i:
                            G.add_edge(i, k)
        monomials_factors[idx][1] = [np.array(monomials_factors[idx][1])[list(component)]
                                     for component in list(nx.connected_components(G))]
    return monomials_factors


def monomials_factors_from_num_to_name(monomials_factors, names):
    monomials_factors_names = monomials_factors.copy()
    for idx, [_, monomial_factors] in enumerate(monomials_factors_names):
        factors_names_list = [to_name(factors, names) for factors in monomial_factors]
        monomials_factors_names[idx][1] = factors_names_list
    return monomials_factors_names


def label_knowable_and_unknowable(monomials_factors, hypergraph):
    monomials_factors_knowable = monomials_factors.copy()
    for idx, [test, monomial_factors] in enumerate(monomials_factors_knowable):
        factors_known_list = [is_knowable(factors, hypergraph) for factors in monomial_factors]
        if all(factors_known_list):
            knowable = 'Yes'
        elif any(factors_known_list):
            knowable = 'Semi'
        else:
            knowable = 'No'
        monomials_factors_knowable[idx][1] = knowable
    return monomials_factors_knowable


def reorder_according_to_known_semiknown_unknown(input_list, tags):
    reordered_list = np.concatenate(
        [input_list[tags[:, 1] == 'Yes'],
         input_list[tags[:, 1] == 'Semi'],
         input_list[tags[:, 1] == 'No']])
    return reordered_list


def combine_products_of_unknowns(measurements, monomials_factors_names_input, monomials_factors_knowable, names,
                                 hypergraph, monomials_list):
    # If there is a semiknown factorization with >1 unknowns, we must use the
    # unfactorized variable. This part must be commented out when combining
    # with scalar extension, and be used with a lot of care when using
    # non-commuting variables (because now we use the fact that all operators commute
    # to put the expressions in "canonical form")
    n_known = sum(monomials_factors_knowable[:, 1] == 'Yes')
    n_something_known = sum(monomials_factors_knowable[:, 1] != 'No')

    flatmeas = np.array(flatten(measurements))
    measnames = np.array([str(meas) for meas in flatmeas])

    monomials_factors_names = monomials_factors_names_input.copy()
    for var, monomial_factors in monomials_factors_names_input[n_known:n_something_known, :]:
        factors_unknown_list = [int(not is_knowable(to_numbers(factors, names), hypergraph))
                                for factors in monomial_factors]
        # Emi: note to self from Alex's code.
        # the condition len(factors_unknown_list) > 2 is because if there is no
        # known factor, Fo eg v2*v3=v1, we already know v1. we can just use the variable v1
        # from before applying the method "factorize_monomials" to v1, which gives us the v2*v3
        if (len(factors_unknown_list) > 2) and (sum(factors_unknown_list) > 1):
            knowns = np.where(np.array(factors_unknown_list) == 0)
            unknowns = np.where(np.array(factors_unknown_list) == 1)
            # Emi: note to self from Alex's code.
            # go from ['ABC','DG'] to ['A','B','C','D','G'] and then
            # multiply symbolically ['A','B','C','D','G'] applying the
            # corresponding commutation relations
            unknown_components = flatten([part.split('*')
                                          for part in np.array(monomial_factors)[unknowns]])
            unknown_operator = mul([flatmeas[measnames == op] for op in unknown_components])[0]
            unknown_var = int(monomials_list[monomials_list[:, 1] == str(unknown_operator)][0][0])
            new_line = [var, np.array(monomial_factors)[knowns].tolist() + [str(unknown_operator)]]
            monomials_factors_names[monomials_factors_names_input[:, 0] == var] = new_line

    return monomials_factors_names


def monomial_to_var_repr(monomials_factors_names, monomials_factors_knowable, monomials_list, flag_use_semiknowns=True):
    n_known = sum(monomials_factors_knowable[:, 1] == 'Yes')
    n_something_known = sum(monomials_factors_knowable[:, 1] != 'No')
    if not flag_use_semiknowns:
        # For only constraints on known
        stop_counting = n_known
    else:
        # For also factorization constraints
        stop_counting = n_something_known

    monomials_factors_vars = monomials_factors_names[:stop_counting, :].copy()
    for idx, [_, monomial_factors] in enumerate(monomials_factors_vars):
        factors_vars_list = [int(monomials_list[monomials_list[:, 1] == factor][0][0])
                             for factor in monomial_factors]
        monomials_factors_vars[idx][1] = factors_vars_list

    return monomials_factors_vars


def change_to_representative_variables(monomials_factors_vars, orbits):
    monomials_factors_reps = monomials_factors_vars.copy()
    for idx, [_, monomial_factors] in enumerate(monomials_factors_reps):
        if len(monomial_factors) > 1:
            factors_reps_list = [int(orbits[factor]) for factor in monomial_factors]
            monomials_factors_reps[idx][1] = factors_reps_list
        else:
            # Sanity check
            original = monomials_factors_vars[idx][0]
            after_factor = monomials_factors_vars[idx][1][0]
            assert original == after_factor, ('The variable {} is not assigned to'
                                              + 'itself but to {}'.format(original,
                                                                          after_factor))
    return monomials_factors_reps


def get_variables_the_user_can_specify(monomials_factors_reps, monomials_list):
    variables_to_be_given = []
    for idx, factors in monomials_factors_reps:
        if len(factors) == 1:
            variables_to_be_given.append([idx, monomials_list[monomials_list[:, 0] == str(idx)][0][1]])
    return variables_to_be_given


def substitute_sym_with_numbers(symbolic_variables_to_be_given,
                                settings_per_party, outcomes_per_party, probability_function, probability_params={}):
    variables_values = symbolic_variables_to_be_given.copy()
    for i in range(len(variables_values)):
        variables_values[i][1] = float(substitute_sym_with_value(symbolic_variables_to_be_given[i][1],
                                                           settings_per_party,
                                                           outcomes_per_party, probability_function, probability_params))
    return variables_values
