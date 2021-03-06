from __future__ import absolute_import
import numpy as np
import itertools

from sys import hexversion

if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property

# Network is the low level original graph representation. InflationGraph will be built as a subclass of DAG, taking inflation order(s) as a further parameter.
# DAG is the user-friendly constructor. It is also a subclass of Network, as it constructs an

class Network:
    def __init__(self, hypergraph, outcome_cardinalities, setting_cardinalities):
        self.hypergraph = hypergraph
        self.outcomes_cardinalities = tuple(outcome_cardinalities)
        self.setting_cardinalities = tuple(setting_cardinalities)
        self.num_observed_vars = len(outcome_cardinalities)
        self.all_moments_shape = self.setting_cardinalities + self.outcomes_cardinalities

class DAG(Network):
    r"""Creates an instance of a cardinality-aware DAG, internally represented as a network with missing data.

    Parameters
    ----------
    hypergraph_structure : matrix (list of lists or numpy array)
        Each row is a source, each column is an observable variable.
        Zeroes indicate no connection. Ones indicate source connected to target.
    directed_structure : square matrix
        The adjacency matrix of the directed structure. A `1` indicates the row-index variable is a causal parents of
        column-index variable.
    outcome_cardinalities : list of integers
        Must have length equal to total number of observable variables.
    private_setting_cardinalities : list of integers
        Must have length equal to total number of observable variables.
        Settingless variables should be treated as having setting cardinality = 1.
    """
    def __init__(self, hypergraph_structure, directed_structure, outcome_cardinalities, private_setting_cardinalities):
        self.inverse_directed_structure = np.transpose(directed_structure)
        self.extra_setting_cardinalities = np.multiply(outcome_cardinalities, self.inverse_directed_structure)
        self.shaped_setting_cardinalities = [np.hstack((sett, cards[cards.nonzero()])) for sett, cards in
                                        zip(private_setting_cardinalities, self.extra_setting_cardinalities)]
        Network.__init__(self, hypergraph_structure, outcome_cardinalities, map(np.prod, self.shaped_setting_cardinalities))
        self.knowable_moments_shape = private_setting_cardinalities + self.outcomes_cardinalities
        self.private_setting_cardinalities = private_setting_cardinalities

    @cached_property
    def form_finder(self):
        private_setting_indices = np.arange(self.num_observed_vars)
        outcome_indices = np.arange(self.num_observed_vars)+self.num_observed_vars
        extra_setting_indices = [np.compress(parents_of, outcome_indices) for parents_of in self.inverse_directed_structure]
        effective_setting_indices = [np.hstack(([private_setting_index],extra_setting_index))
                                    for private_setting_index,extra_setting_index in zip(
                                    private_setting_indices,extra_setting_indices
                                    )]
        return np.hstack((np.hstack(effective_setting_indices),outcome_indices))

    @cached_property
    def form_finder_shape(self):
        return tuple(np.take(self.knowable_moments_shape, self.form_finder))

    # @cached_property
    # def knowable_original_probabilities2(self):
    #     return np.ravel_multi_index(
    #         np.take(list(np.ndindex(transformed_problem.knowable_moments_shape)),
    #                 transformed_problem.form_finder,
    #                 axis=1).T,
    #         transformed_problem.form_finder_shape)

    @cached_property
    def knowable_original_probabilities(self):
        mixed_radix_array = np.take(list(np.ndindex(self.knowable_moments_shape)),
                                    self.form_finder,
                                    axis=1)
        cardinality_converter = np.flip(np.multiply.accumulate(np.hstack((1, np.flip(self.form_finder_shape))))[:-1])
        return np.dot(mixed_radix_array, cardinality_converter)


    def _knowable_original_probabilities(self):
        for assignment in np.ndindex(self.all_moments_shape):
            settings_assignment = assignment[:self.num_observed_vars]
            outcomes_assigment = assignment[self.num_observed_vars:]
            knowable=True
            for setting_integer, setting_shape, parents_of in zip(settings_assignment, self.shaped_setting_cardinalities, self.inverse_directed_structure):
                settings_of_v = np.unravel_index(setting_integer, setting_shape)
                outcomes_relevant_to_v = np.compress(parents_of, outcomes_assigment)
                if not np.array_equal(settings_of_v[1:], outcomes_relevant_to_v):
                    knowable = False
                    break
            yield knowable

    @cached_property
    def knowable_original_probabilities_old(self):
        return np.flatnonzero(np.fromiter(self._knowable_original_probabilities(), np.bool))

if __name__ == '__main__':
    hypergraph_structure = [
        [1, 1, 0],
        [0, 1, 1]]
    directed_structure = [
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0]]
    outcome_cardinalities = (2, 2, 2)
    private_setting_cardinalities = (2, 1, 1)
    transformed_problem = DAG(hypergraph_structure, directed_structure, outcome_cardinalities,
                              private_setting_cardinalities)

    #print(np.maximum(transformed_problem.extra_setting_cardinalities,1))
    #print(transformed_problem.form_finder)
    #print(transformed_problem.form_finder_shape)
    print(transformed_problem.knowable_original_probabilities)
    print(transformed_problem.knowable_original_probabilities_old)





