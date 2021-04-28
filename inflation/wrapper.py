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
    def __init__(self, hypergraph_structure, directed_structure, outcome_cardinalities, private_setting_cardinalities):
        self.inverse_directed_structure = np.transpose(directed_structure)
        self.extra_setting_cardinalities = np.multiply(outcome_cardinalities, self.inverse_directed_structure)
        self.shaped_setting_cardinalities = [np.hstack((sett, cards[cards.nonzero()])) for sett, cards in
                                        zip(private_setting_cardinalities, self.extra_setting_cardinalities)]
        Network.__init__(self, hypergraph_structure, outcome_cardinalities, map(np.prod, self.shaped_setting_cardinalities))

    #def _knowable_original_probabilities(self):
    #    for assignment in np.ndindex(self.all_moments_shape):
    #        settings_assignment = assignment[:self.num_observed_vars]
    #        outcomes_assigment = assignment[self.num_observed_vars:]
    #        for setting_integer, setting_shape, parents_of in zip(settings_assignment, self.shaped_setting_cardinalities, self.inverse_directed_structure):
    #            settings_of_v = np.unravel_index(setting_integer, setting_shape)
    #            outcomes_relevant_to_v = np.compress(parents_of, outcomes_assigment)
    #            yield np.array_equal(settings_of_v[1:], outcomes_relevant_to_v)
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
    def knowable_original_probabilities(self):
        return np.fromiter(self._knowable_original_probabilities(), np.bool).nonzero()






