#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 17:42:31 2021

@author: boraulu
"""

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
    
from wrapper import *
from internal_functions.groups import dimino_sympy, orbits_of_object_under_group_action
from internal_functions.utilities import MoveToFront, MoveToBack


class inflated_hypergraph:
    
    def __init__(self,hypergraph,inflation_orders):
        self.latent_count=len(hypergraph)
        self.observed_count=len(hypergraph[0])
        self.root_structure=[list(np.nonzero(hypergraph[:,observable])[0]) for observable in  range(self.observed_count)]
        self.inflations_orders = np.array(inflation_orders)
        self._latent_ancestors_of=[self.inflations_orders.take(np.nonzero(hypergraph[:,observable])[0]) for observable in range(self.observed_count)]
        self.inflation_copies = np.fromiter(map(np.prod, self._latent_ancestors_of), np.int)
        self.inflation_depths = np.fromiter(map(len, self._latent_ancestors_of), np.int)
        accumulated = np.add.accumulate(self.inflation_copies)
        self.inflated_observed_count = accumulated[-1]
        self.offsets = np.hstack(([0], accumulated[:-1]))
        
    @cached_property
    def inflation_group_generators(self):
        # Upgrade to mixed inflation order
        globalstrategyflat = list(
            np.add(*stuff) for stuff in zip(list(map(np.arange, self.inflation_copies.tolist())), self.offsets))
        reshapings = np.ones((self.observed_count, self.latent_count), np.uint8)
        contractings = np.zeros((self.observed_count, self.latent_count), np.object)
        for idx, latent_ancestors in enumerate(self.root_structure):
            reshapings[idx][latent_ancestors] = self.inflations_orders[latent_ancestors]
            contractings[idx][latent_ancestors] = np.s_[:]
        reshapings = map(tuple, reshapings)
        contractings = map(tuple, contractings)
        globalstrategyshaped = list(np.reshape(*stuff) for stuff in zip(globalstrategyflat, reshapings))
        gloablstrategybroadcast = np.stack(np.broadcast_arrays(*globalstrategyshaped), axis=0)
        indices_to_extract = np.hstack(tuple(shaped_elem[contraction].ravel() for shaped_elem, contraction in zip(
            np.arange(gloablstrategybroadcast.size).reshape(gloablstrategybroadcast.shape), contractings)))
        group_generators = []
        for latent_to_explore, inflation_order_for_U in enumerate(self.inflations_orders):
            generator_count_for_U = np.minimum(inflation_order_for_U, 3) - 1
            group_generators_for_U = np.empty((generator_count_for_U, self.inflated_observed_count), np.int)
            # Maybe assert that inflation order must be a strictly positive integer?
            for gen_idx in np.arange(generator_count_for_U):
                initialtranspose = MoveToFront(self.latent_count + 1, np.array([latent_to_explore + 1]))
                inversetranspose = np.argsort(initialtranspose)
                label_permutation = np.arange(inflation_order_for_U)
                if gen_idx == 0:
                    label_permutation[:2] = [1, 0]
                elif gen_idx == 1:
                    label_permutation = np.roll(label_permutation, 1)
                group_generators_for_U[gen_idx] = gloablstrategybroadcast.transpose(
                    tuple(initialtranspose))[label_permutation].transpose(
                    tuple(inversetranspose)).flat[indices_to_extract]
            group_generators.append(group_generators_for_U)
        return group_generators
    
    @cached_property
    def inflation_group_elements(self):
        return np.array(dimino_sympy([gen for gen in np.vstack(self.inflation_group_generators)]))
    
class packed_inflated_columns(inflated_hypergraph):
    
    def __init__(self,hypergraph,inflation_orders,directed_structure, outcome_cardinalities, private_setting_cardinalities):
        inflated_hypergraph.__init__(self,hypergraph,inflation_orders)
        DAG.__init__(self, hypergraph, directed_structure, outcome_cardinalities, private_setting_cardinalities)
        self.packed_cardinalities=[outcome_cardinalities[observable]**self.setting_cardinalities[observable] for observable in range(self.observed_count)]
        self.inflated_packed_cardinalities_array=np.repeat(self.packed_cardinalities, self.inflation_copies)
        self.inflated_packed_cardinalities_tuple=tuple(self.inflated_packed_cardinalities_array)
        
        self.column_count=self.inflated_packed_cardinalities_array.prod()
        self.shaped_column_integers = np.arange(self.column_count).reshape(self.inflated_packed_cardinalities_tuple)
        print(self.inflated_packed_cardinalities_tuple)
        print(self.inflated_packed_cardinalities_array.prod())
    
    @cached_property
    def column_orbits(self):
        return orbits_of_object_under_group_action(self.shaped_column_integers,self.inflation_group_elements).T
        
if __name__ == '__main__':
    
    hypergraph=np.array([[1,1,0],[0,1,1]])
    directed_structure=np.array([[0,1,1],[0,0,0],[0,0,0]])
    outcome_cardinalities = (2, 2, 2)
    private_setting_cardinalities = (2, 1, 1)
    
    inflation_orders=[2,2]
    print(packed_inflated_columns(hypergraph,inflation_orders,directed_structure, outcome_cardinalities, private_setting_cardinalities).column_orbits)
    
    