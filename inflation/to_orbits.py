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
from internal_functions.groups import dimino_sympy, orbits_of_object_under_group_action, minimize_object_under_group_action
from internal_functions.utilities import MoveToFront, MoveToBack,SparseMatrixFromRowsPerColumn


class inflated_hypergraph:
    
    def __init__(self,hypergraph,inflation_orders):
        self.latent_count=len(hypergraph)
        self.observed_count=len(hypergraph[0])
        self.root_structure=[list(np.nonzero(hypergraph[:,observable])[0]) for observable in  range(self.observed_count)]
        self.inflations_orders = np.array(inflation_orders)
        self._latent_ancestors_of=[self.inflations_orders.take(np.nonzero(hypergraph[:,observable])[0]) for observable in range(self.observed_count)]
        self.inflation_copies = np.fromiter(map(np.prod, self._latent_ancestors_of), np.int)
        self.inflation_depths = np.fromiter(map(len, self._latent_ancestors_of), np.int)
        self.inflation_minima = np.fromiter(map(np.amin, self._latent_ancestors_of), np.int)
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
    
class packed_inflated_columns(inflated_hypergraph,DAG):
    
    def __init__(self,hypergraph,inflation_orders,directed_structure, outcome_cardinalities, private_setting_cardinalities):
        inflated_hypergraph.__init__(self,hypergraph,inflation_orders)
        DAG.__init__(self, hypergraph, directed_structure, outcome_cardinalities, private_setting_cardinalities)
        self.packed_cardinalities=[outcome_cardinalities[observable]**self.setting_cardinalities[observable] for observable in range(self.observed_count)]
        self.inflated_packed_cardinalities_array=np.repeat(self.packed_cardinalities, self.inflation_copies)
        self.inflated_packed_cardinalities_tuple=tuple(self.inflated_packed_cardinalities_array)
        self.inflated_packed_cardinalities_indecies=np.repeat(np.arange(len(outcome_cardinalities)),self.inflation_copies)
        self.column_count=self.inflated_packed_cardinalities_array.prod()
        self.shaped_packed_column_integers = np.arange(self.column_count).reshape(self.inflated_packed_cardinalities_tuple)
    
    @cached_property
    def column_orbits(self):
        AMatrix=orbits_of_object_under_group_action(self.shaped_packed_column_integers,self.inflation_group_elements).T
        AMatrix = np.compress(AMatrix[0]>=0, AMatrix, axis=1)
        return AMatrix

class expressible_sets(packed_inflated_columns):
    
    def __init__(self,hypergraph,inflation_orders,directed_structure, outcome_cardinalities, private_setting_cardinalities):
        packed_inflated_columns.__init__(self,hypergraph,inflation_orders,directed_structure, outcome_cardinalities, private_setting_cardinalities)
        self.unpacked_conf_integers=np.array([list(np.arange(self.setting_cardinalities[packed_obs])+np.array(self.setting_cardinalities)[self.inflated_packed_cardinalities_indecies[:packed_obs_index]].sum()) for packed_obs_index,packed_obs in  enumerate(self.inflated_packed_cardinalities_indecies)],dtype=object)
        #print(self.unpacked_conf_integers)
        self.conf_setting_integers=np.array([range(self.setting_cardinalities[obs]) for obs in range(self.observed_count)],dtype=object)
        #print(self.conf_setting_integers)
        self.conf_setting_indecies=self.conf_setting_integers[np.repeat(np.arange(len(self.conf_setting_integers)),self.inflation_copies)]
        self.ravelled_conf_setting_indecies=[i for j in self.conf_setting_indecies for i in j]
        #print(self.ravelled_conf_setting_indecies)
        
        self.ravelled_conf_var_indecies=np.repeat(np.arange(self.observed_count),np.array(self.setting_cardinalities)*np.array(self.inflation_copies))
        #print(self.ravelled_conf_var_indecies)
        
        self.unpacked_inflated_copies=[self.setting_cardinalities[observable]*self.inflation_copies[observable] for observable in range(self.observed_count)]
        self.inflated_unpacked_cardinalities=list(itertools.chain.from_iterable([list(np.repeat(outcome_cardinalities[observable],self.unpacked_inflated_copies[observable])) for observable in range(self.observed_count)]))
        self.inflated_unpacked_cardinalities_tuple=tuple(self.inflated_unpacked_cardinalities)
        self.shaped_unpacked_column_integers = np.arange(self.column_count).reshape(self.inflated_unpacked_cardinalities_tuple)
        
        self._canonical_pos = [
            np.outer(inflation_minimum ** np.arange(inflation_depth), np.arange(inflation_minimum)).sum(axis=0) + offset
            for inflation_minimum, inflation_depth, offset
            in zip(self.inflation_minima, self.inflation_depths, self.offsets)]
        self.packed_partitioned_eset=[np.compress(np.add(part, 1).astype(np.bool), part)
                for part in itertools.zip_longest(*self._canonical_pos, fillvalue=-1)]
        self.partitioned_unpacked_eset_candidates=[list(itertools.product(*list(self.unpacked_conf_integers[part]))) for part in self.packed_partitioned_eset]
        #print(self.partitioned_unpacked_eset_candidates)
        self.partitioned_unpacked_esets=list(itertools.product(*self.partitioned_unpacked_eset_candidates))
        #print(len(self.partitioned_unpacked_esets))
        self.flat_unpacked_esets=[[elem for part in eset for elem in part] for eset in self.partitioned_unpacked_esets]
        #print(self.flat_unpacked_esets)
        
    def _valid_outcomes(self,eset_part_candidate):
        observables=np.array(self.ravelled_conf_var_indecies)[np.array(eset_part_candidate)]
        settings_assignment=tuple(np.array(self.ravelled_conf_setting_indecies)[np.array(eset_part_candidate)])
        outcome_assignments=np.ndindex(self.outcomes_cardinalities)
        for assignment in outcome_assignments:
            validity=True
            for i in range(len(settings_assignment)):
                setting_integer=settings_assignment[i]
                setting_shape=self.shaped_setting_cardinalities[observables[i]]
                parents_of=self.inverse_directed_structure[observables[i]]
                settings_of_v = np.unravel_index(setting_integer, setting_shape)
                outcomes_relevant_to_v = np.compress(parents_of, assignment)
                validity=validity and np.array_equal(settings_of_v[1:], outcomes_relevant_to_v)
            yield validity
    def valid_outcomes(self,eset_part_candidate):
        return np.fromiter(self._valid_outcomes(eset_part_candidate), np.bool)
        
    def eset_unpacking_rows_to_keep(self,partitioned_eset):
        validoutcomes=[self.valid_outcomes(part) for part in partitioned_eset]
        
        v=validoutcomes[-1]
        for i in range(len(validoutcomes)-1):
            v=np.kron(validoutcomes[len(validoutcomes)-1-i],v)
        
        eset_kept_rows=((np.arange(len(v))+1)*v.astype(np.int))-1
        eset_kept_rows=eset_kept_rows[np.where(eset_kept_rows!=-1)[0]]
        
        return eset_kept_rows
        
        
    def eset_symmetry_group(self,partitioned_eset):
         
        when_sorted = np.argsort(np.hstack(partitioned_eset))
        it = iter(when_sorted)
        _canonical_pos2 = [list(itertools.islice(it, size)) for size in self.inflation_minima]

        unfiltered_variants = np.array([np.hstack(np.vstack(perm).T) for perm in
                                        itertools.permutations(itertools.zip_longest(*_canonical_pos2, fillvalue=-1))])
        expressible_set_variants_filter = np.add(unfiltered_variants, 1).astype(np.bool)
        unfiltered_variants = unfiltered_variants.compress(
            (expressible_set_variants_filter == np.atleast_2d(expressible_set_variants_filter[0])).all(axis=1),
            axis=0)
        filtered_variants = np.array([eset.compress(np.add(eset, 1).astype(np.bool)) for eset in unfiltered_variants])
        return np.take_along_axis(np.argsort(filtered_variants,axis=1), np.atleast_2d(when_sorted),   axis=1)
    
    def eset_symmetry_rows_to_keep(self,partitioned_eset,shape_of_eset,size_of_eset):
        which_rows_to_keep = np.arange(size_of_eset).reshape(shape_of_eset)
        minimize_object_under_group_action(
            which_rows_to_keep,
            self.eset_symmetry_group(partitioned_eset), skip=1)
        which_rows_to_keep = np.unique(which_rows_to_keep.ravel(), return_index=True)[1]
        return which_rows_to_keep
    
    
    def eset_disgarded_rows_to_trash(self,partitioned_eset,offset):
        shape_of_eset = np.take(np.array(self.inflated_unpacked_cardinalities), [elem for part in partitioned_eset for elem in part])
        size_of_eset = shape_of_eset.prod()
        which_rows_to_keep=np.intersect1d(self.eset_unpacking_rows_to_keep(partitioned_eset),self.eset_symmetry_rows_to_keep(partitioned_eset,shape_of_eset,size_of_eset))
        size_of_eset_after_symmetry_and_unpacking = len(which_rows_to_keep)
        #there_are_discarded_rows = (size_of_eset_after_symmetry_and_unpacking < size_of_eset)
        discarded_rows_to_the_back = np.full(size_of_eset, -(offset+1), dtype=np.int)
        np.put(discarded_rows_to_the_back, which_rows_to_keep, np.arange(size_of_eset_after_symmetry_and_unpacking))
        discarded_rows_to_the_back=discarded_rows_to_the_back+offset
        return discarded_rows_to_the_back
    
    
    def columns_to_unique_rows(self, shaped_column_integers,flat_eset):
        data_shape = shaped_column_integers.shape
        # Can be used for off-diagonal expressible sets with no adjustment!
        flat_eset=np.array(flat_eset)
        expr_set_size = np.take(data_shape, flat_eset).prod()
        reshaped_column_integers = shaped_column_integers.transpose(MoveToBack(len(data_shape), flat_eset)).reshape((-1, expr_set_size))            
        encoding_of_columns_to_monomials = np.empty(shaped_column_integers.size, np.int)
        encoding_of_columns_to_monomials[reshaped_column_integers] = np.arange(expr_set_size)
        return encoding_of_columns_to_monomials
    
    def AMatrix(self):
        offset=0
        amatrices=[]
        for partitioned_eset in self.partitioned_unpacked_esets:
            disgarded=self.eset_disgarded_rows_to_trash(partitioned_eset,offset)
            amatrix=disgarded.take(self.columns_to_unique_rows(self.shaped_unpacked_column_integers,[elem for part in partitioned_eset for elem in part])).take(self.column_orbits)
            offset=np.amax(disgarded)+1
            amatrices.append(amatrix)
        
        AMatrix=np.vstack(tuple(amatrices))
        trash=np.amax(AMatrix)+1
        trash_positions=np.where(AMatrix==-1)
        AMatrix[trash_positions[0],trash_positions[1]]=trash
        
        return AMatrix
    
class inflation_problem(expressible_sets):
    
    def __init__(self,hypergraph,inflation_orders,directed_structure, outcome_cardinalities, private_setting_cardinalities):
        expressible_sets.__init__(self,hypergraph,inflation_orders,directed_structure, outcome_cardinalities, private_setting_cardinalities)
        
    def inflation_matrix(self):
        InfMat=SparseMatrixFromRowsPerColumn(self.AMatrix()).asformat('csr', copy=False)
        return InfMat[:-1]
        
if __name__ == '__main__':
    
    hypergraph=np.array([[1,1,0],[0,1,1]])
    directed_structure=np.array([[0,1,1],[0,0,0],[0,0,0]])
    outcome_cardinalities = (2, 2, 2)
    private_setting_cardinalities = (2, 1, 1)
    
    inflation_orders=[2,2]
    p=((0, 4, 12), (2, 10, 14))
    e=expressible_sets(hypergraph,inflation_orders,directed_structure, outcome_cardinalities, private_setting_cardinalities)
    inf=inflation_problem(hypergraph,inflation_orders,directed_structure, outcome_cardinalities, private_setting_cardinalities)
    #print(e.AMatrix())
    print(inf.inflation_matrix().shape)
    