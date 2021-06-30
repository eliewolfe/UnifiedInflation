#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 17:42:31 2021

@author: boraulu
"""

from __future__ import absolute_import
import numpy as np
import itertools
import more_itertools

from sys import hexversion

if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property

from wrapper import *

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from internal_functions.groups import dimino_sympy, orbits_of_object_under_group_action, \
    minimize_object_under_group_action
from internal_functions.utilities_ import MoveToFront, MoveToBack, SparseMatrixFromRowsPerColumn
import functools
from linear_program_options.moseklp import InfeasibilityCertificate
from linear_program_options.moseklp_dual import InfeasibilityCertificateAUTO
from linear_program_options.cvxopt import InflationLP
from internal_functions.inequality_internals import *
import json
from collections import defaultdict

class inflated_hypergraph:

    def __init__(self, hypergraph, inflation_orders):
        self.latent_count = len(hypergraph)
        self.observed_count = len(hypergraph[0])
        self.root_structure = [list(np.nonzero(hypergraph[:, observable])[0]) for observable in
                               range(self.observed_count)]
        self.inflation_orders = np.asarray(inflation_orders)
        self._latent_ancestors_of = [self.inflation_orders.take(np.nonzero(hypergraph[:, observable])[0]) for
                                     observable in range(self.observed_count)]
        self.inflation_copies = np.fromiter(map(np.prod, self._latent_ancestors_of), np.int)
        self.inflation_depths = np.fromiter(map(len, self._latent_ancestors_of), np.int)
        self.inflation_minima = np.fromiter(map(np.amin, self._latent_ancestors_of), np.int)
        accumulated = np.add.accumulate(self.inflation_copies)
        self.inflated_observed_count = accumulated[-1]
        self.offsets = np.hstack(([0], accumulated[:-1]))
        # packed expressible set
        self._canonical_pos = [
            np.outer(inflation_minimum ** np.arange(inflation_depth), np.arange(inflation_minimum)).sum(axis=0) + offset
            for inflation_minimum, inflation_depth, offset
            in zip(self.inflation_minima, self.inflation_depths, self.offsets)]
        self.packed_partitioned_eset = [np.compress(np.add(part, 1).astype(np.bool), part)
                                        for part in itertools.zip_longest(*self._canonical_pos, fillvalue=-1)]
        self.packed_partitioned_eset_tuple = tuple([tuple(i) for i in self.packed_partitioned_eset])



        #print(self.packed_partitioned_eset_tuple)

    @cached_property
    def inflation_group_generators(self):
        # Upgrade to mixed inflation order
        globalstrategyflat = list(
            np.add(*stuff) for stuff in zip(list(map(np.arange, self.inflation_copies.tolist())), self.offsets))
        reshapings = np.ones((self.observed_count, self.latent_count), np.uint8)
        contractings = np.zeros((self.observed_count, self.latent_count), np.object)
        for idx, latent_ancestors in enumerate(self.root_structure):
            reshapings[idx][latent_ancestors] = self.inflation_orders[latent_ancestors]
            contractings[idx][latent_ancestors] = np.s_[:]
        reshapings = map(tuple, reshapings)
        contractings = map(tuple, contractings)
        globalstrategyshaped = list(np.reshape(*stuff) for stuff in zip(globalstrategyflat, reshapings))
        gloablstrategybroadcast = np.stack(np.broadcast_arrays(*globalstrategyshaped), axis=0)
        indices_to_extract = np.hstack(tuple(shaped_elem[contraction].ravel() for shaped_elem, contraction in zip(
            np.arange(gloablstrategybroadcast.size).reshape(gloablstrategybroadcast.shape), contractings)))
        group_generators = []
        for latent_to_explore, inflation_order_for_U in enumerate(self.inflation_orders):
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

    @cached_property
    def eset_symmetry_group(self):

        when_sorted = np.argsort(np.hstack(self.packed_partitioned_eset_tuple))
        it = iter(when_sorted)
        _canonical_pos2 = [list(itertools.islice(it, size)) for size in self.inflation_minima]

        unfiltered_variants = np.array([np.hstack(np.vstack(perm).T) for perm in
                                        itertools.permutations(itertools.zip_longest(*_canonical_pos2, fillvalue=-1))])
        expressible_set_variants_filter = np.add(unfiltered_variants, 1).astype(np.bool)
        unfiltered_variants = unfiltered_variants.compress(
            (expressible_set_variants_filter == np.atleast_2d(expressible_set_variants_filter[0])).all(axis=1),
            axis=0)
        filtered_variants = np.array([eset.compress(np.add(eset, 1).astype(np.bool)) for eset in unfiltered_variants])
        return np.take_along_axis(np.argsort(filtered_variants, axis=1), np.atleast_2d(when_sorted), axis=1)

    @cached_property
    def eset_symmetry_rows_to_keep(self):
        shape_of_eset = np.take(np.array(self.inflated_unpacked_cardinalities),
                                [elem for part in self.packed_partitioned_eset_tuple for elem in part])
        size_of_eset = shape_of_eset.prod()
        which_rows_to_keep = np.arange(size_of_eset).reshape(shape_of_eset)
        minimize_object_under_group_action(
            which_rows_to_keep,
            self.eset_symmetry_group, skip=1)
        which_rows_to_keep = np.unique(which_rows_to_keep.ravel(), return_index=True)[1]
        return which_rows_to_keep


class inflation_problem(inflated_hypergraph, DAG):
    
    @staticmethod
    def MixedCardinalityBaseConversion(cardinality, string):
        #card = np.array([cardinality[i] ** (len(cardinality) - (i + 1)) for i in range(len(cardinality))])
        card = np.flip(np.multiply.accumulate(np.hstack((1, np.flip(cardinality))))[:-1])
        if isinstance(string,str):
            str_to_array = np.array([int(i) for i in string])
        else:
            str_to_array=np.array(string,dtype=np.int64)
            #print(card,str_to_array)
        return np.dot(card, str_to_array)
    
    @staticmethod
    def ReverseMixedCardinalityBaseConversion(cardinality, num, output='array'):
        n=num
        st=[]
        for i in range(len(cardinality)):
            
            j=len(cardinality)-i-1
            #print(n,cardinality[j])
            remainder=int(np.remainder(n,cardinality[j]))
            quotient=(n-remainder)/cardinality[j]
            if output=='string':
                st.append(str(remainder))
            else:
                st.append(remainder)
            n=quotient
            if n < 1:
                break
        st.reverse()
        r=len(cardinality)-len(st)
        if r!=0:
            if output=='string':
                s1=[str(k) for k in np.zeros((1,r),np.uint)[0]]
            else:
                s1=[k for k in np.zeros((1,r),np.uint)[0]]
            s1.extend(st)
            st=s1
        if output=='string':
            return ''.join(st)
        else:
            return np.array(st)     

    def __init__(self, hypergraph, inflation_orders, directed_structure, outcome_cardinalities,
                 private_setting_cardinalities):
        
        inflated_hypergraph.__init__(self, hypergraph, inflation_orders)
        # self.directed_structure=directed_structure
        DAG.__init__(self, hypergraph, directed_structure, outcome_cardinalities, private_setting_cardinalities)
        
        self.original_conf_var_indicies = np.repeat(np.arange(self.observed_count),
                                                    np.array(self.inflation_copies))

        self.packed_partitioned_eset_deflated = [self.original_conf_var_indicies[part] for part in self.packed_partitioned_eset]
        self.knowable_margins = [self.extract_ancestral_closed_subset(part) for part in self.packed_partitioned_eset_deflated]
        ravelled_knowable_margins = list(itertools.chain.from_iterable(self.knowable_margins))

        # Elie deprecated this with the simpler code above.
        # self.knowable_margins=[part for part in self.packed_partitioned_eset_tuple if self.ancestral_closed_Q(self.original_conf_var_indicies[list(part)])]
        # problematic_partitions=list(set(self.packed_partitioned_eset_tuple)-set(self.knowable_margins))
        # if problematic_partitions:
        #     for problematic_partition in problematic_partitions:
        #         for subset_size in range(len(problematic_partition)-1,0,-1):
        #             possible_subsets=itertools.combinations(problematic_partition,subset_size)
        #             for subset in possible_subsets:
        #                 is_subset_closed=self.ancestral_closed_Q(self.original_conf_var_indicies[list(subset)])
        #                 if is_subset_closed:
        #                     break
        #             if is_subset_closed:
        #                     break
        #         if is_subset_closed:
        #             self.knowable_margins.append(subset)
        #
        # print(self.knowable_margins)
        # ravelled_knowable_margins=self.original_conf_var_indicies[[i for j in self.knowable_margins for i in j]].tolist()
        # #packed_exp_set_w_original_indices=self.original_conf_var_indicies[np.array([i for j in self.packed_partitioned_eset for i in j])]
        # #original_copy_count=np.array([packed_exp_set_w_original_indices.tolist().count(var) for var in range(self.observed_count)])
        # copy_count = np.array([ravelled_knowable_margins.count(var) for var in range(self.observed_count)])

        copy_count = np.fromiter((ravelled_knowable_margins.count(var) for var in range(self.observed_count)), int)
        new_inflation_order_candidate=np.multiply(copy_count,hypergraph).max(axis=1)
        if not np.array_equal(new_inflation_order_candidate, self.inflation_orders):
            #inflation_orders=new_inflation_order_candidate
            inflated_hypergraph.__init__(self, hypergraph, new_inflation_order_candidate)
            print('Inflation orders too large, switching to optimized inflation orders:',new_inflation_order_candidate)
        
        
        self.packed_cardinalities = [outcome_cardinalities[observable] ** self.setting_cardinalities[observable] for
                                     observable in range(self.observed_count)]
        self.inflated_packed_cardinalities_array = np.repeat(self.packed_cardinalities, self.inflation_copies)
        self.inflated_packed_cardinalities_tuple = tuple(self.inflated_packed_cardinalities_array)
        self.inflated_packed_cardinalities_indicies = np.repeat(np.arange(len(outcome_cardinalities)),
                                                                self.inflation_copies)
        self.column_count = self.inflated_packed_cardinalities_array.prod()
        self.shaped_packed_column_integers = np.arange(self.column_count).reshape(
            self.inflated_packed_cardinalities_tuple)

        # Expressible set related properties

        self.unpacked_conf_integers = np.array([list(
            np.arange(self.setting_cardinalities[packed_obs]) + np.array(self.setting_cardinalities)[
                self.inflated_packed_cardinalities_indicies[:packed_obs_index]].sum()) for packed_obs_index, packed_obs
            in enumerate(self.inflated_packed_cardinalities_indicies)],
            dtype=object)
        # print(self.unpacked_conf_integers)
        self.conf_setting_integers = np.array(
            [range(self.setting_cardinalities[obs]) for obs in range(self.observed_count)], dtype=object)
        # print(self.conf_setting_integers)
        self.conf_setting_indicies = self.conf_setting_integers[
            np.repeat(np.arange(len(self.conf_setting_integers)), self.inflation_copies)]
        self.ravelled_conf_setting_indicies = [i for j in self.conf_setting_indicies for i in j]
        #print(self.ravelled_conf_setting_indicies)

        self.ravelled_conf_var_indicies = np.repeat(np.arange(self.observed_count),
                                                    np.array(self.setting_cardinalities) * np.array(
                                                        self.inflation_copies))
        # print(self.ravelled_conf_var_indicies)

        self.unpacked_inflated_copies = [self.setting_cardinalities[observable] * self.inflation_copies[observable] for
                                         observable in range(self.observed_count)]
        self.inflated_unpacked_cardinalities = list(itertools.chain.from_iterable(
            [list(np.repeat(outcome_cardinalities[observable], self.unpacked_inflated_copies[observable])) for
             observable in range(self.observed_count)]))
        self.inflated_unpacked_cardinalities_tuple = tuple(self.inflated_unpacked_cardinalities)
        self.shaped_unpacked_column_integers = np.arange(self.column_count).reshape(
            self.inflated_unpacked_cardinalities_tuple)

        #print(self.packed_partitioned_eset)
        self.partitioned_unpacked_eset_candidates = [list(itertools.product(*list(self.unpacked_conf_integers[part])))
                                                     for part in self.packed_partitioned_eset]
        #print(self.partitioned_unpacked_eset_candidates)
        self.partitioned_unpacked_esets = list(itertools.product(*self.partitioned_unpacked_eset_candidates))
        #print(self.partitioned_unpacked_esets)
        self.flat_unpacked_esets = [[elem for part in eset for elem in part] for eset in
                                    self.partitioned_unpacked_esets]
        # print(self.flat_unpacked_esets)

    @cached_property
    def column_orbits(self):
        AMatrix = orbits_of_object_under_group_action(self.shaped_packed_column_integers,
                                                      self.inflation_group_elements).T
        AMatrix = np.compress(AMatrix[0] >= 0, AMatrix, axis=1)
        return AMatrix

    def _valid_outcomes(self, eset_part_candidate):
        observables = np.array(self.ravelled_conf_var_indicies)[np.array(eset_part_candidate)]
        settings_assignment = np.take(self.ravelled_conf_setting_indicies, np.array(eset_part_candidate))
        outcome_assignments = np.ndindex(tuple(np.take(self.outcomes_cardinalities, observables)))
        for outcomes_assigment in outcome_assignments:
            validity = True
            for setting_integer, setting_shape, parents_of in zip(settings_assignment,
                                                                  self.shaped_setting_cardinalities,
                                                                  self.inverse_directed_structure):
                settings_of_v = np.unravel_index(setting_integer, setting_shape)
                outcomes_relevant_to_v = np.compress(parents_of, outcomes_assigment)
                if not np.array_equal(settings_of_v[1:], outcomes_relevant_to_v):
                    validity = False
                    break
            yield validity

    def valid_outcomes(self, eset_part_candidate):
        return np.fromiter(self._valid_outcomes(eset_part_candidate), np.bool)

    def eset_unpacking_rows_to_keep(self, eset):
        validoutcomes = [self.valid_outcomes(part) for part in eset.partitioned_tuple_form]

        # # v = validoutcomes[-1]
        # v = validoutcomes[0]
        # for i in range(len(validoutcomes)-1):
        #     v = np.kron(v,validoutcomes[i+1])
        #     #v = np.kron(v, validoutcomes[len(validoutcomes) - 1 - i])

        v = functools.reduce(np.kron,validoutcomes)

        eset_kept_rows = np.flatnonzero(v.astype(np.int))

        eset.unpacking_rows_to_keep = eset_kept_rows
        # return eset_kept_rows

    def eset_discarded_rows_to_trash(self, eset):
        eset.which_rows_to_keep = np.intersect1d(eset.unpacking_rows_to_keep, eset.symmetry_rows_to_keep)
        size_of_eset_after_symmetry_and_unpacking = len(eset.which_rows_to_keep)
        eset.final_number_of_rows = size_of_eset_after_symmetry_and_unpacking
        # there_are_discarded_rows = (size_of_eset_after_symmetry_and_unpacking < size_of_eset)
        discarded_rows_to_the_back = np.full(eset.size_of_eset, 0, dtype=np.int)  # make it 0 instead of -1
        np.put(discarded_rows_to_the_back, eset.which_rows_to_keep,
               np.arange(size_of_eset_after_symmetry_and_unpacking) + 1)  # add the offset here
        discarded_rows_to_the_back = discarded_rows_to_the_back
        eset.discarded_rows_to_trash_no_offsets = discarded_rows_to_the_back
        # return discarded_rows_to_the_back

    def columns_to_unique_rows(self, eset):
        """"
        Since an unpacked eset has constant setting assignments, it means that the events in the rows
        corresponding to it are all locally orthogonal. That is, a column can hit AT MOST one row from
        this row block.
        As such. eset_discarded_rows_to_trash.take(eset.columns_to_rows) yields a 1d list where position indicates
        which column of the inflation matrix we are talking about and the value in the list indicates which row IN THIS BLOCK
        get "hit" by said column.
        """
        data_shape = self.shaped_unpacked_column_integers.shape
        reshaped_column_integers = self.shaped_unpacked_column_integers.transpose(
            MoveToBack(len(data_shape), eset.flat_form)).reshape((-1, eset.size_of_eset))
        encoding_of_columns_to_monomials = np.empty(self.shaped_unpacked_column_integers.size, np.int)
        encoding_of_columns_to_monomials[reshaped_column_integers] = np.arange(eset.size_of_eset)
        eset.columns_to_rows = encoding_of_columns_to_monomials
        # return encoding_of_columns_to_monomials

    def generate_symbolic_b_block(self,eset):
        eset.cardinalities=np.array(self.outcomes_cardinalities)[self.ravelled_conf_var_indicies[eset.flat_form]]
        size_of_each_part=[len(part) for part in eset.partitioned_tuple_form]
        loc_of_each_part=np.add.accumulate(np.array([0]+size_of_each_part))
        sym_b=[]
        for row in eset.which_rows_to_keep:
            eset_outcomes=self.ReverseMixedCardinalityBaseConversion(eset.cardinalities, row)
            product=''
            for part_index in range(len(eset.partitioned_tuple_form)):
                part_outcomes=eset_outcomes[loc_of_each_part[part_index]:loc_of_each_part[part_index+1]]
                part_settings=eset.settings_of[part_index]
                assignment_string=''.join(str(int(e)) for e in list(part_outcomes))+'|'+''.join(str(e) for e in list(part_settings))
                if size_of_each_part[part_index]<self.observed_count:
                    part_original_indices=eset.original_indicies[part_index]
                    marginal_of=''.join([chr(65+i) for i in part_original_indices])
                    string='P['+marginal_of+']('+assignment_string+')'
                else:
                    string='P('+assignment_string+')'
                product=product+string
            sym_b.append(product)
        eset.symbolic_b_block=np.array(sym_b)

    def generate_numeric_b_block(self, eset,rawdata):
        size_of_each_part=[len(part) for part in eset.partitioned_tuple_form]
        loc_of_each_part=np.add.accumulate(np.array([0]+size_of_each_part))
        num_b=[]
        #print(eset.which_rows_to_keep)
        #print(eset.settings_of)
        #print(eset.partitioned_tuple_form)
        for row in eset.which_rows_to_keep:
            eset_outcomes=self.ReverseMixedCardinalityBaseConversion(eset.cardinalities, row)
            #print(eset_outcomes)
            product=1
            #for part_index in range(len(eset.partitioned_tuple_form)):
            for part_index in range(len(eset.partitioned_tuple_form)):
                part_outcomes=eset_outcomes[loc_of_each_part[part_index]:loc_of_each_part[part_index+1]]
                #print(part_outcomes)
                #print(eset.settings_of)
                part_settings=eset.settings_of[part_index]
                #print(part_settings)
                if size_of_each_part[part_index]<self.observed_count:
                    part_original_indices=np.array(eset.original_indicies[part_index])
                    part_settings_template=np.full(self.observed_count,0)
                    part_settings_template[part_original_indices]=part_settings
                    part_settings=part_settings_template
                    relevant_sets_and_outs=[''.join(str(e) for e in self.ReverseMixedCardinalityBaseConversion(self.all_moments_shape, dist)[range(self.observed_count)+list(np.array(part_original_indices)+self.observed_count)]) for dist in self.knowable_original_probabilities]
                    probs_to_be_summed=rawdata[np.where(relevant_sets_and_outs==''.join(str(e) for e in list(part_settings)+list(part_outcomes)))[0]]
                    marginal=probs_to_be_summed.sum()
                    """
                    part_original_indices=eset.original_indicies[part_index]
                    
                    filled_data=np.zeros(np.prod(np.array(self.all_moments_shape)),dtype=np.float)
                    np.put(filled_data,self.knowable_original_probabilities,rawdata)
                    data_reshaped=filled_data.reshape(self.all_moments_shape)
                    
                    marginalised_indecies_ellipsis=np.full(self.observed_count,Ellipsis)
                    np.put(marginalised_indecies_ellipsis,np.array(part_original_indices),part_outcomes)
                    
                    marginal=data_reshaped[tuple(list(marginalised_indecies_ellipsis)+list(part_settings))].sum()
                    """
                    
                else:
                    #print(self.knowable_original_probabilities)
                    #print(np.array(list(part_settings)+list(part_outcomes)))
                    #print(self.MixedCardinalityBaseConversion(eset.cardinalities, np.array(list(part_settings)+list(part_outcomes))))
                    #print(np.where(self.knowable_original_probabilities==int(self.MixedCardinalityBaseConversion(eset.cardinalities, np.array(list(part_settings)+list(part_outcomes)))))[0])
                    #print(len(rawdata))
                    marginal=rawdata[np.where(np.array(self.knowable_original_probabilities).ravel()==int(self.MixedCardinalityBaseConversion(self.all_moments_shape, np.array(list(part_settings)+list(part_outcomes)))))[0]][0]
                    #print(marginal,'------')
                product=product*marginal
                #print(product,'-------')
            num_b.append(product)
            #print(num_b)
        
        eset.numeric_b_block = np.array(num_b)

    class expressible_set:
        def __init__(self, partitioned_eset):
            self.partitioned_tuple_form = partitioned_eset
            self.flat_form = np.array([elem for part in self.partitioned_tuple_form for elem in part])

    @cached_property
    def expressible_sets(self):
        esets = tuple(map(self.expressible_set, self.partitioned_unpacked_esets))
        offset = 0
        er=0
        for eset in esets:
            eset.original_indicies = tuple(
                [tuple(self.ravelled_conf_var_indicies[np.array(part)]) for part in eset.partitioned_tuple_form])
            eset.settings_of = tuple([tuple(np.array(self.ravelled_conf_setting_indicies)[np.array(part)]) for part in
                                      eset.partitioned_tuple_form])
            eset.shape_of_eset = np.take(np.array(self.inflated_unpacked_cardinalities), eset.flat_form)
            eset.size_of_eset = eset.shape_of_eset.prod()
            eset.symmetry_rows_to_keep = self.eset_symmetry_rows_to_keep
            self.eset_unpacking_rows_to_keep(eset)
            self.eset_discarded_rows_to_trash(eset)
            self.columns_to_unique_rows(eset)
            self.generate_symbolic_b_block(eset)
            #print(eset.symbolic_b_block)
            # setting offsets
            offset_array = np.zeros(len(eset.discarded_rows_to_trash_no_offsets), dtype=np.int)
            offset_array[np.flatnonzero(eset.discarded_rows_to_trash_no_offsets)] = offset
            eset.discarded_rows_to_trash = eset.discarded_rows_to_trash_no_offsets + offset_array
            offset = offset + eset.final_number_of_rows
        return esets

    @cached_property
    def AMatrix(self):
        single_shape = list(self.column_orbits.shape)
        amatrices = np.empty([len(self.expressible_sets)] + single_shape, np.int)
        for i, eset in enumerate(self.expressible_sets):
            amatrices[i] = eset.discarded_rows_to_trash.take(eset.columns_to_rows).take(self.column_orbits)
        single_shape[0] = -1
        amatrices = amatrices.reshape(tuple(single_shape))
        #NEW: Adding filter to remove duplicate columns
        amatrices.sort(axis=0)
        print("Filtering duplicate columns...")
        amatrices = amatrices[:, amatrices.any(axis=0)] #Removes columns hitting only trash rows
        amatrices = amatrices[:, np.lexsort(amatrices)] #Sorts the columns, so we can use justseen instead of everseen (which is slightly faster than everseen + I think less memory)
        amatrices = np.fromiter(itertools.chain.from_iterable(more_itertools.unique_justseen(amatrices.T, key=tuple)), int).reshape((-1, len(amatrices))).T
        amatrices = np.unique(amatrices, axis=1)
        return amatrices

    @cached_property
    def inflation_matrix(self):
        InfMat = SparseMatrixFromRowsPerColumn(self.AMatrix)
        return InfMat
    
    @cached_property
    def symbolic_b(self):
       return np.hstack([eset.symbolic_b_block for eset in self.expressible_sets])

class inflation_LP(inflation_problem):

    def __init__(self, hypergraph, inflation_orders, directed_structure, outcome_cardinalities,
                 private_setting_cardinalities,rawdata,solver):

        inflation_problem.__init__(self,hypergraph, inflation_orders, directed_structure, outcome_cardinalities,
                            private_setting_cardinalities)
        [self.generate_numeric_b_block(eset,rawdata) for eset in self.expressible_sets]
        self.numeric_b=np.hstack([eset.numeric_b_block for eset in self.expressible_sets])
        
        self.InfMat = self.inflation_matrix
        
        if not ((solver == 'moseklp') or (solver == 'CVXOPT') or (solver == 'mosekAUTO')):
            raise TypeError("The accepted solvers are: 'moseklp', 'CVXOPT' and 'mosekAUTO'")

        if solver == 'moseklp':

            self.solve = InfeasibilityCertificate(self.InfMat, self.numeric_b)

        elif solver == 'CVXOPT':

            self.solve = InflationLP(self.InfMat, self.numeric_b)

        elif solver == 'mosekAUTO':

            self.solve = InfeasibilityCertificateAUTO(self.InfMat, self.numeric_b)

        self.tol = self.solve[
                       'gap'] / 10  # TODO: Choose better tolerance function. This is yielding false incompatibility claims.
        self.yRaw = np.array(self.solve['x']).ravel()

    def WitnessDataTest(self, y):
        IncompTest = (np.amin(y) < 0) and (np.dot(y, self.numeric_b) < self.tol)
        if IncompTest:
            print('Distribution Compatibility Status: INCOMPATIBLE')
        else:
            print('Distribution Compatibility Status: COMPATIBLE')
        return IncompTest


    def Inequality(self,output=[]):
        # Modified Feb 2, 2021 to pass B_symbolic as an argument for Inequality
        # Modified Feb 25, 2021 to accept custom output options from user
        if self.WitnessDataTest(self.yRaw):
            y = IntelligentRound(self.yRaw, self.InfMat)
            
            if output==[]:
            
                idxtally=indextally(y)
                symtally=symboltally(indextally(y),self.symbolic_b)
                ineq_as_str=inequality_as_string(y,self.symbolic_b)

                print("Writing to file: 'inequality_output.json'")

                returntouser = {
                    'Raw solver output': self.yRaw.tolist(),
                    'Inequality as string': ineq_as_str,
                    'Coefficients grouped by index': idxtally,
                    'Coefficients grouped by symbol': symtally,
                    'Clean solver output': y.tolist()
                }
                f = open('inequality_output.json', 'w')
                print(json.dumps(returntouser), file=f)
                f.close()
                
            else:
                returntouser={}
                
                if 'Raw solver output' in output:
                    returntouser['Raw solver output']=self.yRaw.tolist()
                if 'Inequality as string' in output:
                    ineq_as_str=inequality_as_string(y,self.symbolic_b)
                    returntouser['Inequality as string']=ineq_as_str
                if 'Coefficients grouped by index' in output:
                    idxtally=indextally(y)
                    returntouser['Coefficients grouped by index']=idxtally
                if 'Coefficients grouped by symbol' in output:
                    symtally=symboltally(indextally(y),self.symbolic_b)
                    returntouser['Coefficients grouped by symbol']=symtally
                if 'Clean solver output' in output:
                    returntouser['Clean solver output']=y.tolist()
                
                f = open('inequality_output.json', 'w')
                print(json.dumps(returntouser), file=f)
                f.close()
                
            return returntouser
        else:
            return print('Compatibility Error: The input distribution is compatible with given inflation order test.')

if __name__ == '__main__':
    
    """
    hypergraph = np.array([[1, 1, 0,0], [0, 1, 1,0],[0,0,1,1]])
    #hypergraph=np.array([[1, 1, 0], [0, 1, 1]])
    directed_structure = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    directed_structure=np.array([[0,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,1,0]])
    outcome_cardinalities = (2, 2, 2,2)
    private_setting_cardinalities = (2, 1, 1,1)

    inflation_orders = [2, 3,3]
    p = ((0, 4, 12), (2, 10, 14))
    # e=expressible_sets(hypergraph,inflation_orders,directed_structure, outcome_cardinalities, private_setting_cardinalities)
    inf = inflation_problem(hypergraph, inflation_orders, directed_structure, outcome_cardinalities,
                            private_setting_cardinalities)
    # print(e.AMatrix())
    print(inf.inflation_matrix.shape)
    rawdata=np.arange(len(inf.knowable_original_probabilities),dtype=np.float)
    [inf.generate_numeric_b_block(eset,rawdata) for eset in inf.expressible_sets]
    print(inf.symbolic_b)
    print(np.hstack([eset.numeric_b_block for eset in inf.expressible_sets]).shape)
    #print(inf.b_vector.shape)
    # print(inf.expressible_sets[3].discarded_rows_to_trash)
    # print(inf.expressible_sets[3].offset_array)
    """
    
    #TRIANGLE SCENARIO
    
    hypergraph = np.array([[1,1,0],[0,1,1],[1,0,1]])
    directed_structure = np.array([[0,0,0],[0,0,0],[0,0,0]])
    outcome_cardinalities = (4,4,4)
    private_setting_cardinalities = (1,1,1)
    inflation_orders = [2,2,2]
    rawdata = np.array([0.12199995751046305, 0.0022969343799089472, 0.001748319476328954, 3.999015242496535e-05, 0.028907881434196828, 0.0005736087488455967, 0.0003924033706699725, 1.1247230369521505e-05, 0.0030142577390317635, 0.09234476010282468, 4.373922921480586e-05, 0.0014533921021948346, 0.0007798079722868244, 0.024091567451515063, 1.1247230369521505e-05, 0.0003849052170902915, 0.020774884184769502, 0.000396152447459813, 0.0003049249122403608, 4.998769053120669e-06, 0.10820335492385, 0.0020794879260981982, 0.0015546171755205281, 2.4993845265603346e-05, 0.0006260958239033638, 0.020273757587194154, 7.498153579681003e-06, 0.0003374169110856452, 0.0028942872817568676, 0.08976414557915113, 2.624353752888351e-05, 0.0012984302615480939, 0.002370666223442477, 4.7488306004646356e-05, 0.0999928767540993, 0.001957018084296742, 0.0006198473625869629, 8.747845842961171e-06, 0.02636975644747481, 0.0005198719815245496, 1.4996307159362007e-05, 0.000403650601039494, 0.0005498645958432735, 0.017359475229224805, 7.123245900696953e-05, 0.002346922070440154, 0.0033754188031197316, 0.10295964618712641, 0.00038740460161685187, 7.498153579681003e-06, 0.01608353942841575, 0.000306174604503641, 0.0021319750011559654, 4.248953695152569e-05, 0.09107007399427891, 0.001860791780024169, 5.998522863744803e-05, 0.0018395470115484063, 0.002570616985567304, 0.0766411271224461, 1.874538394920251e-05, 0.00048238121362614454, 0.0006410921310627258, 0.020223769896662948])
    
    inf = inflation_problem(hypergraph, inflation_orders, directed_structure, outcome_cardinalities,
                            private_setting_cardinalities)
    
    print(inf.symbolic_b)
    print(inf.inflation_matrix.shape)
    
    solver='moseklp'
    inequality=inflation_LP(hypergraph, inflation_orders, directed_structure, outcome_cardinalities,
                            private_setting_cardinalities,rawdata,solver).Inequality()
    
