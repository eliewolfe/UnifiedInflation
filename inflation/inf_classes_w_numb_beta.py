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
if __name__ == '__main__':
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from internal_functions.groups import dimino_sympy, orbits_of_object_under_group_action, \
    minimize_object_under_group_action
from internal_functions.utilities import MoveToFront, MoveToBack, SparseMatrixFromRowsPerColumn
import functools



class inflated_hypergraph:

    def __init__(self, hypergraph, inflation_orders):
        self.latent_count = len(hypergraph)
        self.observed_count = len(hypergraph[0])
        self.root_structure = [list(np.nonzero(hypergraph[:, observable])[0]) for observable in
                               range(self.observed_count)]
        self.inflations_orders = np.array(inflation_orders)
        self._latent_ancestors_of = [self.inflations_orders.take(np.nonzero(hypergraph[:, observable])[0]) for
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
        DAG.__init__(self, hypergraph, directed_structure, outcome_cardinalities, private_setting_cardinalities)
        
        
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
                    part_original_indices=eset.original_indicies[part_index]
                    relevant_sets_and_outs=[''.join(str(e) for e in self.ReverseMixedCardinalityBaseConversion(self.all_moments_shape, dist)[list(part_original_indices)+list(np.array(part_original_indices)+size_of_each_part[part_index])]) for dist in self.knowable_original_probabilities]
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
        return amatrices.reshape(tuple(single_shape))

    @cached_property
    def inflation_matrix(self):
        InfMat = SparseMatrixFromRowsPerColumn(self.AMatrix)
        return InfMat
    
    @cached_property
    def symbolic_b(self):
       return np.hstack([eset.symbolic_b_block for eset in self.expressible_sets])

if __name__ == '__main__':
    hypergraph = np.array([[1, 1, 0], [0, 1, 1]])
    directed_structure = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    outcome_cardinalities = (2, 2, 2)
    private_setting_cardinalities = (2, 1, 1)

    inflation_orders = [2, 1]
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
