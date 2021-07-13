#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions.
"""

from __future__ import absolute_import
import numpy as np
# from numba import njit
from scipy.sparse import coo_matrix
from functools import lru_cache
from operator import itemgetter


def Deduplicate(
        ar):  # Alternatives include unique_everseen and panda.unique, see https://stackoverflow.com/a/15637512 and https://stackoverflow.com/a/41577279
    (vals, idx) = np.unique(ar, return_index=True)
    return vals[np.argsort(idx)]

#
# @njit
def MoveToFront(num_var, ar):
    return np.hstack((ar, np.delete(np.arange(num_var), ar)))

#
# @njit
def MoveToBack(num_var, ar):
    return np.hstack((np.delete(np.arange(num_var), ar), ar))

#
# @njit
# @lru_cache(2)
def GenShapedColumnIntegers(range_shape):
    return np.arange(0, np.prod(np.array(range_shape)), 1, np.int32).reshape(range_shape)
    #return np.arange(np.array(range_shape).prod()).reshape(range_shape)


# def PositionIndex(arraywithduplicates):
#    arraycopy=np.empty_like(arraywithduplicates)
#    u=np.unique(arraywithduplicates)
#    arraycopy[u]=np.arange(len(u))
#    return arraycopy.take(arraywithduplicates)

def columns_to_unique_rows_old(cardinalities, subset_indices):
    data = GenShapedColumnIntegers(cardinalities)
    subset_size = np.take(cardinalities, subset_indices).prod()
    reshaped_column_integers = data.transpose(
        MoveToBack(len(cardinalities), subset_indices)).reshape((-1, subset_size))
    encoding_of_columns_to_monomials = np.empty(np.prod(cardinalities), np.int)
    encoding_of_columns_to_monomials[reshaped_column_integers] = np.arange(subset_size, dtype=np.int)
    return encoding_of_columns_to_monomials

def columns_to_unique_rows(cardinalities, subset_indices):
    rawshape = np.ones(len(cardinalities), dtype=np.uint)
    mini_shape = np.take(cardinalities, subset_indices)
    rawshape[subset_indices] = mini_shape
    subset_size = mini_shape.prod()
    return np.broadcast_to(
        np.arange(subset_size, dtype=np.int).reshape(mini_shape).transpose(np.argsort(subset_indices)).reshape(rawshape),
        cardinalities
    ).ravel()


def PositionIndex(arraywithduplicates):
    # u,inv,idx=np.unique(arraywithduplicates,return_inverse=True)[1]
    return np.unique(arraywithduplicates, return_inverse=True)[1]

#
# @njit
def reindex_list(ar):
    seenbefore = np.full(np.max(ar) + 1, -1)
    newlist = np.empty(len(ar), np.uint)
    currentindex = 0
    for idx, val in enumerate(ar):
        if seenbefore[val] == -1:
            seenbefore[val] = currentindex
            newlist[idx] = currentindex
            currentindex += 1
        else:
            newlist[idx] = seenbefore[val]
    return newlist

def SparseMatrixFromRowsPerColumn(OnesPositions, sort_columns=True):
    #Assumes that OnesPositions is a 2d numpy array of integers.
    #First dimension indicates which ORBIT we are considering.
    #Second dimension indicates which COLUMN we are listing rows for.
    there_are_discarded_rows=np.any(OnesPositions==0)
    if not there_are_discarded_rows:
        OnesPositions=OnesPositions-1
    
    columncount = OnesPositions.shape[-1]
    rowcount = int(np.amax(OnesPositions)) + 1
    if sort_columns:
        ar_to_broadcast = np.lexsort(OnesPositions)
    else:
        ar_to_broadcast = np.arange(columncount)
    columnspec = np.broadcast_to(ar_to_broadcast, OnesPositions.shape).ravel()
    sparse_matrix_output = coo_matrix((np.ones(OnesPositions.size, np.uint), (OnesPositions.ravel(), columnspec)),
                      (rowcount, columncount), dtype=np.uint)
    if there_are_discarded_rows:
        return sparse_matrix_output.asformat('csr', copy=False)[1:]
    else:
        return sparse_matrix_output.asformat('csr', copy=False)

def partsextractor(thing_to_take_parts_of, indices):
    if len(indices) == 0:
        return tuple()
    elif len(indices) == 1:
        return (itemgetter(*indices)(thing_to_take_parts_of),)
    else:
        return itemgetter(*indices)(thing_to_take_parts_of)



if __name__ == '__main__':
    print(SparseMatrixFromRowsPerColumn(np.array([[1,1,3],[2,5,3]])).tocsr())

    print(columns_to_unique_rows_old(np.broadcast_to(2, 6), [0, 3, 1, 4]))
    print(columns_to_unique_rows(np.broadcast_to(2, 6), [0, 3, 1, 4]))
    print(np.array_equal(
        columns_to_unique_rows_old(np.broadcast_to(2, 12), [0, 4, 8, 1, 5, 9, 2, 6, 10]),
        columns_to_unique_rows(np.broadcast_to(2, 12), [0, 4, 8, 1, 5, 9, 2, 6, 10])
    ))