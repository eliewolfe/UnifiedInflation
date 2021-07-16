from __future__ import absolute_import
import numpy as np
from inflation.internal_functions.groups import minimize_object_under_group_action, dimino_sympy
from inflation.internal_functions.utilities_ import PositionIndex
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.group_constructs import DirectProduct

def unique_product(interpretation, product_dims):
    clean_interpretation = PositionIndex(list(map(str,interpretation)))
    #print(clean_interpretation)
    object = np.arange(np.prod(product_dims)).reshape(product_dims)
    group = group_from_clean_interpretation(clean_interpretation)
    minimize_object_under_group_action(object, group)
    #print(object)
    return np.unique(object)

def group_from_clean_interpretation(clean_interpretation):
    """
    :param clean_interpretation: 1d numpy array of integers
    :return: product of symmetry groups
    """
    group_generators = []
    for e in range(clean_interpretation.max()+1):
        # base_array = np.array(clean_interpretation.size)
        positions_of_element = np.flatnonzero(clean_interpretation == e)
        if len(positions_of_element) >= 2:
            group_generators.append(Permutation([positions_of_element[:2].tolist()], size=clean_interpretation.size))
        if len(positions_of_element) >= 3:
            group_generators.append(Permutation([positions_of_element.tolist()], size=clean_interpretation.size))
    if len(group_generators)>=1:
        return list(PermutationGroup(group_generators).generate_schreier_sims(af=True))
    else:
        return []

# def group_from_clean_interpretation(clean_interpretation):
#     """
#     :param clean_interpretation: 1d numpy array of integers
#     :return: product of symmetry groups
#     """
#     groups_to_take_product = []
#     for e in range(clean_interpretation.max()+1):
#         group_generators = []
#         # base_array = np.array(clean_interpretation.size)
#         positions_of_element = np.flatnonzero(clean_interpretation == e)
#         if len(positions_of_element) >= 2:
#             group_generators.append(Permutation([positions_of_element[:2].tolist()], size=clean_interpretation.size))
#         if len(positions_of_element) >= 3:
#             group_generators.append(Permutation([positions_of_element.tolist()], size=clean_interpretation.size))
#         groups_to_take_product.append(PermutationGroup(group_generators))
#     if len(groups_to_take_product)>=1:
#         group =  list(DirectProduct(*groups_to_take_product).generate_schreier_sims(af=True))
#         print(group)
#         return group
#     else:
#         return []

if __name__ == '__main__':
    # print(unique_product([[1,2,3],[1,2,3],[4,5,6],[1,2,3],[4,5,6]],[3,3,2,3,2]))
    # print(unique_product([[1, 2, 3], [1, 2, 3]], [2,2]))
    print(unique_product([[1,2,3],[1,2,3],[4,5,6]],[3,3,1]))