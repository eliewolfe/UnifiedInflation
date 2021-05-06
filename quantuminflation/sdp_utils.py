from typing import Tuple

import numpy as np
import picos
from scipy.io import loadmat

import picos

def solveSDP(filename_SDP_info: str, use_semiknown: bool = True, verbosity: int = 0, solver: str = 'cvxopt') -> Tuple[picos.modeling.solution.Solution, float]:
    """
    Solves an SDP and returns the solution object and the optimum objective. Takes as input the filename for a .mat file
    with the different variables that are needed.
    """
    SDP_data = loadmat(filename_SDP_info)
    positionsmatrix = SDP_data['G']
    nr_variables = np.max(positionsmatrix)  # We need to do this before offsetting the MATLAB index

    positionsmatrix = positionsmatrix - 1  # Offset MATLAB +1 index

    known_vars_array = SDP_data['known_moments'][0]
    semiknown_vars_array = SDP_data['propto']

    known_vars = {key: val for key, val in enumerate(known_vars_array)}
    semiknown_vars = {key - 1: [val1, val2 - 1] for key, val1, val2 in semiknown_vars_array}

    nr_known_vars = len(known_vars)

    if use_semiknown:
        nr_semiknown_vars = len(semiknown_vars)
        nr_unknown_vars = nr_variables - nr_known_vars - nr_semiknown_vars
        # idxoffset will be used when calling the picos variables
        # for example if G[i,j] = 77
        # then at i,j we will place variable[77 - idxoffset]
        # because variable will be a vector of size the unknown vars
        idxoffset = nr_known_vars + nr_semiknown_vars
    else:
        semiknown_vars = {}
        nr_unknown_vars = nr_variables - nr_known_vars
        idxoffset = nr_known_vars

    # print("counters", nr_variables, nr_known_vars, nr_unknown_vars, nr_semiknown_vars, idxoffset)

    unknownvariables = picos.RealVariable("v", int(nr_unknown_vars))

    lam = picos.RealVariable("lam")

    P = picos.Problem()

    dim, dim = positionsmatrix.shape

    rows = []
    for i in range(dim):
        j = 0
        idx = positionsmatrix[i, j]
        try:
            var = known_vars[idx]
            if isinstance(var,float):
                row = picos.Constant(var)
            else:
                row = var
        except KeyError:
            if use_semiknown:
                try:
                    value = semiknown_vars[idx]
                    row = value[0] * unknownvariables[value[1] - idxoffset]
                except KeyError:
                    row = unknownvariables[idx - idxoffset]
            else:
                row = unknownvariables[idx - idxoffset]
        for j in range(1, dim):
            idx = positionsmatrix[i, j]
            try:
                row = (row & known_vars[idx])
            except KeyError:
                if use_semiknown:
                    try:
                        value = semiknown_vars[idx]
                        row = row & value[0] * unknownvariables[value[1] - idxoffset]
                    except KeyError:
                        row = row & unknownvariables[idx - idxoffset]
                else:
                    row = row & unknownvariables[idx - idxoffset]
        rows.append(row)
    momentmatrix = rows[0]
    for row_idx in range(1,len(rows)):
        momentmatrix = momentmatrix // rows[row_idx]

    P.add_constraint(momentmatrix - lam * picos.Constant(np.eye(dim)) >> 0)

    P.set_objective("max", lam)

    sol = P.solve(solver=solver, verbosity=verbosity)

    return sol, lam.value
