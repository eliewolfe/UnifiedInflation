import numpy as np

#Acknowledgement to https://digitalcommons.njit.edu/cgi/viewcontent.cgi?article=2820&context=theses

def transitive_closure(adjmat):
    n=len(adjmat)
    closure_mat = np.bitwise_or(np.asarray(adjmat, dtype=bool),np.identity(n, dtype=bool))
    while n>0:
        n = np.floor_divide(n,2)
        next_closure_mat = np.matmul(closure_mat, closure_mat)
        if np.array_equal(closure_mat,next_closure_mat):
            break
        else:
            closure_mat=next_closure_mat
    return np.bitwise_and(closure_mat,np.invert(np.identity(len(adjmat), dtype=bool)))

def transitive_reduction(adjmat):
    n = len(adjmat)
    closure_mat=transitive_closure(adjmat)
    #closure_minus_identity = np.bitwise_and(transitive_closure(adjmat),np.invert(np.identity(n, dtype=bool)))
    return np.bitwise_and(closure_mat,np.invert(np.matmul(closure_mat, closure_mat)))

if __name__ == '__main__':
    adjmat = np.asarray(np.array(
    [[1,0,	1,	1,	0,	0],
    [0,	1,	0,	0,	1,	1],
    [0,	1,	1,	0,	0,	0],
    [0,	0,	0,	1,	0,	0],
    [0,	0,	0,	0,	1,	0],
    [0,	0,	0,	0,	0,	1]],dtype=int),dtype=bool)+np.identity(6, dtype=bool)

    print(transitive_closure(adjmat).astype(int))
    print(transitive_reduction(adjmat).astype(int))
    # print(np.linalg.matrix_power(adjmat, 2).astype(np.int_))