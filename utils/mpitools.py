from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank0 = (comm.Get_rank() == 0)

def glmin(x, comm=MPI.COMM_WORLD):
    if np.isscalar(x):
        local = x
    else:
        local = np.min(np.asarray(x))
    return comm.allreduce(local, op=MPI.MIN)


def glmax(x, comm=MPI.COMM_WORLD):
    if np.isscalar(x):
        local = x
    else:
        local = np.max(np.asarray(x))
    return comm.allreduce(local, op=MPI.MAX)


def glmm(x, comm=MPI.COMM_WORLD):
    return glmin(x, comm), glmax(x, comm)


def glsum(x, comm=MPI.COMM_WORLD):
    if np.isscalar(x):
        local = x
    else:
        local = np.sum(np.asarray(x))
    return comm.allreduce(local, op=MPI.SUM)


def glmn(x, comm=MPI.COMM_WORLD):
    if np.isscalar(x):
        local_sum = x
        local_n = 1
    else:
        x = np.asarray(x)
        local_sum = np.sum(x)
        local_n = x.size

    return comm.allreduce(local_sum, op=MPI.SUM) / comm.allreduce(local_n, op=MPI.SUM)


def printl(*args, comm=MPI.COMM_WORLD, root=0, **kwargs):
    if comm.Get_rank() == root:
        print(*args, **kwargs)