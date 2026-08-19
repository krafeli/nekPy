from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
rank0 = rank == 0

def print0(*args, **kwargs):
    if rank0: print(*args, **kwargs)