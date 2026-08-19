import datetime

import numpy as np

from pathlib import Path

from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from utils import run_command, print, logger, write_json, copy, read_par, write_par
from nekPy.optimization.custom_callbacks import ExpectedMinimumStopper
from inflow import write_inflow

np.random.seed(69)

base = Path('/home/felix/run/3d/opt/base')
out = Path('/home/felix/run/3d/opt/test')
bl = Path('/home/felix/run/2d/DU95W180/glob/Re1e6/bl/bl.pkl')

opti_params = {'J': 'shear',
               'bounds': [(200, 2000)],
               'x0': [800],
               'y0': None,
               'opts': {
                   'disp': True,
                   'neval': 20,
                   'nstarts': 3,
                   'tol': 1e-2,
                        },
               'run_expected': True
               }

info = {'Nfeval': 0, 'all_vecs': [], 'J': []}

#
out.mkdir(parents=True, exist_ok=True)
# logging
logger(out)
write_json(opti_params, out/'opti_params.json')

# get the base simulation
copy(base, out / base.name)

def objective(x):
    global opti_params, info

    x = np.asarray(x)
    print(f"Considering new params: {x}")

    # setup new sim
    ################
    it = info['Nfeval']
    outit = out / str(it)
    copy(base, outit)
    # write par file
    parfile = list(outit.glob('*.par'))[0]
    cfg = read_par(parfile)
    cfg['VELOCITY']['viscosity'] = -float(x[0])
    write_par(cfg, parfile)

    # generate BCs
    write_inflow(float(x[0]), bl, outit)

    # start the simulation
    name = parfile.name.split('.')[0]
    run_command(f'echo "{name}\n{str(outit)}" > SESSION.NAME', outit)
    run_command('mpirun -np 8 ./nek5000 > log.txt', outit)

    # postproc
    avgfile = list(outit.glob('avg*.f*'))[0]


    info['Nfeval'] += 1
    return -float(x[0])


print("Running optimization...")
print(datetime.datetime.now().isoformat(timespec='seconds', sep=' '))
res = gp_minimize(objective,
                  opti_params['bounds'],
                  x0=opti_params['x0'], y0=opti_params['y0'],
                  n_initial_points=opti_params['opts']['nstarts'],
                  initial_point_generator='lhs',
                  verbose=opti_params['opts']['disp'],
                  acq_func='EI',
                  callback=[CheckpointSaver(out / 'checkpoint.pkl', store_objective=False),
                            ExpectedMinimumStopper(rel_tol=opti_params['opts']['tol'], save=out)],)
print(datetime.datetime.now().isoformat(timespec='seconds', sep=' '))
