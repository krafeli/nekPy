import time, functools, datetime
from pathlib import Path

print = functools.partial(print, flush=True)

from nekPy.utils.misc import logger
from nekPy.utils.bash import run_command, mkdir
from nekPy.preprocessor import PreProcessor
from nekPy.launcher import Launcher
from nekPy.optimization import Optimization
from nekPy.optimization.objectives import misfit


#config  = Path("/home/felix.kranz/thermoSurf/3d/opt/config")
#outbase = Path("/home/felix.kranz/thermoSurf/3d/opt/out")
#blfile  = Path("/home/felix.kranz/thermoSurf/3d/opt/data/Re1e6.pkl")
#obsfile = Path("/home/felix.kranz/thermoSurf/3d/opt/data/Rek900_xc005.f00001")

config  = Path("/scratch/projects/hbi00065/3d/DU95W180/opt/config")
outbase = Path("/scratch/projects/hbi00065/3d/DU95W180/opt/out")
blfile  = Path("/scratch/projects/hbi00065/3d/DU95W180/opt/data/Re1e6.pkl")
obsfile = Path("/scratch/projects/hbi00065/3d/DU95W180/opt/data/Rek900_xc005.f00001")

defaults = {'Rek': 900, 'xc': 0.05, 'eta': 1.0}

bounds = {
    'Rek': (250., 2500.),
    #'xc' : (0.03, 0.3),
    #'eta': (0.5,  5.0),
         }

mode = len(bounds)
keys = list(bounds.keys())
name = '_'.join(['Cf_fitting_normalized'] +  keys)

out = outbase / str(mode) / name
mkdir(out)
log = logger(out)

opt = Optimization(out, bounds=list(bounds.values()), neval=50, tol=1e-2)
#opt = Optimization(out, algorithm='scipy', x0=287., bounds=[(250., 2500.)], method='SLSQP', 
#                   options={'eps': 50., 'maxiter': 25, 'disp': True})


def objective(x, opt):
    
    params = defaults | dict(zip(keys, x))
    Rek  = float(params['Rek'])
    xloc = float(params['xc'])
    eta  = float(params['eta'])
    
    print(f"Considering new params: {params}")

    # Setup new simulation
    print(f"Setting up simulation...")
    outit = opt.outdir / str(opt.iters)
    pre = PreProcessor(outit, usr=config/'loc3.usr', par=config/'loc3.par', size=config/'SIZE',
                       #re2=config/'loc3.re2', 
                       #ma2=config/'loc3.ma2'
                       )
    
    # generate mesh
    print(f"Generating mesh (eta={eta:.2f})...")
    pre.generate_mesh(k=1., eta=eta, 
                      Lx=75., Ly=10., Lz=20., Lin=15.,
                      size_k = 0.075 + (eta / 0.5) * 0.025,
                      size_x = 1.10,
                      size_y = 1.00,
                      size_z = 0.75,
                      size_yk= 0.15, 
                      )
    print(F"Computing connectivity map...")
    pre.msh2nek(periodic_pairs=[(3, 4)])
    
    print(f"Setting parameters (Rek={Rek:.0f})...")
    pre.parameters.set('VELOCITY', 'viscosity', -Rek)
    print(f"Generating BCs and IC (xc={xloc:.2f})...")
    pre.generate_bc(blfile, mode='blade', loc=xloc, Lin=15.)
    pre.generate_ic(blfile, loc=xloc,     Lin=15., Lout=75.)
    
    # launch the sim
    print("Submitting simulation. Waiting to finish...")
    while True:
        done_flag = outit / 'done.flag'
        done_flag.unlink(missing_ok=True)

        launcher = Launcher(outit)
        launcher.submit(slurm_script=config/'run.slurm', job_name=f'Rek{Rek:.0f}_xc{xloc:.2f}_eta{eta:.1f}')

        # Wait until Slurm job creates done.flag
        while not done_flag.exists(): time.sleep(10)

        # check if the simulation succeeded
        sim_res = outit / 'avg0.f00001'
        if sim_res.exists():
            print("Simulation done. Postprocessing...")
            break
        print("Simulation failed. Resubmitting...")

    J = misfit(sim_res, obsfile, bounds=[10., 60., -10., 10.], nuobs=1./900., verbose=False)

    # cleanup
    run_command('rm -rf obj/ build.log *.msh run.sh makenek.log makefile done.flag *loc30.f*', outit)

    print(f"Obtained function value: J({params})={J}\n")
    return J

opt.set_objective(objective)

print(datetime.datetime.now().isoformat(timespec='seconds', sep=' '))
print("Running optimization...")
result = opt.run()
print("Optimization done. Result:")
print(result)
print(datetime.datetime.now().isoformat(timespec='seconds', sep=' '))
