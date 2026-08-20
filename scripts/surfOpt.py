import time, datetime
import numpy as np
from pathlib import Path
import functools
print = functools.partial(print, flush=True)

from nekPy.utils.misc import logger
from nekPy.utils.bash import run_command, mkdir
from nekPy.preprocessor import PreProcessor, BoundaryCondition
from nekPy.launcher import Launcher
from nekPy.optimization import Optimization
from nekPy.optimization.objectives import misfit


config = Path("/home/felix.kranz/thermoSurf/3d/opt/config")
out = Path("/home/felix.kranz/thermoSurf/3d/opt/out/test_opt")
blfile = Path("/home/felix.kranz/thermoSurf/3d/opt/data/Re1e6.pkl")
obsfile = Path("/home/felix.kranz/thermoSurf/3d/opt/data/Rek800_xc005.f00001")

mkdir(out)
log = logger(out)

opt = Optimization(out, bounds=[(200, 2000)], neval=25)

def objective(x, opt):
    x = np.asarray(x)
    Rek = float(x[0])
    xloc = float(x[1]) if len(x) > 1 else 0.05
    print(f"Considering new params: {x}")

    # Setup new simulation
    outit = opt.outdir / str(opt.iters)
    pre = PreProcessor(outit,  usr=config/'loc3.usr', par=config/'loc3.par', size=config/'SIZE',
                       re2=config/'loc3.re2', ma2=config/'loc3.ma2')
    pre.parameters.set('VELOCITY', 'viscosity', -Rek)
    pre.generate_bc(blfile, mode='blade', loc=xloc, Lin=15.)

    # launch the sim
    launcher = Launcher(outit)
    launcher.submit(slurm_script=config/'run.slurm')
    print("\nSimulation submitted. Waiting to finish...")

    # check if it finished
    while True:
        if (outit / 'done.flag').exists(): break
        time.sleep(10)
    print("\nSimulation done. Postprocessing")

    sim_res = list(outit.glob('avg*.f*'))[0]
    J = misfit(sim_res, obsfile, obsnu=1./800., bounds=[.5, 50., -4., 4.], verbose=False)

    # cleanup
    run_command('rm -rf obj/ build.log *.msh run.sh makenek.log makefile done.flag', outit)

    print("(x, J(x))=", x, J)
    return J

opt.set_objective(objective)

print(datetime.datetime.now().isoformat(timespec='seconds', sep=' '))
print("Running optimization...")
opt.run()
x_opt, J_opt = opt.expected_minimum()
print("Expected Minimum:", x_opt, J_opt)

