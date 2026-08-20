from pathlib import Path
from nekPy.launcher.launcher import Launcher
from nekPy.preprocessor.preprocessor import PreProcessor

config = Path("/scratch/projects/hbi00065/3d/DU95W180/long/base")
bl = config / 'bl.pkl'
slurm = config / 'run.slurm'
loc = 0.05
modes = ['blade', 'blasius']

Reks = [500., 600., 700., 800.]

for mode in modes:
    for Rek in Reks:
        job_name = f'Rek{Rek:.0f}_{loc:.2f}_{mode}'
        print(f'Building and submitting {job_name}')
        out = Path(f"/scratch/projects/hbi00065/3d/DU95W180/long/{mode}/xc{str(loc).replace('.', '')}/Rek{Rek:.0f}")


        preproc = PreProcessor(out, 
                            name='loc3', 
                            usr=config / f'loc3_{mode}.usr', 
                            par=config / f'loc3_{mode}.par',
                            size= config / 'SIZE', 
                            re2=config / 'loc3.re2', 
                            ma2=config / 'loc3.ma2')

        preproc.parameters.set('VELOCITY', 'viscosity', -Rek)

        if mode == 'blasius':
            preproc.generate_bc(bl, mode=mode, loc=loc)
            preproc.parameters.set('GENERAL', 'userParam10', preproc.bc.xloc_shifted_k)
            preproc.parameters.set('GENERAL', 'userParam11', preproc.bc.ukb_shifted_raw)
        elif mode == 'blade':
            preproc.generate_bc(bl, mode=mode, loc=loc, Lin=15., verbose=True)

        launcher = Launcher(out)
        launcher.submit(slurm_script=slurm, job_name=job_name)



