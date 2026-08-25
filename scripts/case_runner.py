from pathlib import Path
from nekPy.launcher import Launcher
from nekPy.preprocessor import PreProcessor

config = Path("/scratch/projects/hbi00065/3d/DU95W180/lx150/base")
bl = config / 'bl.pkl'
slurm = config / 'run.slurm'
loc = 0.25
modes = ['blasius', 'blade']
Reks = [700.]

for mode in modes:
    for Rek in Reks:
        job_name = f'Rek{Rek:.0f}_{loc:.2f}_{mode}'
        print(f'Building and submitting {job_name}')
        
        out = Path(f"/scratch/projects/hbi00065/3d/DU95W180/lx150/{mode}/xc{str(loc).replace('.', '')}/Rek{Rek:.0f}")

        #init = Path(f"/scratch/projects/hbi00065/3d/DU95W180/{mode}/xc{str(loc).replace('.', '')}/Rek{Rek:.0f}/loc30.f00005")
        
        preproc = PreProcessor(out, 
                            name='loc3', 
                            usr=config / f'loc3_{mode}.usr', 
                            par=config / f'loc3_{mode}.par',
                            size= config / 'SIZE', 
                            re2=config / 'loc3.re2', 
                            ma2=config / 'loc3.ma2',
                            additional_files=None)

        #preproc.parameters.set('GENERAL', 'startFrom', 'loc30.f00005 int time=0.0')
        preproc.parameters.set('VELOCITY', 'viscosity', -Rek)
        preproc.parameters.set('GENERAL', 'userParam02', 1000.)
        preproc.parameters.set('GENERAL', 'userParam06', 1250.)

        if mode == 'blasius':
            preproc.generate_bc(bl, mode=mode, loc=loc)
            preproc.parameters.set('GENERAL', 'userParam10', preproc.bc.xloc_shifted_k)
            preproc.parameters.set('GENERAL', 'userParam11', preproc.bc.ukb_shifted_raw)
        elif mode == 'blade':
            preproc.generate_bc(bl, mode=mode, loc=loc, Lin=15., verbose=True)

        launcher = Launcher(out)
        launcher.submit(slurm_script=slurm, job_name=job_name)



