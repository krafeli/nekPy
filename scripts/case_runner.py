from pathlib import Path
from nekPy.launcher import Launcher
from nekPy.preprocessor import PreProcessor

base = Path("/scratch/projects/hbi00065/3d/DU95W180/lx75")
config = base / 'config'

bl = config / 'bl.pkl'
slurm = config / 'run.slurm'
locs = [0.25]
modes = ['blasius']
Reks = [800.]

nodes = 4
partition = 'cpu-clx:test'

for mode in modes:
    for Rek in Reks:
        for loc in locs:
            job_name = f'Rek{Rek:.0f}_{loc:.2f}_{mode}'
            print(f'Building and submitting {job_name}')
            
            loc_str = f"{loc:.2f}".replace('.', '')
            
            out = base / 'test_pert' / f"{mode}"/ f"xc{loc_str}"/ f"Rek{Rek:.0f}"
            
            #init = base / 'continue' / f"{mode}" / f"xc{loc_str}" / f'Rek{Rek:.0f}' / 'loc30.f00025'
            
            pert = base / f"{mode}" / f"xc{loc_str}" / f'Rek{900.:.0f}' / 'loc30.f00015'
            
            preproc = PreProcessor(out, 
                                name='loc3', 
                                usr=config / f'loc3_{mode}_pertu.usr', 
                                par=config / f'loc3_{mode}.par',
                                size= config / 'SIZE', 
                                re2=config / 'loc3.re2', 
                                ma2=config / 'loc3.ma2',
                                additional_files=[[pert, 'pert_field.f00001'], 
                                                  [config / 'probes.xyz', 'probes.xyz']])

            #preproc.parameters.set('GENERAL', 'startFrom', 'loc30.f00007 int UP time=0.0')
            preproc.parameters.set('GENERAL', 'endTime', 5000.)
            preproc.parameters.set('GENERAL', 'writeInterval', 1.)

            preproc.parameters.set('VELOCITY', 'viscosity', -Rek)
            preproc.parameters.set('GENERAL', 'userParam02', 3000.) # start to compute stats
            preproc.parameters.set('GENERAL', 'userParam06', 3500.) # minimal runtime
            
            preproc.parameters.set('GENERAL', 'userParam15', 10.)  # perturbation injection time
            preproc.parameters.set('GENERAL', 'userParam16', 1e-4) # perturbation amplitdue
            
            preproc.size.set('lx1', 6)

            if mode == 'blasius':
                preproc.generate_bc(bl, mode=mode, loc=loc)
                print("Writing Blasius params")
                print(preproc.bc.xloc_shifted_k, preproc.bc.ukb_shifted_raw)
                preproc.parameters.set('GENERAL', 'userParam10', preproc.bc.xloc_shifted_k)
                preproc.parameters.set('GENERAL', 'userParam11', preproc.bc.ukb_shifted_raw)
            elif mode == 'blade':
                preproc.generate_bc(bl, mode=mode, loc=loc, Lin=15., verbose=True)

            launcher = Launcher(out)
            launcher.submit(slurm_script=slurm, job_name=job_name, nodes=nodes, partition=partition)



