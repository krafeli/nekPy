from pathlib import Path
from natsort import natsorted

from nekPy.launcher import Launcher
from nekPy.preprocessor import PreProcessor

base = Path("/scratch/projects/hbi00065/3d/DU95W180/lx75")
config = base / 'config'

bl = config / 'bl.pkl'
slurm = config / 'run_clx.slurm'

locs = [0.25]
modes = ['blasius', 'blade']
Reks = [2400.]

Lx1 = [10, 10, 6, 6, 6]
Nodes = [192, 192, 16, 16, 16]

eta = 1.0

partition = 'cpu-clx'
time = '12:00:00'
account = 'hbi00065'

for mode in modes:
    for i, Rek in enumerate(Reks):
        for loc in locs:
            
            loc_str = f"{loc:.2f}".replace('.', '')
            eta_str = f"eta{eta:.2f}".replace('.','p')
            
            #out = base / 'transition' / 'same_pert' / f"{mode}"/ f"xc{loc_str}"/ f"Rek{Rek:.0f}"
            #out = base / 'sfd' / f"{mode}"/ f"xc{loc_str}"/ f"Rek{Rek:.0f}"
            out = base / f"{mode}"/ f"xc{loc_str}"/ f"Rek{Rek:.0f}"
            job_name = f'Rek{Rek:.0f}_{loc:.2f}_{mode}'
            
            lx1, nodes = Lx1[i], Nodes[i]
            print(f'\nBuilding and submitting {job_name}')
            print(f'Rek={Rek:.0f}')
            print(f'loc={loc:.2f}')
            print(f'eta={eta:.2f}')
            print(f'lx1={lx1}')
            print(f'nodes={nodes}')
            print(f'out={out}\n')

            pert_base = base / f"blasius" / f"xc005" / f'Rek{900.:.0f}'
            pert_mean = natsorted(list(pert_base.glob('avgloc*.f*')))[-1]
            pert_inst = natsorted(list(pert_base.glob('loc*.f*')))[-1]
            
            additional_files = [
                            #[pert_inst, 'pert_field0.f00001'], 
                            #[pert_mean, 'avg_field0.f00001'], 
                            [config / 'probes.xyz', 'probes.xyz']
                            #[config / 'SFDD', 'SFDD']
                                ]
            
            preproc = PreProcessor(out, 
                                name='loc3', 
                                usr=config / f'loc3_{mode}.usr', 
                                par=config / f'loc3_{mode}.par',
                                size= config / 'SIZE', 
                                re2=config / f'loc3{eta_str}.re2', 
                                ma2=config / f'loc3{eta_str}.ma2',
                                additional_files=additional_files)

            preproc.parameters.set('GENERAL', 'startFrom', 0.0)
            preproc.parameters.set('GENERAL', 'writeInterval', 50.0)
            
            preproc.parameters.set('GENERAL', 'userParam02', 250.0)
            preproc.parameters.set('GENERAL', 'userParam06', 500.0)

            preproc.parameters.set('VELOCITY', 'viscosity', -Rek)   # set Rek 
            
            """
            # Perturbation Parameters
            ftt = 75. 
            injection_time = 5. * ftt
            start_stats    = injection_time + 5. * ftt
            min_stats = start_stats + 5. * ftt
            preproc.parameters.set('GENERAL', 'userParam02', start_stats)       # start to compute stats
            preproc.parameters.set('GENERAL', 'userParam06', min_stats)         # minimal runtime
            preproc.parameters.set('GENERAL', 'userParam15', injection_time)    # perturbation injection time
            preproc.parameters.set('GENERAL', 'userParam16', 1e-2)              # perturbation amplitdue
            """
            
            """
            # SFD parameters
            preproc.parameters.set('GENERAL', 'userParam01', 0)       # disable averaging
            preproc.parameters.set('GENERAL', 'userParam04', 1e-7)    # SFD tolerance
            """
            
            # SIZE parameters
            preproc.size.set('lx1', lx1)

            if mode == 'blasius':
                preproc.generate_bc(bl, mode=mode, loc=loc)
                print("Writing Blasius params")
                print(preproc.bc.xloc_shifted_k, preproc.bc.ukb_shifted_raw)
                preproc.parameters.set('GENERAL', 'userParam10', preproc.bc.xloc_shifted_k)
                preproc.parameters.set('GENERAL', 'userParam11', preproc.bc.ukb_shifted_raw)
            elif mode == 'blade':
                preproc.generate_bc(bl, mode=mode, loc=loc, Lin=15., verbose=True)
                preproc.generate_ic(bl, loc=loc, Lin=15., Lout=75.)

            launcher = Launcher(out)
            launcher.submit(slurm_script=slurm, job_name=job_name, nodes=nodes, partition=partition, time=time, account=account)