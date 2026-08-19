from pathlib import Path
from nekPy.utils.nektools import cleandir, cleannek, makenek
from nekPy.utils.bash import run_command, copy
from nekPy.utils.io import write_json

class Launcher():

    def __init__(self, dir):

        self.dir = Path(dir)
        self.files = [f for f in self.dir.iterdir() if f.is_file()]

        usr_files = [f for f in self.files if f.suffix == ".usr"]
        par_files = [f for f in self.files if f.suffix == ".par"]
        size_files = [f for f in self.files if f.name == "SIZE"]
        re2_files = [f for f in self.files if f.suffix == ".re2"]
        ma2_files = [f for f in self.files if f.suffix == ".ma2"]

        if len(usr_files) != 1: raise ValueError("Invalid number of .usr files.", usr_files)
        if len(par_files) != 1: raise ValueError("Invalid number of .par files.", par_files)
        if len(size_files) != 1: raise ValueError("Invalid number of SIZE files.", size_files)
        if len(re2_files) != 1: raise ValueError("Invalid number of .re2 files.", re2_files)
        if len(ma2_files) != 1: raise ValueError("Invalid number of .ma2 files.", ma2_files)

        self.usr_file = usr_files[0]
        self.par_file = par_files[0]
        self.size_file = size_files[0]
        self.re2_file = re2_files[0]
        self.ma2_file = ma2_files[0]
        self.run_file = None

        self.name = self.usr_file.stem

        self.slurm = None
        self.time = None
        self.nodes = None
        self.cores = None
        self.partition = None
        self.account = None
        self.job_name = None
        self.jobid = None

    def clean(self, cleanall=False):
        if cleanall:
            cleandir(dir=self.dir)
        else:
            cleannek(dir=self.dir)

    def makenek(self, cleanall=False):
        self.clean(cleanall=cleanall)
        makenek(dir=self.dir, name=self.name)

    def session_name(self):
        run_command(f'echo "{self.name}" > SESSION.NAME', dir=self.dir)
        run_command(f'echo "{self.dir}" >> SESSION.NAME', dir=self.dir)

    def save_config(self):

        config = {
            "name": self.name,
            "dir": str(self.dir),
            "usr_file": str(self.usr_file),
            "par_file": str(self.par_file),
            "size_file": str(self.size_file),
            "re2_file": str(self.re2_file),
            "ma2_file": str(self.ma2_file),
            "run_file": str(self.run_file) if self.run_file is not None else None,
            "time": self.time,
            "nodes": self.nodes,
            "cores": self.cores,
            "partition": self.partition,
            "account": self.account,
            "job_name": self.job_name,
            "jobid": self.jobid,
        }

        write_json(config, str(self.dir / "launcher.json"))

    def run(self, ncores, make=True):
        self.cores = ncores
        if make:
            self.makenek(cleanall=False)
        self.session_name()
        self.save_config()
        run_command([f"mpirun -np {ncores} ./nek5000 > log.txt" ], dir=self.dir)

    def submit(self, make=True, slurm_script=None, time=None, nodes=None, partition=None, account=None, job_name=None):

        self.time = time
        self.nodes = nodes
        self.partition = partition
        self.account = account
        self.job_name = job_name

        if make:
            self.makenek(cleanall=False)

        if not slurm_script:
            run_files = [f for f in self.files if f.suffix == ".slurm"]
            if len(run_files) != 1: raise ValueError("Invalid number of .slurm files.", run_files)
            self.run_file = run_files[0]
        else:
            dst = copy(slurm_script, self.dir / 'run.slurm')
            self.run_file = Path(dst)

        self.session_name()

        cmd = "sbatch"
        if time is not None: cmd += f" --time {time}"
        if nodes is not None: cmd += f" --nodes {nodes}"
        if partition is not None: cmd += f" --partition {partition}"
        if account is not None: cmd += f" --account {account}"
        if job_name is not None: cmd += f" --job-name {job_name}"

        cmd += f" {self.run_file.name}"
        result = run_command(cmd, dir=self.dir, verbose=False)

        if result is None:
            raise RuntimeError("Failed to submit Slurm job")

        self.jobid = int(result.stdout.split()[-1])

        self.save_config()

        return self.jobid