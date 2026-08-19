import sys
from pathlib import Path

def logger(outdir, logfile='logger.txt', errlogfile='errlog.txt'):
    outdir = Path(outdir)
    sys.stdout = open((outdir / logfile), 'w')
    sys.stderr = open((outdir / errlogfile), 'w')
    return