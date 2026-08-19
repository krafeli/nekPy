import sys, subprocess, functools, json, shutil
import numpy as np
from pathlib import Path

def run_command_(cmd, dir, verbose=True, input=None):
    try:
        with subprocess.run(cmd, cwd=dir, shell=True, input=input, check=True, text=True,
                            stdout=sys.stdout if verbose else subprocess.DEVNULL,
                            stderr=sys.stderr if verbose else subprocess.DEVNULL) as process:
            for line in process.stdout:
                if verbose:
                    print(line, end='', flush=True)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args)
    except:
        return

def run_command(cmd, dir, verbose=True, input=None):

    process = subprocess.Popen(
        cmd,
        cwd=dir,
        shell=True,
        text=True,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output = []

    if input is not None:
        process.stdin.write(input)
        process.stdin.close()

    for line in process.stdout:
        output.append(line)
        if verbose:
            print(line, end="", flush=True)

    process.wait()

    return "".join(output)

def copy(src, dst, overwrite=True):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if src.is_file():
        if dst.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {dst}")
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    elif src.is_dir():
        if dst.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {dst}")
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        shutil.copytree(src, dst)
    else:
        raise ValueError(f"Source is neither a file nor a directory: {src}")
    return dst

def mkdir(path, overwrite=False):
    path = Path(path)

    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Path exists and is not a directory: {path}")
        if overwrite:
            shutil.rmtree(path)
        else:
            return path

    path.mkdir(parents=True, exist_ok=True)
    return path