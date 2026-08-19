import datetime, time

from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize, dump, expected_minimum
from skopt.callbacks import CheckpointSaver

from nekPy.utils.io import write_json
from nekPy.utils.bash import mkdir
from nekPy.optimization.callbacks import ExpectedMinimumStopper



class Optimization():

    def __init__(self, outdir, algorithm='bayesian', bounds=None, x0=None, y0=None, **kwargs):

        self.outdir = Path(outdir)
        mkdir(self.outdir)

        self.algorithm = algorithm.lower()

        self.objective = None

        self.bounds = bounds
        self.x0 = x0
        self.y0 = y0

        self.iters = 0
        self.all_vecs = []
        self.all_objs = []

        self.best_vec = None
        self.best_obj = None
        self.best_it = None

        self.result = None

        # Algorithm-specific options
        self.options = kwargs

        self.total_time = 0.0
        self.iter_times = []
        self.start_time = None

    def __str__(self):
        return (
            f"Optimization:\n"
            f"  outdir      = {self.outdir}\n"
            f"  algorithm   = {self.algorithm}\n"
            f"  objective   = {self.objective}\n"
            f"\n"
            f"  bounds      = {self.bounds}\n"
            f"  x0          = {self.x0}\n"
            f"  y0          = {self.y0}\n"
            f"\n"
            f"  iters       = {self.iters}\n"
            f"  all_vecs    = {self.all_vecs}\n"
            f"  all_objs    = {self.all_objs}\n"
            f"\n"
            f"  best_vec    = {self.best_vec}\n"
            f"  best_obj    = {self.best_obj}\n"
            f"  best_it     = {self.best_it}\n"
            f"\n"
            f"  result      = {self.result}\n"
            f"  options     = {self.options}"
            f"\n"
            f"  start_time  = {self.start_time}\n"
            f"  total_time  = {self.total_time:.3f} h\n"
            f"  iter_times  = {self.iter_times}\n"
        )

    def set_objective(self, objective):
        self.objective = objective

    def update(self, x, J):
        x = np.asarray(x, dtype=float)
        J = float(J)
        self.all_vecs.append(x)
        self.all_objs.append(J)
        idx = int(np.argmin(self.all_objs))
        self.best_it = idx
        self.best_vec = self.all_vecs[idx]
        self.best_obj = self.all_objs[idx]
        self.iters += 1
        self.save()

    def save(self):

        info = {"algorithm": self.algorithm,
                "Nfeval": self.iters,
                "X": [x.tolist() for x in self.all_vecs],
                "J": self.all_objs,
                "opt": {"it": self.best_it,
                        "X": self.best_vec.tolist() if self.best_vec is not None else None,
                        "J": self.best_obj
                        },
            "options": self.options,
            "time": {"start": self.start_time,
                     "total_hours": self.total_time,
                     "iteration_hours": self.iter_times,
                    },
        }
        write_json(info,self.outdir / "info.json")

    def _objective(self, x):
        if self.objective is None: raise ValueError("No objective function has been set.")
        start = time.perf_counter()
        J = self.objective(x, self)

        elapsed = (time.perf_counter() - start) / 3600.0
        self.iter_times.append(elapsed)
        self.total_time += elapsed

        self.update(x, J)
        return float(J)

    def run(self):
        if self.objective is None: raise ValueError("No objective function has been set.")
        self.start_time = datetime.datetime.now().isoformat(timespec="seconds",sep=" ")
        self.save()
        if self.algorithm == "bayesian": self.result = self._run_bayesian()
        elif self.algorithm == "scipy": self.result = self._run_scipy()
        else: raise ValueError(f"Unknown optimization algorithm: {self.algorithm}")
        return self.result

    def _run_bayesian(self):

        if self.bounds is None: raise ValueError("Bayesian optimization requires bounds.")
        dim = len(self.bounds)

        neval = self.options.get("neval", 25 * dim)
        nstarts = self.options.get("nstarts", 5 * dim)
        tol = self.options.get("tol", 1e-2)
        patience = self.options.get("patience", 5)
        disp = self.options.get("disp", True)
        acq_func = self.options.get("acq_func", "EI")
        initial_point_generator = self.options.get("initial_point_generator", "lhs")

        callbacks = [CheckpointSaver(self.outdir / "checkpoint.pkl", store_objective=False),
                     ExpectedMinimumStopper(rel_tol=tol, save=self.outdir, patience=patience)]

        result = gp_minimize(
            self._objective,
            self.bounds,
            x0=self.x0,
            y0=self.y0,
            n_calls=neval,
            n_initial_points=nstarts,
            initial_point_generator=initial_point_generator,
            acq_func=acq_func,
            verbose=disp,
            callback=callbacks)

        dump(result, str(self.outdir / "optres.pkl"), store_objective=False)
        return result

    def _run_scipy(self):
        if self.x0 is None: raise ValueError("Scipy optimization requires x0.")
        method = self.options.get("method", "SLSQP")
        tol = self.options.get("tol", None)
        scipy_options = self.options.get("options", {})
        result = minimize(
            self._objective,
            self.x0,
            method=method,
            bounds=self.bounds,
            tol=tol,
            options=scipy_options)

        return result

    def expected_minimum(self):
        if self.result is None: raise ValueError("Optimization has not been done.")
        if self.algorithm != "bayesian": raise ValueError("expected_minimum is only for Bayesian optimization.")
        return expected_minimum(self.result)