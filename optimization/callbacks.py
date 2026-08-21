import os
import numpy as np
from skopt.callbacks import *
from skopt.utils import expected_minimum
from nekPy.utils.io import write_json

class ExpectedMinimumStopper(EarlyStopper):

    def __init__(self, rel_tol=1e-2, patience=5, expected_optima=None, save=False, random_state=0):
        super(EarlyStopper, self).__init__()
        self.rel_tol = rel_tol
        self.patience = patience
        self.save = save
        self.random_state = random_state
        self.expected_optima = ([] if expected_optima is None else list(expected_optima))

    def _criterion(self, res):
        
        convergence = None
        
        if len(res.models) >= 1:
            expected_min = expected_minimum(res, n_random_starts=100, random_state=self.random_state)
            self.expected_optima.append(expected_min)
            print('Current expected optimum: ', expected_min)
        
        if len(self.expected_optima) >= self.patience + 1:
            
            print("\n Starting convergence analysis:")
            print("----------------------------------")
            
            ye = np.array([e[1] for e in self.expected_optima])
            xe = np.array([e[0] for e in self.expected_optima])
            abs_diff = np.abs(np.diff(ye))
            rel_diff = abs_diff / np.maximum(np.abs(ye[:-1]), np.finfo(float).eps)
            
            print("Expected minima:")
            print(ye)
            print("Relative differences:")
            print(rel_diff)
            
            convergence = (rel_diff[-self.patience:] <= self.rel_tol).all()
            if convergence:
                print(f"Expected minimum converged for "
                      f"{self.patience} consecutive changes "
                      f"(relative tolerance: {self.rel_tol:.2e}).\n")
            else:
                print(f"Expected minimum not yet converged " 
                      f"(relative tolerance: {self.rel_tol:.2e}).\n")
            
            if self.save:
                conv_res = {
                'rel_tol': self.rel_tol,
                'patience': self.patience,
                'expected_y': np.array(ye),
                'expected_x': np.array(xe),
                'abs_diff': np.array(abs_diff),
                'rel_diff': np.array(rel_diff),
                }
                write_json(conv_res, os.path.join(self.save, 'convergence.json'))

        return convergence

