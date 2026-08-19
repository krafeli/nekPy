import numpy as np
from skopt.callbacks import *
from skopt.utils import expected_minimum
from nekPy.utils.io import write_json

class ExpectedMinimumStopper(EarlyStopper):

    def __init__(self, rel_tol=1e-3, patience=5, expected_optima=[], save=False):
        super(EarlyStopper, self).__init__()
        self.rel_tol = rel_tol
        self.patience = patience
        self.save = save
        self.expected_optima = expected_optima

    def _criterion(self, res):
        
        convergence = None
        
        if len(res.models) >= 1:
            expected_min = expected_minimum(res, n_random_starts=100)
            self.expected_optima.append(expected_min)
            print('Current expected optimum: ', expected_min)
        
        if len(self.expected_optima) >= self.patience + 1:
            
            print("\n Starting convergence analysis:")
            print("----------------------------------")
            
            ye = np.array([e[1] for e in self.expected_optima])
            xe = np.array([e[0] for e in self.expected_optima])
            dye = np.abs(np.diff(ye)) / np.abs(ye[:-1])
            print(ye)
            print(dye)
            if (dye[-self.patience:] <= self.rel_tol).all():
                print('relative difference in expected minima <= ', str(self.rel_tol), '\n')
                convergence = True
            else:
                print('relative difference in expected minima > ', str(self.rel_tol), '\n')
                convergence = False
            
            if self.save:
                conv_res = {
                'rel_tol': self.rel_tol,
                'patience': self.patience,
                'expected_y': np.array(ye),
                'expected_x': np.array(xe),
                'rel_diff': np.array(dye),
                }
                write_json(conv_res, os.path.join(self.save, 'convergence.json'))

        return convergence

