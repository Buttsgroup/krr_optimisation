import sys
import os
sys.path.append(os.path.realpath(os.path.dirname(__file__))+'/../')
from hyperparameter_tuning import HPS_gaussian as hpsgauss

from model.fchl_model import FCHLmodel
from model.features.fchl_input import add_fchl_to_atom_df

import pandas as pd

def test_gaussian():
    
    model = FCHLmodel()
    
    param_ranges = {'cutoff': {'min': 1, 
                                'max': 5,
                                'log': False},
                    'sigma': {'min': -3, 
                                'max': 1,
                                'log': True},
                    'lamda': {'min': -6, 
                                'max': -4,
                                'log': True}}
                                
    tr_atom = pd.read_pickle('tests/test_data/atoms.pkl')
    tr_pair = pd.read_pickle('tests/test_data/pairs.pkl')
    
    rep_func = add_fchl_to_atom_df
    
    cv_range = 3
    
    opt, util = hpsgauss.setup_gaussian(param_ranges, kappa=0.2, xi=0.0)
    
    for _ in range(5):
        score, params, opt = hpsgauss.gaussian_iteration(opt, util, model, tr_atom, tr_pair, rep_func,
                                                param_ranges, cv_range)                
                                                