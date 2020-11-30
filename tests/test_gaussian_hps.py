import sys
import os
sys.path.append(os.path.realpath(os.path.dirname(__file__))+'/../')
from hyperparameter_tuning import HPS_gaussian as hpsgauss

from model.fchl_model import FCHLmodel
from model.features.fchl_input import add_fchl_to_atom_df

import pandas as pd

def test_gaussian():
    
    model = FCHLmodel()
    
    param_ranges = {'cutoff': {'min': 3, 
                                'max': 10,
                                'log': False},
                    'sigma': {'min': -2, 
                                'max': 5,
                                'log': True},
                    'lamda': {'min': -9, 
                                'max': -2,
                                'log': True}}
                                
    tr_atom = pd.read_pickle('tests/test_data/atoms.pkl')
    tr_pair = pd.read_pickle('tests/test_data/pairs.pkl')
    
    rep_func = add_fchl_to_atom_df
    
    cv_range = 5
    
    opt, util = hpsgauss.setup_gaussian(param_ranges, kappa=5, xi=0.1)
    
    for _ in range(10):
        score, params, opt = hpsgauss.gaussian_iteration(opt, util, model, tr_atom, tr_pair, rep_func,
                                                param_ranges, cv_range)                
        print(score, params)
        
    #assert False                