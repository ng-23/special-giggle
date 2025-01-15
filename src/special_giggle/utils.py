'''
Contains various utilty code used in other modules/notebooks.
'''

import pandas as pd

def load_data(filepath:str) -> pd.DataFrame:
    '''
    Load a timeseries dataset from disk
    '''

    pass

def calc_metrics(preds, targets) -> dict:
    '''
    Calculate performance metrics from model predictions and known truths
    '''

    pass

def save_model(model, filepath=''):
    '''
    Save a trained model to disk
    '''

    pass

def load_model(filepath:str):
    '''
    Load a trained model from disk
    '''

    pass