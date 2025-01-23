import argparse
import xgboost as xgb
import pandas as pd
from special_giggle import utils

def get_args_parser():
    '''
    Creates commandline argument parser
    '''
    pass

def train(xgbc:xgb.XGBClassifier, train_ts:pd.DataFrame):
    '''
    Trains an XGBoost Classifier on a timeseries dataset
    '''
    pass

def inference(xgbc:xgb.XGBClassifier, test_ts:pd.DataFrame):
    '''
    Performs inference on an unlabeled test timeseries dataset using a trained XGBoost Classifier
    '''
    pass

def main(args:argparse.Namespace):
    '''
    Entry point if running module as a script via command line

    Trains an XGBoost Classifier on the train timeseries dataset and uses it to perform inference on
    the unlabeled test timeseries dataset

    Saves train metrics (accuracy, precision, f1, recall, etc.) and inference results to disk as CSV
    '''
    pass

if __name__ == '__main__':
    # entry point if running as a script
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)