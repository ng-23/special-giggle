'''
Python implementation of the Alpha approach with Optuna.

Can be run as a script via command line or imported into other modules.
'''

import os
import json
import argparse
import pandas as pd
import optuna as op
import xgboost as xgb
from sklearn.utils import compute_class_weight, compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from special_giggle.core.schemas import validate_config
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

class AlphaObjective():
    '''
    Objective function to optimize when using Optuna for automated hyperparameter optimization for the Alpha approach

    Alpha relies solely on the timeseries precipitation data, assumed to have been feature engineered to include more relevant info.

    Per the competition, log-loss is the optimization metric and as such should be minimized

    See https://colab.research.google.com/drive/1wCQw3NTad2S4kXxxAqN0WjORhiqyVV2v?usp=sharing
    '''

    def __init__(self, ts:pd.DataFrame, search_space:dict, use_gpu:bool=False, seed:int=42, output_dir:str=''):
        self.ts = ts
        self.ts_les = self._encode_categorical_features()
        self.search_space = self._load_search_space(search_space)
        self.seed = seed
        self.output_dir = output_dir
        self.days = self.ts['event_t'].unique()
        self.tscv = self._init_ts_splitter()
        self.device = 'cuda' if use_gpu else 'cpu'

    def _encode_categorical_features(self):
        '''
        Preprocess the timeseries data so it can seamlessly be passed into XGBoost Classifier

        This mainly entails encoding categorical features
        '''

        feature_cols = [col for col in self.ts.columns if col != 'label'] 

        ts_les = {} # maps a feature (column) name to its label encoder
        for col in feature_cols:
            if self.ts[col].dtype == 'object':
                le = LabelEncoder()
                self.ts[col] = le.fit_transform(self.ts[col])

                ts_les[col] = le

        return ts_les

    def _load_search_space(self, search_space:dict):
        '''
        Validate each hyperparameter's search space
        '''

        new_search_space = {}
        
        for param in search_space:
            param_search_space = validate_config('OptunaSearchSpace', search_space[param])
            new_search_space[param] = param_search_space

        return new_search_space
    
    def _init_ts_splitter(self):
        '''
        Initialize timeseries cross-validation splitter

        Splits are based on the day (0-729) to preserve temporal structure

        Ensures at least 1 flood event sample is present in each split

        Ex:\n
        Split 1: train=[0,1] test=[2]\n
        Split 2: train=[0,1,2] test=[3]\n
        Split 3: train=[0,1,2,3] test=[4]\n
        '''

        # determine day of first flood event
        first_flood_event = self.ts.loc[(self.ts['label'] == 1.0)]['event_t'].min()

        # define number of splits (need to exclude an additional day for the test set)
        n_splits = len(self.days) - first_flood_event - 1

        return TimeSeriesSplit(n_splits=n_splits, test_size=1)

    def _get_hyperparams(self, trial:op.Trial):
        '''
        Sample hyperparameters based on the provided search space
        '''

        hyperparams = {}

        for param in self.search_space:
            val = None
            val_search_space = self.search_space[param]

            type = val_search_space['type']
            if type == 'float':
                val = trial.suggest_float(
                    param, 
                    val_search_space['low'], 
                    val_search_space['high'], 
                    log=val_search_space['log'],
                    step=val_search_space['step'],
                    )
            elif type == 'int':
                val = trial.suggest_int(
                    param, 
                    val_search_space['low'], 
                    val_search_space['high'], 
                    step=val_search_space['step'] if val_search_space['step'] is not None else 1, # special case since step cannot be None
                    log=val_search_space['log'],
                    )

            hyperparams[param] = val

        # see https://xgboosting.com/xgboost-scale_pos_weight-vs-sample_weight-for-imbalanced-classification/
        # see https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
        class_weights = compute_class_weight(class_weight='balanced', classes=self.ts['label'].unique(), y=self.ts['label'])
        if 'pos_cls_weight' not in hyperparams:
            #hyperparams['scale_pos_weight'] = len(self.ts.loc[(self.ts['label'] == 0.0)]) / len(self.ts.loc[(self.ts['label'] == 1.0)])
            hyperparams['pos_cls_weight'] = class_weights[1]
        if 'neg_cls_weight' not in hyperparams:
            hyperparams['neg_cls_weight'] = class_weights[0]

        return hyperparams

    def _train_cv(self, hyperparams:dict):
        '''
        Train XGBoost Classifiers using cross-validation

        See https://stackoverflow.com/questions/63224426/how-can-i-cross-validate-by-pytorch-and-optuna
        '''

        hyperparams['device'] = self.device

        pos_cls_weight, neg_cls_weight = hyperparams.pop('pos_cls_weight'), hyperparams.pop('neg_cls_weight')
        hyperparams['scale_pos_weight'] = pos_cls_weight

        # get column names of input features
        feature_cols = [col for col in self.ts.columns if col not in {'label','event_t','event_id'}]

        val_metrics = {'fold':[], 'log_loss':[], 'brier_score':[], 'roc_auc':[]}

        for i, (train_index, test_index) in enumerate(self.tscv.split(self.days)):
            # see https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
            model = xgb.XGBClassifier(**hyperparams, objective='binary:logistic', seed=self.seed)

            print(f'Fold: {i}')

            # get the training data
            train_days = self.days[train_index]
            train_set = self.ts[self.ts['event_t'].isin(train_days)]
            X_train, y_train = train_set[feature_cols], train_set['label']
            print(f'Train days: {train_days[0]}-{train_days[-1]}')

            # get the validation data 
            val_days = self.days[test_index]
            val_set = self.ts[self.ts['event_t'].isin(val_days)]
            X_val, y_val = val_set[feature_cols], val_set['label']
            print(f'Validation days: {val_days[0]}')

            # fit model on train data
            print('Fitting model...')
            model.fit(X_train, y_train)

            # make probability predictions on validation data
            y_pos_proba = model.predict_proba(X_val)[:,-1]

            sample_weight = compute_sample_weight(class_weight={0.0:neg_cls_weight, 1.0:pos_cls_weight}, y=y_val)

            # calc val metrics and store for later
            val_metrics['fold'].append(i)
            val_metrics['log_loss'].append(log_loss(y_val, y_pos_proba, labels=[0.0,1.0], sample_weight=sample_weight))
            val_metrics['brier_score'].append(brier_score_loss(y_val, y_pos_proba, pos_label=1.0, sample_weight=sample_weight))
            val_metrics['roc_auc'].append(roc_auc_score(y_val, y_pos_proba, average='micro', labels=[0.0,1.0], sample_weight=sample_weight))

            # print val metrics for current fold
            for metric in val_metrics:
                if metric != 'fold':
                    print(f'Validation {metric}: {val_metrics[metric][-1]}')

        # average val metrics across all folds and print
        val_metrics = pd.DataFrame.from_dict(val_metrics)
        for col in val_metrics.columns:
            if col != 'fold':
                val_metrics[f'avg_{col}'] = val_metrics.expanding(min_periods=1)[col].mean()
                print(f'Average validation {col}: {val_metrics.loc[len(val_metrics)-1][f'avg_{col}']}')

        return val_metrics
    
    def refit_model(self, trial:op.Trial):
        '''
        Refits an XGBoost Classifier on all the training data
        '''

        params = trial.params
        params['device'] = self.device

        feature_cols = [col for col in self.ts.columns if col not in {'label','event_t','event_id'}]
        X_train, y_train = self.ts[feature_cols], self.ts['label']

        model = xgb.XGBClassifier(**params, objective='binary:logistic', seed=self.seed)
        model.fit(X_train, y_train)

        return model
    
    def get_metrics(self, trial:op.Trial):
        '''
        Get the performance metrics of the trained XGBoost Classifiers from a completed Optuna trial
        '''

        return trial.user_attrs['val_metrics_cv']
    
    def save_trial(self, trial:op.Trial, output_dir:str=''):
        trial_output_dir = os.path.join(output_dir, f'trial{trial.number}')
        os.makedirs(trial_output_dir, exist_ok=True)

        trial.user_attrs['val_metrics_cv'].to_csv(os.path.join(trial_output_dir, 'val_metrics_cv.csv'), index=False)
        with open(os.path.join(trial_output_dir, 'hyperparams.json'), 'w') as f:
            json.dump(trial.params, f, indent=4)

    def __call__(self, trial:op.Trial):
        # sample some hyperparameters based on the search space
        hyperparams = self._get_hyperparams(trial)

        print(f'Trial: {trial.number}\nParams: {hyperparams}')

        # train an XGBoost Classifier with the hyperparameters
        val_metrics = self._train_cv(hyperparams)

        # store performance metrics in completed trial
        trial.set_user_attr('val_metrics_cv', val_metrics)
        trial.set_user_attr('model', None)

        self.save_trial(trial, output_dir=self.output_dir)

        return val_metrics.loc[len(val_metrics)-1]['avg_log_loss']
    
def get_args_parser():
    '''
    Creates commandline argument parser
    '''

    parser = argparse.ArgumentParser(
        description='Alpha Approach', 
        add_help=True,
        )
    
    parser.add_argument(
        'train_ts_path', 
        metavar='train-ts-path', 
        type=str,
        help='Filepath to train timeseries CSV',
        )
    
    parser.add_argument(
        'test_ts_path',
        metavar='test-ts-path',
        type=str,
        help='Filepath to test timeseries CSV',
        )
    
    parser.add_argument(
        'hyperparam_search_space',
        metavar='hyperparam-search-space',
        type=str,
        help='Filepath to JSON hyperparameter search space config',
        )
    
    parser.add_argument(
        '--trials', 
        type=int, 
        default=10, 
        help='Number of iterations to perform in discovery of optimal hyperparameters',
        )

    parser.add_argument(
        '--use-gpu', 
        action='store_true', 
        help='If specified, use GPU for training/testing',
        )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Value of seed for controlling random state',
        )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None, 
        help='Filepath to directory to save outputs to',
        )
    
    return parser

def inference(xgbc:xgb.XGBClassifier, test_ts:pd.DataFrame) -> pd.DataFrame:
    '''
    Performs inferencing on an unlabeled test timeseries dataset using a trained XGBoost Classifier
    '''
    pass

def main(args:argparse.Namespace):
    '''
    Entry point if running module as a script via command line

    Trains an XGBoost Classifier on the entire train timeseries dataset, with Optuna for automated hyperparameter optimization

    Trained classifier is then used to perform inference on the unlabeled test timeseries dataset

    Saves train metrics (accuracy, precision, f1, recall, etc.) and inference results to disk as CSV, along with other outputs
    '''
    
    # make output dir if specified, otherwise save everything to current working directory
    output_dir = args.output_dir if args.output_dir is not None else os.curdir
    os.makedirs(output_dir, exist_ok=True,)

    # save supplied command line args to disk as JSON
    with open(os.path.join(output_dir, 'cmd_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # load train timeseries data
    train_ts = pd.read_csv(args.train_ts_path)

    # define objective function to optimize
    obj = AlphaObjective(
        train_ts, 
        json.load(open(args.hyperparam_search_space)), 
        use_gpu=args.use_gpu, 
        seed=args.seed, 
        output_dir=output_dir,
        )

    # define the optuna study
    study = op.create_study(
        sampler=op.samplers.TPESampler(seed=args.seed),
        study_name='xgbc-hyperparam-optimization', 
        direction='minimize',
        )
    
    # carry out hyperparameter optimization of objective function using optuna study
    print(f'Performing {args.trials} trials of hyperparameter optimization for XGBoost Classifier...')
    study.optimize(obj, n_trials=args.trials)

    # get the best trial's data
    print(f'Best trial: {study.best_trial.number}')
    metrics = obj.get_metrics(study.best_trial)
    print(f'Best training metrics:\n{metrics}')
    print(f'Best hyperparameters:\n{study.best_params}')
    print('-'*75)

    # refit best model on all training data
    model = obj.refit_model(study.best_trial)

    # load test timeseries data
    #test_ts = pd.read_csv(args.test_ts_path)

    # perform inferencing on test data using best trained model
    #test_preds = inference(model, test_ts)

    # save inference results to disk as CSV
    #test_preds.to_csv(path_or_buf=os.path.join(output_dir, 'test_preds.csv'), index=False)

    # save the best trial's model to disk
    # see https://stackoverflow.com/questions/58149861/dump-xgboost-model-with-feature-map-using-xgbclassifier and https://stackoverflow.com/questions/43691380/how-to-save-load-xgboost-model
    model.save_model(os.path.join(output_dir, f'best_model{study.best_trial.number}.json'))
    model.get_booster().dump_model(os.path.join(output_dir, f'best_model{study.best_trial.number}_dump.txt'))

if __name__ == '__main__':
    # entry point if running as a script
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)