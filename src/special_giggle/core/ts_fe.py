'''
Feature engineering for time series precipitation data.

Can be run as a script via command line or imported into other modules.
'''

import argparse
import json
import os
import pandas as pd
import time
from special_giggle import utils
from sklearn.preprocessing import LabelEncoder
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import from_tsfresh_dataset

# runtime config, maps a feature engineering function name to a dict of its arguments
DEFAULT_CONFIG = {
    'month': {},
    'season': {},
    'l_precip_1H': {},
    'l_precip_intensity_1d': {},
    'l_cum_sum_precip_1d': {},
    'l_cum_avg_precip_1d': {},
    'l_cum_avg_precip_1H': {},
    'l_lag_cum_sum_precip_1d': {'days':3},
    'l_lag_cum_avg_precip_1d': {'days':3},
    'l_lag_cum_avg_precip_1H': {'days':3},
    'r_cum_sum_precip_1d': {},
    'g_cum_sum_precip_1d': {},
}

registered_fe_funcs = {} # maps a function name to a feature engineering function

def register_fe_func(func_name:str):
    '''
    Registers a function as a feature engineering function

    See https://www.programiz.com/python-programming/decorator
    '''
    def decorator(func):
        registered_fe_funcs[func_name] = func
        return func
    return decorator

def get_args_parser():
    '''
    Creates commandline argument parser
    '''

    parser = argparse.ArgumentParser(
        description='time series Feature Engineering', 
        add_help=True,
        )

    parser.add_argument(
        'train_ts_path', 
        metavar='train-ts-path', 
        type=str,
        help='Filepath to train time series CSV',
        )
    
    parser.add_argument(
        'test_ts_path',
        metavar='test-ts-path',
        type=str,
        help='Filepath to test time series CSV',
        )
    
    parser.add_argument(
        '--user-config', 
        type=str, 
        default=None,
        help='Filepath to JSON runtime configuration file. Overrides default configuration if specified',
        )
    
    parser.add_argument(
        '--cluster-locations', 
        type=int, 
        default=0, 
        help='If > 1, perform K-Means clustering to group similar locations into regions',
        )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Value of seed for controlling random state',
        )
    
    parser.add_argument(
        '--get-default-config', 
        action='store_true', 
        help='If specified, output the default configuration as JSON',
        )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None, 
        help='Filepath to directory to save outputs to',
        )

    return parser

def init_ts(filepath:str):
    '''
    Loads time series from CSV and performs base preprocessing
    '''
    
    ts = pd.read_csv(filepath_or_buffer=filepath)

    # each event is of the format id_location_X_timestamp
    # we wish to separate the event identifier (aka the location) from the timestamp
    ts['event_id'] = sorted(['_'.join(event_str.split('_')[0:2]) for event_str in ts['event_id']])
    ts['event_id'] = ts['event_id'].astype('category')

    # at this point the event identifier (location) is separate from the timestamp
    # but in the process we've lost the timestamp, so we need to get it back
    # each location has data for 730 days, so we need 730 timestamps per location
    ts['event_t'] = ts.groupby(['event_id'], sort=False, observed=True).cumcount()

    return ts

@register_fe_func('month')
def month(ts:pd.DataFrame, year=2025):
    '''
    Derives the month from the day-of-the-year timestamp

    Certain months may correspond to a wet/dry season for a particular location,
    which can help explain increased/decreased precipitation that could relate to flood events

    Feature type: categorical

    Features created: 1
    '''

    if 'event_t' not in ts: raise Exception('Missing required column - event_t')
    
    ts['month'] = [utils.get_month_from_year_day(year, day) for day in ts['event_t']]
    ts['month'] = ts['month'].astype('category')

    return ts

@register_fe_func('season')
def season(ts:pd.DataFrame):
    '''
    Derives the season from the month, according to https://southafrica-info.com/land/south-africa-weather-climate/

    Feature type: categorical

    Features created: 1
    '''

    if 'month' not in ts: raise Exception('Missing required column - month')
    
    ts['season'] = [utils.get_season_from_month(month) for month in ts['month']]
    ts['season'] = ts['season'].astype('category')

    return ts

@register_fe_func('l_precip_1H')
def local_hourly_precip(ts:pd.DataFrame):
    '''
    Derives the hourly precipitation for each day from the total daily (24 hours) precipitation

    Feature type: numerical

    Features created: 1
    '''

    if 'precipitation' not in ts: raise Exception('Missing required column - precipitation')
    
    ts['l_precip_1H'] = [daily_precip / 24.0 for daily_precip in ts['precipitation']]

    return ts

@register_fe_func('l_precip_intensity_1d')
def local_daily_precip_intensity(ts:pd.DataFrame):
    '''
    Tracks daily precipitation intensity based on the hourly precipitation for each day

    American Meteorological Society (AMS) defines intensities for precipitation periods (https://glossary.ametsoc.org/wiki/Rain)

    Feature type: categorical

    Features created: 1
    '''

    if 'l_precip_1H' not in ts: raise Exception('Missing required column - l_precip_1H')

    ts['l_precip_intensity_1d'] = [utils.get_precip_intensity(hourly_precip) for hourly_precip in ts['l_precip_1H']]
    ts['l_precip_intensity_1d'] = ts['l_precip_intensity_1d'].astype('category')
    
    return ts

@register_fe_func('l_cum_sum_precip_1d')
def local_cum_sum_daily_precip(ts:pd.DataFrame):
    '''
    Tracks the cumulative total of daily precipitation for each month, grouped by event_id and separate for y1/y2

    Feature type: numerical

    Features created: 1
    '''

    y1_data = ts.loc[(ts['event_t'] <= 364)].copy() # see https://stackoverflow.com/a/66362915
    y2_data = ts.loc[(ts['event_t'] > 364)].copy()

    y1_data['l_cum_sum_precip_1d'] = y1_data.groupby(['event_id','month'], sort=False, observed=True)['precipitation'].expanding(min_periods=1).sum().reset_index(level=[0,1], drop=True)
    y2_data['l_cum_sum_precip_1d'] = y2_data.groupby(['event_id','month'], sort=False, observed=True)['precipitation'].expanding(min_periods=1).sum().reset_index(level=[0,1], drop=True)

    return pd.concat([y1_data, y2_data], axis=0)

@register_fe_func('l_cum_avg_precip_1d')
def local_cum_avg_daily_precip(ts:pd.DataFrame):
    '''
    Tracks the cumulative average of daily precipitation for each month, grouped by event_id and separate for y1/y2

    Feature type: numerical

    Features created: 1
    '''
    
    y1_data = ts.loc[(ts['event_t'] <= 364)].copy()
    y2_data = ts.loc[(ts['event_t'] > 364)].copy()

    # see https://stackoverflow.com/a/56911728/ and https://stackoverflow.com/a/58851846/
    y1_data['l_cum_avg_precip_1d'] = y1_data.groupby(['event_id','month'], sort=False, observed=True)['precipitation'].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)
    y2_data['l_cum_avg_precip_1d'] = y2_data.groupby(['event_id','month'], sort=False, observed=True)['precipitation'].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)

    return pd.concat([y1_data, y2_data], axis=0)

@register_fe_func('l_cum_avg_precip_1H')
def local_cum_avg_hourly_precip(ts:pd.DataFrame):
    '''
    Tracks the cumulative average of hourly precipitation for each month, grouped by event_id and separate for y1/y2

    Feature type: numerical

    Features created: 1
    '''

    y1_data = ts.loc[(ts['event_t'] <= 364)].copy()
    y2_data = ts.loc[(ts['event_t'] > 364)].copy()

    y1_data['l_cum_avg_precip_1H'] = y1_data.groupby(['event_id','month'], sort=False, observed=True)['l_precip_1H'].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)
    y2_data['l_cum_avg_precip_1H'] = y2_data.groupby(['event_id','month'], sort=False, observed=True)['l_precip_1H'].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)

    return pd.concat([y1_data, y2_data], axis=0)

@register_fe_func('l_lag_cum_sum_precip_1d')
def local_lag_cum_sum_daily_precip(ts:pd.DataFrame, days:int):
    '''
    Tracks the rolling total of daily precipitation for a given number of past days, grouped by event_id

    Feature type: numerical

    Features created: 1
    '''

    if 'precipitation' not in ts: raise Exception('Missing required column - precipitation')

    col_name = f'l_{days}d_cum_sum_precip_1d'

    lag_data = ts.groupby(['event_id'], sort=False, observed=True)['precipitation'].rolling(window=days, min_periods=1).sum().reset_index(level=[0], drop=True)

    ts[col_name] = lag_data

    return ts

@register_fe_func('l_lag_cum_avg_precip_1d')
def local_lag_cum_avg_daily_precip(ts:pd.DataFrame, days:int):
    '''
    Tracks the rolling average of daily precipitation for a given number of past days, grouped by event_id

    Feature type: numerical

    Features created: 1
    '''
    
    if 'precipitation' not in ts: raise Exception('Missing required column - precipitation')

    col_name = f'l_{days}d_cum_avg_precip_1d'

    lag_data = ts.groupby(['event_id'], sort=False, observed=True)['precipitation'].rolling(window=days, min_periods=1).mean().reset_index(level=[0], drop=True)

    ts[col_name] = lag_data

    return ts

@register_fe_func('l_lag_cum_avg_precip_1H')
def local_lag_cum_avg_hourly_precip(ts:pd.DataFrame, days:int):
    '''
    Tracks the rolling average of hourly precipitation for a given number of past days, grouped by event_id

    Feature type: numerical

    Features created: 1
    '''

    if 'l_precip_1H' not in ts: raise Exception('Missing required column - l_precip_1H')

    col_name = f'l_{days}d_cum_avg_precip_1H'

    lag_data = ts.groupby(['event_id'], sort=False, observed=True)['l_precip_1H'].rolling(window=days, min_periods=1).mean().reset_index(level=[0], drop=True)

    ts[col_name] = lag_data

    return ts

def cluster_locations(train_ts:pd.DataFrame, test_ts:pd.DataFrame, n_regions:int=8, seed:int=42):
    '''
    Performs K-Means clustering on locations using Dynamic Barycenter Averaging (DBA) to 
    group similar time series together based on their Dynamic Time Warping (DTW) similarity/distance

    Single model is fitted on train time series then applied to test time series, grouping locations into "regions"

    Features created: 1

    Feature type: numerical
    '''

    ts_kmeans = TimeSeriesKMeans(n_clusters=n_regions, random_state=seed)

    # tslearn doesn't work directly with dataframes but has a utility to convert them to the necessary data structure
    # first we format the dataframes according the tsfresh "flat" style (see https://tsfresh.readthedocs.io/en/latest/text/data_formats.html#input-option-1-flat-dataframe)
    train_ts, test_ts = train_ts.rename(columns={'event_t':'time', 'event_id':'id'}), test_ts.rename(columns={'event_t':'time', 'event_id':'id'})
    
    # we assume train/test have same feature columns
    ignore_cols = [col for col in train_ts.columns if col == 'label' or col.startswith('r') or col.startswith('g')] # only use local features (not regional/global)
    feature_cols = [col for col in train_ts.columns if col not in ignore_cols] 

    # tslearn can't handle categorical features, so we need to numerically encode them
    train_les = {} # maps a feature (column) name to its label encoder
    test_les = {}
    for col in feature_cols:
        if train_ts[col].dtype == 'category':
            train_le = LabelEncoder()
            train_ts[col] = train_le.fit_transform(train_ts[col])

            test_le = LabelEncoder()
            test_ts[col] = test_le.fit_transform(test_ts[col])

            if col == 'id':
                # special case since train/test time series have different locations
                # we numerically encode them differently, since id=1 in train is not the same location as id=1 in test
                test_ts['id'] += train_ts['id'].max()+1 # +1 since id is zero-indexed

            train_les[col] = train_le
            test_les[col] = test_le

    # now train/test time series are in tsfresh flat style, which tslearn has utility function to convert from
    X_train = from_tsfresh_dataset(train_ts.loc[:, feature_cols])
    train_regions = ts_kmeans.fit_predict(X_train) # should be a list where l[i] is the region for ith id

    # map each id to its region for the train time series
    train_id_region_mappings = {id: train_regions[i] for i, id in enumerate(train_ts['id'].unique())}
    train_ts['region'] = train_ts['id'].map(train_id_region_mappings)

    # use trained clusterer to cluster test ids
    X_test = from_tsfresh_dataset(test_ts.loc[:, feature_cols])
    test_regions = ts_kmeans.predict(X_test)

    # map each id to its region for the test time series
    test_id_region_mappings = {id: test_regions[i] for i, id in enumerate(test_ts['id'].unique())}
    test_ts['region'] = test_ts['id'].map(test_id_region_mappings)

    # decode the encoded categorical features for each time series
    for col in train_les:
        train_le, test_le = train_les[col], test_les[col]

        if col == 'id':
            test_ts[col] -= train_ts['id'].max()+1
            
        test_ts[col] = test_le.inverse_transform(test_ts[col])
        test_ts[col] = test_ts[col].astype('category')

        train_ts[col] = train_le.inverse_transform(train_ts[col])
        train_ts[col] = train_ts[col].astype('category')

    train_ts, test_ts = train_ts.rename(columns={'time':'event_t', 'id':'event_id'}), test_ts.rename(columns={'time':'event_t', 'id':'event_id'})

    return train_ts, test_ts, ts_kmeans

@register_fe_func('r_cum_sum_precip_1d')
def regional_cum_sum_daily_precip(ts:pd.DataFrame, keep_sum_daily_precip:bool=True):
    '''
    Tracks the cumulative total of precipitation across all locations in a region, grouped by month and separate for y1/y2

    If `keep_sum_daily_precip` is true, additional feature for regional sum of daily precipitation will be created, separate for y1/y2

    Feature type: numerical

    Features created: 1-2
    '''

    y1_data = ts.loc[(ts['event_t'] <= 364)].copy()
    y2_data = ts.loc[(ts['event_t'] > 364)].copy()

    y1_data['r_sum_precip_1d'] = y1_data.groupby(['region','event_t'], sort=False, observed=True)['precipitation'].transform('sum')
    y2_data['r_sum_precip_1d'] = y2_data.groupby(['region','month','event_t'], sort=False, observed=True)['precipitation'].transform('sum')
    
    y1_data['r_cum_sum_precip_1d'] = y1_data.groupby(['event_id','region','month'], sort=False, observed=True)['r_sum_precip_1d'].cumsum()
    y2_data['r_cum_sum_precip_1d'] = y2_data.groupby(['event_id','region','month'], sort=False, observed=True)['r_sum_precip_1d'].cumsum()
    
    if not keep_sum_daily_precip:
        y1_data = y1_data.drop(labels=['r_sum_precip_1d'], axis=1)
        y2_data = y2_data.drop(labels=['r_sum_precip_1d'], axis=1)

    return pd.concat([y1_data, y2_data], axis=0)

@register_fe_func('g_cum_sum_precip_1d')
def global_cum_sum_daily_precip(ts:pd.DataFrame, keep_sum_daily_precip:bool=True):
    '''
    Tracks the cumulative total of precipitation across all locations, grouped by month and separate for y1/y2

    If `keep_sum_daily_precip` is true, additional feature for regional sum of daily precipitation will be created, separate for y1/y2

    Feature type: numerical

    Features created: 1-2
    '''

    y1_data = ts.loc[(ts['event_t'] <= 364)].copy()
    y2_data = ts.loc[(ts['event_t'] > 364)].copy()

    y1_data['g_sum_precip_1d'] = y1_data.groupby(['month','event_t'], sort=False, observed=True)['precipitation'].transform('sum')
    y2_data['g_sum_precip_1d'] = y2_data.groupby(['month','event_t'], sort=False, observed=True)['precipitation'].transform('sum')

    y1_data['g_cum_sum_precip_1d'] = y1_data.groupby(['event_id','month'], sort=False, observed=True)['g_sum_precip_1d'].cumsum()
    y2_data['g_cum_sum_precip_1d'] = y2_data.groupby(['event_id','month'], sort=False, observed=True)['g_sum_precip_1d'].cumsum()
    
    if not keep_sum_daily_precip:
        y1_data = y1_data.drop(labels=['g_sum_precip_1d'], axis=1)
        y2_data = y2_data.drop(labels=['g_sum_precip_1d'], axis=1)

    return pd.concat([y1_data, y2_data], axis=0)

@register_fe_func('g_cum_avg_precip_1d')
def global_cum_avg_daily_precip(ts:pd.DataFrame):
    pass

@register_fe_func('g_cum_avg_precip_1h')
def global_cum_avg_hourly_precip(ts:pd.DataFrame):
    pass

def main(args:argparse.Namespace):
    '''
    Entry point if running module as a script via command line

    Performs feature engineering on train/test time series datasets based on command line args,
    saving new CSV files to disk as outputs for downstream processing
    '''

    # make output dir if specified, otherwise save everything to current working directory
    output_dir = args.output_dir if args.output_dir is not None else os.curdir
    os.makedirs(output_dir, exist_ok=True,)

    # save supplied command line args to disk as JSON
    with open(os.path.join(output_dir, 'cmd_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # save the default config to disk if desired
    if args.get_default_config:
        with open(os.path.join(output_dir, 'default_config.json'), 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)

    # load user-supplied config if specified otherwise just use default one
    config = DEFAULT_CONFIG if args.user_config is None else json.load(open(args.user_config))

    # initialize train/test time series dataframes
    train_ts, test_ts = init_ts(args.train_ts_path), init_ts(args.test_ts_path)

    # perform feature engineering based on config
    total_time = 0.0
    regional_feature_funcs = {}
    # engineer the local and global features first
    for func_name in config:
        # check if there's a feature engineering function associated with this name
        if func_name not in registered_fe_funcs:
            raise Exception(f'Unknown/unsupported feature engineering function {func_name}')
        func = registered_fe_funcs[func_name]

        # store regional feature functions for later calculation
        if func_name.startswith('r'):
            regional_feature_funcs[func_name] = registered_fe_funcs[func_name]
            continue

        # get the parameters to pass into the function (assumed to be a dict)
        params = config[func_name]

        # call the function on both time series using the same params, unpacked from the dict
        print(f'Adding feature {func_name} to train time series...')
        start_time = time.time()
        train_ts = func(train_ts, **params)
        end_time = time.time()
        print(f'Took {end_time-start_time} seconds')
        total_time += (end_time-start_time)
        
        print(f'Adding feature {func_name} to test time series...')
        start_time = time.time()
        test_ts = func(test_ts, **params)
        end_time = time.time()
        print(f'Took {end_time-start_time} seconds')
        total_time += (end_time-start_time)

    if args.cluster_locations >= 2:
        # perform k-means clustering on locations so we can engineer so-called regional features
        print('Performing k_means clustering...')
        start_time = time.time()
        train_ts, test_ts, clusterer = cluster_locations(train_ts, test_ts, n_regions=args.cluster_locations, seed=args.seed)
        end_time = time.time()
        print(f'Took {end_time-start_time} seconds')
        total_time += (end_time-start_time)
        
        # engineer the regional features
        for func_name in regional_feature_funcs:
            params = config[func_name]
            func = registered_fe_funcs[func_name] # we already know the function is registered at this point

            print(f'Adding feature {func_name} to train time series...')
            start_time = time.time()
            train_ts = func(train_ts, **params)
            end_time = time.time()
            print(f'Took {end_time-start_time} seconds')
            total_time += (end_time-start_time)
            
            print(f'Adding feature {func_name} to test time series...')
            start_time = time.time()
            test_ts = func(test_ts, **params)
            end_time = time.time()
            print(f'Took {end_time-start_time} seconds')
            total_time += (end_time-start_time)
            print("-"*75)

    print(f'Finished feature engineering - took {total_time} seconds')

    # move event_t column to very front just for neatness in train/test time series
    train_ts.insert(0, 'event_t', train_ts.pop('event_t'))
    test_ts.insert(0, 'event_t', test_ts.pop('event_t'))

    # move label (flood/no flood) column to very end just for neatness in train time series
    train_ts.insert(len(train_ts.columns)-1, 'label', train_ts.pop('label'))

    # reorder train/test time series
    train_ts = train_ts.sort_values(by=['region','event_t']).reset_index(drop=True)
    test_ts = test_ts.sort_values(by=['region','event_t']).reset_index(drop=True)

    # save feature-engineered time series dataframes to disk as CSV files and print first couple rows
    train_ts.to_csv(path_or_buf=os.path.join(output_dir, 'train_ts.csv'), index=False)
    test_ts.to_csv(path_or_buf=os.path.join(output_dir, 'test_ts.csv'), index=False)
    print(f'Train time series:\n {train_ts.head(n=25)}')
    print('-'*75)
    print(f'Test time series:\n {test_ts.head(n=25)}')

if __name__ == '__main__':
    # entry point if running as a script
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
    