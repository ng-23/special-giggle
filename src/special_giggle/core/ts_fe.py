'''
Feature engineering for timeseries precipitation data.

Can be run as a script via command line or imported into other modules.
'''

import argparse
import json
import os
import pandas as pd
import time
from special_giggle import utils

# runtime config, maps a feature engineering function name to a dict of its arguments
DEFAULT_CONFIG = {
    'month': {},
    'season': {},
    'precip_1H': {},
    'precip_intensity_1d': {},
    'cum_sum_precip_1d': {},
    'cum_avg_precip_1d': {},
    'cum_avg_precip_1H': {},
    'cum_std_precip_1d': {},
    'lag_cum_sum_precip_1d': {'days':3},
    'lag_cum_avg_precip_1d': {'days':3},
    'lag_cum_avg_precip_1H': {'days':3},
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
        description='Timeseries Feature Engineering', 
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
        '--user-config', 
        type=str, 
        default=None,
        help='Filepath to JSON runtime configuration file. Overrides default configuration if specified',
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
    Loads timeseries from CSV and performs base preprocessing
    '''
    
    ts = pd.read_csv(filepath_or_buffer=filepath)

    # each event is of the format id_location_X_timestamp
    # we wish to separate the event identifier (aka the location) from the timestamp
    ts['event_id'] = sorted(['_'.join(event_str.split('_')[0:2]) for event_str in ts['event_id']])

    # at this point the event identifier (location) is separate from the timestamp
    # but in the process we've lost the timestamp, so we need to get it back
    # each location has data for 730 days, so we need 730 timestamps per location
    ts['event_t'] = ts.groupby('event_id').cumcount()

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

@register_fe_func('precip_1H')
def hourly_precip(ts:pd.DataFrame):
    '''
    Derives the hourly precipitation for each day from the total daily (24 hours) precipitation

    Feature type: numerical

    Features created: 1
    '''

    if 'precipitation' not in ts: raise Exception('Missing required column - precipitation')
    
    ts['precip_1H'] = [daily_precip / 24.0 for daily_precip in ts['precipitation']]

    return ts

@register_fe_func('precip_intensity_1d')
def daily_precip_intensity(ts:pd.DataFrame):
    '''
    Tracks daily precipitation intensity based on the hourly precipitation for each day

    American Meteorological Society (AMS) defines intensities for precipitation periods (https://glossary.ametsoc.org/wiki/Rain)

    Feature type: categorical

    Features created: 1
    '''

    if 'precip_1H' not in ts: raise Exception('Missing required column - precip_1H')

    ts['precip_intensity_1d'] = [utils.get_precip_intensity(hourly_precip) for hourly_precip in ts['precip_1H']]
    ts['precip_intensity_1d'] = ts['precip_intensity_1d'].astype('category')
    
    return ts

@register_fe_func('std_precip_1d')
def std_daily_precip(ts:pd.DataFrame):
    '''
    Tracks the standard deviation of daily precipitation for each month, grouped by event_id and separate for y1/y2

    Feature type: numerical

    Features created: 1
    '''

    y1_data = ts.loc[(ts['event_t'] <= 364)].copy()
    y2_data = ts.loc[(ts['event_t'] > 364)].copy()

    # TODO: this seems to produce NaNs - how to fix?
    y1_data['std_precip_1d'] = y1_data.groupby(['event_id','month'])['precipitation'].std().reset_index(level=[0,1], drop=True)
    y2_data['std_precip_1d'] = y2_data.groupby(['event_id','month'])['precipitation'].std().reset_index(level=[0,1], drop=True)

    return pd.concat([y1_data,y2_data], axis=0) # vertically concatenate y1/y2 dataframes

@register_fe_func('std_precip_1H')
def std_hourly_precip(ts:pd.DataFrame):
    '''
    Tracks the standard deviation of hourly precipitation for each month, grouped by event_id and separate for y1/y2

    Feature type: numerical

    Features created: 1
    '''

    pass

@register_fe_func('cum_sum_precip_1d')
def cum_sum_daily_precip(ts:pd.DataFrame):
    '''
    Tracks the cumulative total of daily precipitation for each month, grouped by event_id and separate for y1/y2

    Feature type: numerical

    Features created: 1
    '''

    y1_data = ts.loc[(ts['event_t'] <= 364)].copy() # see https://stackoverflow.com/a/66362915
    y2_data = ts.loc[(ts['event_t'] > 364)].copy()

    y1_data['cum_sum_precip_1d'] = y1_data.groupby(['event_id','month'], sort=False, observed=True)['precipitation'].expanding(min_periods=1).sum().reset_index(level=[0,1], drop=True)
    y2_data['cum_sum_precip_1d'] = y2_data.groupby(['event_id','month'], sort=False, observed=True)['precipitation'].expanding(min_periods=1).sum().reset_index(level=[0,1], drop=True)

    return pd.concat([y1_data, y2_data], axis=0)

@register_fe_func('cum_avg_precip_1d')
def cum_avg_daily_precip(ts:pd.DataFrame):
    '''
    Tracks the cumulative average of daily precipitation for each month, grouped by event_id and separate for y1/y2

    Feature type: numerical

    Features created: 1
    '''
    
    y1_data = ts.loc[(ts['event_t'] <= 364)].copy()
    y2_data = ts.loc[(ts['event_t'] > 364)].copy()

    # see https://stackoverflow.com/a/56911728/ and https://stackoverflow.com/a/58851846/
    y1_data['cum_avg_precip_1d'] = y1_data.groupby(['event_id','month'], sort=False, observed=True)['precipitation'].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)
    y2_data['cum_avg_precip_1d'] = y2_data.groupby(['event_id','month'], sort=False, observed=True)['precipitation'].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)

    return pd.concat([y1_data, y2_data], axis=0)

@register_fe_func('cum_avg_precip_1H')
def cum_avg_hourly_precip(ts:pd.DataFrame):
    '''
    Tracks the cumulative average of hourly precipitation for each month, grouped by event_id and separate for y1/y2

    Feature type: numerical

    Features created: 1
    '''

    y1_data = ts.loc[(ts['event_t'] <= 364)].copy()
    y2_data = ts.loc[(ts['event_t'] > 364)].copy()

    y1_data['cum_avg_precip_1H'] = y1_data.groupby(['event_id','month'], sort=False, observed=True)['precip_1H'].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)
    y2_data['cum_avg_precip_1H'] = y2_data.groupby(['event_id','month'], sort=False, observed=True)['precip_1H'].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)

    return pd.concat([y1_data, y2_data], axis=0)

@register_fe_func('cum_std_precip_1d')
def cum_std_daily_precip(ts:pd.DataFrame):
    '''
    Tracks the cumulative standard deviation of daily precipitation for each month, grouped by event_id and separate for y1/y2

    Feature type: numerical

    Features created: 1
    '''

    y1_data = ts.loc[(ts['event_t'] <= 364)].copy()
    y2_data = ts.loc[(ts['event_t'] > 364)].copy()

    # TODO: what's the best way to fill NaNs?
    y1_data['cum_std_precip_1d'] = y1_data.groupby(['event_id','month'], sort=False, observed=True)['precipitation'].expanding(min_periods=1).std().reset_index(level=[0,1], drop=True)
    y1_data['cum_std_precip_1d'] = y1_data['cum_std_precip_1d'].fillna(y1_data.groupby(['event_id','month'], sort=False, observed=True)['cum_std_precip_1d'].median().reset_index(level=[0,1], drop=True))
    
    y2_data['cum_std_precip_1d'] = y2_data.groupby(['event_id','month'], sort=False, observed=True)['precipitation'].expanding(min_periods=1).std().reset_index(level=[0,1], drop=True)
    y2_data['cum_std_precip_1d'] = y2_data['cum_std_precip_1d'].fillna(y1_data.groupby(['event_id','month'], sort=False, observed=True)['cum_std_precip_1d'].median().reset_index(level=[0,1], drop=True))

    return pd.concat([y1_data, y2_data], axis=0)

@register_fe_func('cum_std_precip_1H')
def cum_std_hourly_precip(ts:pd.DataFrame):
    '''
    Tracks the cumulative standard deviation of hourly precipitation 

    Feature type: numerical

    Features created: 1
    '''

    pass

@register_fe_func('lag_cum_sum_precip_1d')
def lag_cum_sum_daily_precip(ts:pd.DataFrame, days:int):
    '''
    Tracks the rolling total of daily precipitation for a given number of past days, grouped by event_id

    Feature type: numerical

    Features created: 1
    '''

    if 'precipitation' not in ts: raise Exception('Missing required column - precipitation')

    col_name = f'lag_{days}d_cum_sum_precip_1d'

    lag_data = ts.groupby(['event_id'])['precipitation'].rolling(window=days, min_periods=1).sum().reset_index(level=[0], drop=True)

    ts[col_name] = lag_data

    return ts

@register_fe_func('lag_cum_avg_precip_1d')
def lag_cum_avg_daily_precip(ts:pd.DataFrame, days:int):
    '''
    Tracks the rolling average of daily precipitation for a given number of past days, grouped by event_id

    Feature type: numerical

    Features created: 1
    '''
    
    if 'precipitation' not in ts: raise Exception('Missing required column - precipitation')

    col_name = f'lag_{days}d_cum_avg_precip_1d'

    lag_data = ts.groupby(['event_id'])['precipitation'].rolling(window=days, min_periods=1).mean().reset_index(level=[0], drop=True)

    ts[col_name] = lag_data

    return ts

@register_fe_func('lag_cum_avg_precip_1H')
def lag_cum_avg_hourly_precip(ts:pd.DataFrame, days:int):
    '''
    Tracks the rolling average of hourly precipitation for a given number of past days, grouped by event_id

    Feature type: numerical

    Features created: 1
    '''

    if 'precip_1H' not in ts: raise Exception('Missing required column - precip_1H')

    col_name = f'lag_{days}d_cum_avg_precip_1H'

    lag_data = ts.groupby(['event_id'])['precip_1H'].rolling(window=days, min_periods=1).mean().reset_index(level=[0], drop=True)

    ts[col_name] = lag_data

    return ts

def main(args:argparse.Namespace):
    '''
    Entry point if running module as a script via command line

    Performs feature engineering on train/test timeseries datasets based on command line args,
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

    # initialize train/test timeseries dataframes
    train_ts, test_ts = init_ts(args.train_ts_path), init_ts(args.test_ts_path)

    # reorder train/test timeseries by event_t to better capture temporal structure
    # sort event_id alphabetically just for neatness
    train_ts = train_ts.sort_values(by=['event_t','event_id']).reset_index(drop=True)
    test_ts = test_ts.sort_values(by=['event_t','event_id']).reset_index(drop=True)

    # perform feature engineering based on config
    total_time = 0.0
    for func_name in config:
        # check if there's a feature engineering function associated with this name
        if func_name not in registered_fe_funcs:
            raise Exception(f'Unknown/unsupported feature engineering function {func_name}')
        func = registered_fe_funcs[func_name]

        # get the parameters to pass into the function (assumed to be a dict)
        params = config[func_name]

        # call the function on both timeseries using the same params, unpacked from the dict
        print(f'Adding feature {func_name} to train timeseries...')
        start_time = time.time()
        train_ts = func(train_ts, **params)
        end_time = time.time()
        print(f'Took {end_time-start_time} seconds')
        total_time += (end_time-start_time)
        
        print(f'Adding feature {func_name} to test timeseries...')
        start_time = time.time()
        test_ts = func(test_ts, **params)
        end_time = time.time()
        print(f'Took {end_time-start_time} seconds')
        total_time += (end_time-start_time)
        print("-"*75)
    print(f'Finished feature engineering - took {total_time} seconds')

    # move event_t column to very front just for neatness in train/test timeseries
    train_ts.insert(0, 'event_t', train_ts.pop('event_t'))
    test_ts.insert(0, 'event_t', test_ts.pop('event_t'))

    # move label (flood/no flood) column to very end just for neatness in train timeseries
    train_ts.insert(len(train_ts.columns)-1, 'label', train_ts.pop('label'))

    # save feature-engineered timeseries dataframes to disk as CSV files and print first couple rows
    train_ts.to_csv(path_or_buf=os.path.join(output_dir, 'train_ts.csv'), index=False)
    test_ts.to_csv(path_or_buf=os.path.join(output_dir, 'test_ts.csv'), index=False)
    print(f'Train timeseries:\n {train_ts.head(n=25)}')
    print('-'*75)
    print(f'Test timeseries:\n {test_ts.head(n=25)}')

if __name__ == '__main__':
    # entry point if running as a script
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
    