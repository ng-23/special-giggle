'''
Feature engineering for timeseries precipitation data.

Can be run as a script via command line or imported into other modules.
'''

import argparse
import json
import os
import pandas as pd
from special_giggle import utils

# runtime config, maps a feature engineering function name to a dict of its arguments
DEFAULT_CONFIG = {
    'month': {},
    'season': {},
    'hourly_precip': {},
    'daily_precip_intensity': {},
    'total_monthly_precip': {},
    'avg_daily_precip': {},
}

registered_fe_funcs = {} # maps a function name to a feature engineering function

def register_fe_func(func_name:str):
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
    ts['event_id'] = ['_'.join(event_str.split('_')[0:2]) for event_str in ts['event_id']]

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

    return ts

@register_fe_func('hourly_precip')
def hourly_precip(ts:pd.DataFrame):
    '''
    Derives the hourly precipitation for each day from the total daily (24 hours) precipitation

    Feature type: numerical

    Features created: 1
    '''

    if 'precipitation' not in ts: raise Exception('Missing required column - precipitation')
    
    ts['hourly_precip'] = [daily_precip / 24.0 for daily_precip in ts['precipitation']]

    return ts

@register_fe_func('daily_precip_intensity')
def daily_precip_intensity(ts:pd.DataFrame):
    '''
    Tracks daily precipitation intensity based on the hourly precipitation for each day

    American Meteorological Society (AMS) defines intensities for precipitation periods (https://glossary.ametsoc.org/wiki/Rain)

    Feature type: categorical

    Features created: 1
    '''

    if 'hourly_precip' not in ts: raise Exception('Missing required column - hourly_precip')

    ts['daily_precip_intensity'] = [utils.get_precip_intensity(hourly_precip) for hourly_precip in ts['hourly_precip']]
    
    return ts

@register_fe_func('total_monthly_precip')
def total_monthly_precip(ts:pd.DataFrame):
    '''
    Tracks the running total of daily precipitation for each month

    Feature type: numerical

    Features created: 1
    '''

    event_ids = ts['event_id'].unique() # get all event ids

    temp_dfs = []

    # TODO: this is slow - is there any way to speed it up?
    for event_id in event_ids:
        # separating y1 and y2 data for event_id into 2 different dataframes for further processing
        y1_data = ts.loc[(ts['event_id'] == event_id) & (ts['event_t'] <= 364)].copy() # see https://stackoverflow.com/a/66362915
        y2_data = ts.loc[(ts['event_id'] == event_id) & (ts['event_t'] > 364)].copy()

        # create new column in each dataframe to store the cumulative sume of the daily precipitation for each month
        y1_data['total_monthly_precip'] = y1_data.groupby(['month'], sort=False)['precipitation'].cumsum()
        y2_data['total_monthly_precip'] = y2_data.groupby(['month'], sort=False)['precipitation'].cumsum()

        temp_dfs.append(y1_data); temp_dfs.append(y2_data)

    return pd.concat(temp_dfs, axis=0) # vertically concatenate y1/y2 dataframes for each event id

@register_fe_func('avg_daily_precip')
def avg_daily_precip(ts:pd.DataFrame):
    '''
    Tracks the running average of daily precipitation for each month

    Feature type: numerical

    Features created: 1
    '''
    
    event_ids = ts['event_id'].unique()

    temp_dfs = []

    # TODO: this is slow - is there any way to speed it up?
    for event_id in event_ids:
        y1_data = ts.loc[(ts['event_id'] == event_id) & (ts['event_t'] <= 364)].copy()
        y2_data = ts.loc[(ts['event_id'] == event_id) & (ts['event_t'] > 364)].copy()

        # see https://stackoverflow.com/a/56911728/ and https://stackoverflow.com/a/58851846/
        y1_data = y1_data.assign(avg_monthly_precip = y1_data.groupby(['month'], sort=False)['precipitation'].expanding(min_periods=1).mean().reset_index(drop=True))
        y2_data = y2_data.assign(avg_monthly_precip = y2_data.groupby(['month'], sort=False)['precipitation'].expanding(min_periods=1).mean().reset_index(drop=True))

        temp_dfs.append(y1_data); temp_dfs.append(y2_data)

    return pd.concat(temp_dfs, axis=0)

@register_fe_func('rolling_total_daily_precip')
def rolling_total_daily_precip(ts:pd.DataFrame, days:int):
    '''
    Tracks the rolling total of daily precipitation for a given number of past days

    Feature type: numerical

    Features created: 1
    '''
    pass

@register_fe_func('rolling_avg_daily_precip')
def rolling_avg_daily_precip(ts:pd.DataFrame, months:int):
    '''
    Tracks the rolling average of daily precipitation for a given number of past days

    Feature type: numerical

    Features created: 1
    '''
    pass

def lagged_daily_precip_intensity():
    '''
    Tracks the daily precipitation intensity for a given number of past days

    Feature type: categorical

    Features created: variable
    '''
    pass

def rolling_total_monthly_precip():
    '''
    Tracks the total precipitation for a given number of past months

    Feature type: numerical

    Features created: 1
    '''
    pass

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

    # perform feature engineering based on config
    # TODO: track total feature engineering time in seconds
    for func_name in config:
        # check if there's a feature engineering function associated with this name
        if func_name not in registered_fe_funcs:
            raise Exception(f'Unknown/unsupported feature engineering function {func_name}')
        func = registered_fe_funcs[func_name]

        # get the parameters to pass into the function (assumed to be a dict)
        params = config[func_name]

        # call the function on both timeseries using the same params, unpacked from the dict
        print(f'Adding feature {func_name} to train timeseries...')
        train_ts = func(train_ts, **params)
        print(f'Adding feature {func_name} to test timeseries...')
        test_ts = func(test_ts, **params)

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
    