'''
Feature engineering for timeseries precipitation data.

Can be run as a script via command line or imported into other modules.
'''

import argparse
from special_giggle import utils

def get_args_parser():
    '''
    Creates commandline argument parser
    '''

    pass

def init_df():
    '''
    Loads timeseries from CSV and performs base preprocessing
    '''
    pass

def track_location():
    '''
    Extracts the location from the timestamp
    
    Each location is geographically different, so flood conditions may vary across locations

    Feature type: categorical

    Features created: 1
    '''
    pass

def track_month():
    '''
    Derives the month from the timestamp

    Certain months may correspond to a wet/dry season for a particular location,
    which can help explain increased/decreased precipitation that could relate to flood events

    Feature type: categorical

    Features created: 1
    '''
    pass

def track_season():
    '''
    Derives the season from the month, according to https://southafrica-info.com/land/south-africa-weather-climate/

    Feature type: categorical

    Features created: 1
    '''    
    pass

def track_daily_precip_intensity():
    '''
    Tracks daily precipitation intensity

    American Meteorological Society (AMS) defines intensities for precipitation periods (https://glossary.ametsoc.org/wiki/Rain)

    Feature type: categorical

    Features created: 1
    '''
    pass

def track_total_monthly_precip():
    '''
    Tracks the running total of daily precipitation for each month

    Feature type: numerical

    Features created: 1
    '''
    pass

def track_avg_monthly_precip():
    '''
    Tracks the running average of daily precipitation for each month

    Feature type: numerical

    Features created: 1
    '''
    pass

def rolling_total_daily_precip():
    '''
    Tracks the rolling total of daily precipitation for a given number of past days

    Feature type: numerical

    Features created: 1
    '''
    pass

def rolling_avg_daily_precip():
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

def main():
    '''
    Entry point if running module as a script via command line

    Parses command line args and performs feature engineering on train/test timeseries datasets, saving new
    CSV files to disk as outputs for downstream processing
    '''
    pass