'''
Contains various utilty code used in other modules/notebooks.
'''

from datetime import datetime
from datetime import timedelta

def get_month_from_year_day(year, day, zero_indexed=True):
    '''
    Determines the month from the day of the year

    Largely taken from https://stackoverflow.com/a/32047761/
    '''

    if zero_indexed:
        date = datetime(year, 1, 1) + timedelta(day)
    else:
        date = datetime(year, 1, 1) + timedelta(day - 1)

    return date.strftime('%B').lower()

def get_season_from_month(month:str):
    '''
    Determines the season from the month

    Assumes location is South Africa, so seasons are determined according to https://southafrica-info.com/land/south-africa-weather-climate/
    '''
    
    month = month.lower()

    if month in ('september','october','november'):
        return 'spring'
    elif month in ('december','january','february'):
        return 'summer'
    elif month in ('march','april','may'):
        return 'autumn'
    elif month in ('june','july','august'):
        return 'winter'
    else:
        raise ValueError(f'Unknown month {month}')
    