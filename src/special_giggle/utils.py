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
    
def get_precip_intensity(precip_per_hour:float):
    '''
    Determines the precipitation intensity according to American Meteorological Society (AMS) definitions

    Assumes `precip_per_hour` to be in units of millimeters

    See https://glossary.ametsoc.org/wiki/Rain
    '''
    
    if precip_per_hour <= 0.0:
        return 'none'
    elif precip_per_hour > 0.0 and precip_per_hour <= 2.5:
        return 'light'
    elif precip_per_hour > 2.5 and precip_per_hour <= 7.6:
        return 'moderate'
    else:
        return 'heavy'

    