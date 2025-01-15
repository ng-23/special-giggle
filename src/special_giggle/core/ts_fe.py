
# The aim of this scrupt is to derive and introduce additional relevant features to the train/test timeseries datasets 
# as a means of enhancing the model's performance.

def init_df():
    '''
    Loads timeseries from CSV and performs base preprocessing
    '''
    pass

def extract_location():
    '''
    Extracts the location from the timestamp
    
    Each location is geographically different, so flood conditions may vary across locations

    Feature type: categorical
    '''
    pass

def derive_month():
    '''
    Derives the month from the timestamp

    Certain months may correspond to a wet/dry season for a particular location,
    which can help explain increased/decreased precipitation that could relate to flood events

    Feature type: categorical
    '''
    pass

def derive_season():
    '''
    Derives the season from the month, according to https://southafrica-info.com/land/south-africa-weather-climate/

    Feature type: categorical
    '''    
    pass

def derive_precip_intensity():
    '''
    Derives daily precipitation intensity

    American Meteorological Society (AMS) defines intensities for precipitation periods (https://glossary.ametsoc.org/wiki/Rain)

    Feature type: categorical
    '''
    pass

def track_monthly_total_precip():
    '''
    Tracks the running total of daily precipitation for each month

    Feature type: numerical
    '''
    pass

def track_monthly_avg_precip():
    '''
    Tracks the running average of daily precipitation for each month

    Feature type: numerical
    '''
    pass

def lagged_precip():
    '''
    Tracks the previous daily precipitation for a given number of past days

    Feature type: numerical
    '''
    pass

def lagged_precip_intensity():
    '''
    Tracks the previous daily precipitation intensity for a given number of past days

    Feature type: numerical
    '''
    pass