import numpy as np
import pandas as pd

from datetime import datetime
from datetime import timedelta

# Date and Time were recorded for multiple data sources.
# Data was matched unilaterally by a time from t=0 column.

# days from anchor date
def DeltaDays(day):
    date_format = "%Y%m%d"
    d0 = datetime.strptime(str(day), date_format)
    d1 = datetime.strptime('20171001', date_format)
    delta = d0 - d1
    delta = (delta.days)
    NewTime = (d1 + timedelta(days=(delta)) ).strftime('%Y%m%d')
    return delta

# days from anchor date
def DeltaDays(day):
    date_format = "%Y%m%d"
    d0 = datetime.strptime(str(day), date_format)
    d1 = datetime.strptime('20171001', date_format)
    delta = d0 - d1
    delta = (delta.days)

    day = int(int(mins)/1440)
    mins = int(mins) - (day*1440)
    hour = int((mins/1440)*24)

    NewTime = (d1 + timedelta(days=(delta)) ).strftime('%Y%m%d')
    return delta

# --- Given min get new date --- #

def min2date(mins):
    date_format = "%Y%m%d"
    day = int(int(mins)/1440)
    mins = int(mins) - (day*1440)
    hour = int((mins/1440)*24)
    delta = day
    date_format = "%Y%m%d"
    d1 = datetime.strptime('20171001', date_format)
    NewTime = (d1 + timedelta(days=(delta)) ).strftime('%Y%m%d')
    return NewTime, day, hour, mins

def date2list(start, end):
    date_format = "%Y%m%d"
    d0 = datetime.strptime(start, date_format).date()
    d1 = datetime.strptime(end, date_format).date()
    delta = d1 - d0
    delta.days
    datelist = []
    datelist.append(d0.strftime('%Y%m%d'))
    count = 0
    delta = d1 - d0
    delta = (delta.days)
    while delta > 0:
        delta -= 1 
        count += 1
        NewTime = (d0 + timedelta(days=(count)) ).strftime('%Y%m%d')
        datelist.append(str(NewTime))    
    return datelist
    
# --- Time converter --- #  
def TimeConverter(item):
        Time = (int(int(item)/100))*60 + (item % 100)
        return Time

def DateTime2List(df):
    time = df[['Date1', 'Time1']]
    date_list = time.Date1.tolist()
    time_list = time.Time1.tolist()
    d1 = time.Date1.apply(DeltaDays, date_list)
    d1 = d1.apply(lambda x: (x * 24 * 60))
    t1 = time.Time1.apply(TimeConverter, time_list)
    t1 = t1.apply(lambda x: 5 * round(x/5))
    df['Time'] = d1 + t1
#     df2 = pd.concat([t1, d1], axis=1)
    return df
