import os
import pickle
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium import webdriver
from termcolor import colored
import time
from datetime import datetime
from datetime import timedelta

def save_obj(obj, file_name):
    if not os.path.exists('obj/'):
        os.makedirs('obj/')
    with open('obj/'+ file_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    with open('obj/' + file_name + '.pkl', 'rb') as f:
        return pickle.load(f)

def interact(driver,xpath,click=True,delay=2,count=100,status_rate=1):
    r,i = None,0
    while r is None:
        try:
            r = driver.find_element_by_xpath(xpath)
            if click:
                r.click()
                return True
            else:
                return r
        except (NoSuchElementException, ElementNotInteractableException) as e:
            if i > count:
                return None
            i += 1
            time.sleep(delay)
            if i % status_rate == 0:
                print(colored('Searching for item...', 'red'))


# one hot times into bool value one_hots
def OneHot(NewName, oldName, df, exceptions=[]):
    uniqueL = df[str(oldName)].unique()
    for item in uniqueL:
        if (item in exceptions):
            df[str(NewName) + '_Other'] =  np.where(df[str(oldName)] == str(item), 1, 0)
        else:
            df[str(NewName) + '_' + str(item)] =  np.where(df[str(oldName)] == str(item), 1, 0)
    df = df.drop([str(oldName)], axis = 1)
    return df
# creat bool weekend/weekday
def weekBool(x):
    x = reformat_time(x, return_value=False)
    dayNum = datetime.strptime(x, "%m/%d/%Y").weekday()
    return 'weekday' if dayNum in [0,1,2,3,4] else 'weekend'
# creat bool rush_hr/non_rush_hr
def rushBool(x):
    x = reformat_time(x, return_value=False, map_date=False) # returns time format hr:min 06:35
    x = time2Mins(x)
    if (int(x) >= 84) and (int(x) <= 120):# 7am-10am
        x = 'rush_hr'
    elif (int(x) >= 180) and (int(x) <= 240):# 3pm-8pm
        x = 'rush_hr'
    else:
        x = 'non_rush_hr'
    return x

