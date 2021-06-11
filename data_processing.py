import os, sys, time
import numpy as np
import sympy
from sympy import *
import pandas as pd
import statistics

import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import random
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import datetime
import json, requests, shutil
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--headless")

# chrome_options.headless = True # also works
driver = webdriver.Chrome(
				#executable_path = '/home/clint/temp/UGA-Masters/VDS_CCS_Project/GUI/GUI/chromedriver',
				options=chrome_options)
from selenium.webdriver.common.keys import Keys
import urllib.request

from selenium.common.exceptions import StaleElementReferenceException, WebDriverException, NoSuchElementException
from termcolor import colored
import plotly.figure_factory as ff

from sklearn import preprocessing
import sklearn.metrics
import sklearn.metrics.pairwise
from skimage.transform import resize
import matplotlib.pyplot as plt
import pywt
import seaborn as sns
import scaleogram as scg

from project_tools import load_obj, interact, save_obj
pipe = load_obj('pickel_pipe_minMax_fixed_full')
pipeline_image = load_obj('pipeline_saved_MM_NORM')


"""
create scroll wheel and station to weather converter
"""
df = pd.read_csv('data/vds_info_table_and_weatherKNN.csv')
df['top_k_near'] = df['k_near'].map(lambda x: x.replace('[','').replace(']','').replace("'",'').replace(",",'').split()[0])
vds2weatherStation = dict(zip(df['ID2'].astype(str).tolist(),df['top_k_near'].tolist()))

# full set!
df = pd.read_csv('data/vds_selected.csv')
OptionList = [{'label': lab, 'value': val} for lab, val in zip(df['ID'].tolist(),df['ID2'].astype(str).tolist())]
vds2stationName = dict(zip(df['ID2'].astype(str).tolist(),df['ID'].tolist()))

from statistics import mean

GDOT_SQL = {'template_url0': 'http://C0007419:GreatDay2021@gdot-reportserver.dot.ga.gov/ReportServer_VSQL04/Pages/ReportViewer.aspx?/vds',
            'template_url1':'Data&id=',
            'template_url2':'&road=0&dir=0&startDate=',
            'template_url3':'%20',
            'template_url4':'%3A',
            'template_url5':'%3A00&endDate=',
            'template_url6':'%20',
            'template_url7':'%3A',
            'template_url8':'%3A00&rs:Command=Render&rc:Parameters=false&rc:Zoom=100&rc:Toolbar=true',
            'banner': '/html/body/form/table/tbody/tr/td/span[2]/div/table/tbody/tr[5]/td[3]/div/div[1]/div/table/tbody/tr/td/table/tbody/tr[1]/td/div/table/tbody/tr[3]/td[5]/table/tbody/tr/td/table/tbody/tr[2]/td[3]/div/div/div'}

culled_vds2stationName = {}
for key in os.listdir('data/historic_ccs_data'):
    key = key.replace('.csv','')
    try:
        culled_vds2stationName[key] = vds2stationName[key]
    except:
        print(key)

# good set
vds2stationName = culled_vds2stationName
sorted_vds2stationName = dict(sorted(vds2stationName.items(), key=lambda item: item[1]))

# OptionList = [{'label': lab, 'value': val} for val, lab in vds2stationName.items()]
OptionList = [{'label': lab, 'value': val} for val, lab in sorted_vds2stationName.items()]

colors = {
    'background': 'rgb(54,54,54)',
    'text': 'rgb(255,255,255)',
    'second': 'rgb(255,255,255)'
}

html_format = {
    'right_col' : '25%',
    'left_col' : '75%'
}

def generate_url(vds_id, start_date, SQL_URL=GDOT_SQL, period=0):
    date_1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    dayb4 = date_1 + datetime.timedelta(days=-1)
    dayb4 = dayb4.strftime("%Y-%m-%d")

    url = SQL_URL['template_url0']
    for i, (var) in enumerate(['5Min', vds_id, start_date, '00', '00', start_date, '23','55']):
        url = url + str(var) + SQL_URL[f'template_url{i+1}']

    urlb4 = SQL_URL['template_url0']


    for i, (var) in enumerate(['5Min', vds_id, dayb4, '00', '00', dayb4, '23','55']):
        urlb4 = urlb4 + str(var) + SQL_URL[f'template_url{i+1}']
    return url, urlb4

def get_table(driver):
    html = driver.page_source
    df_list = pd.read_html(html) # this parses all the tables in webpages to a list
    df = df_list[0]
    col_row = np.where(df.values =='TOT VOL')[0]
    df.columns = df.iloc[col_row,:].values[0] # set col names
    df = df.iloc[(col_row[0]+1):(288+col_row[0]),:] # trim
    cols = df.columns.tolist() # set cols
    cols[2] = 'TIME'
    df.columns = cols
    df['TOT VOL'] = pd.to_numeric(df['TOT VOL'])
    if len(df['TOT VOL']) == 287:
        df2 = df.append(df.iloc[-1,:])
        return df2
    else:
        return df

"""
reformat for prediction of VDS
"""
def inc_data_int(train_data,test_data=[],interval=288):
    # create training and test data
    X_train = []
    y_train = []
    for i in range(interval,len(train_data)):
        X_train.append(train_data[i-interval:i])
        y_train.append(train_data[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshaping X_train for efficient modelling
    X_train = np.array(X_train)
    # Preparing X_test
    if len(test_data)>0:
        X_test = []
        y_test = []
        for i in range(interval,len(test_data)):
            X_test.append(test_data[i-interval:i])
            y_test.append(test_data[i])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.array(X_test)
        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train

"""
Weather data info
"""
def get_weather_url(driver, date, station='KGAATLAN528'):
    station='KGALILBU42'
    url = f'https://www.wunderground.com/history/daily/us/ga/lilburn/{station}/date/{date}'
    driver.get(url)

def targeted_scrape(driver, I, data, K, ID='NA'):
#     station = 'KGALILBU42'
#     url = f'https://www.wunderground.com/history/daily/us/ga/lilburn/{station}/date/{date}'
#     driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html)
    table = '//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]/table'
    username = interact(driver,table,click=False,delay=0.2,count=100)
    data[f'k{K}-{ID}'] = username.text.split('\n') if username else f'missing {station}'
    return data

def get_weather_data(driver):
#     station = 'KGAATLAN528'
    data = {}
    total = []
    data = targeted_scrape(driver, 'new', data, '1')
    total = [] # get Condition weather data ie t-storm
    for x in data['k1-NA'][10:]:
        x = x.split(' in ')[2]
        total.append(x)
    fill = [] # stretching the smaller interval weather data
    for idx, item in enumerate(total):
        if idx == (len(total)-1):
            num = 288 - len(fill)
        else:
            num = floor(288/len(total)) if idx%2 == 0 else ceiling(288/len(total))
        fill = fill + [item]*num
    return fill

groupings = {
             'Haze':'ImpartedVisibility',
             'ShallowFog':'ImpartedVisibility',
             'DrizzleandFog':'ImpartedVisibility',
             'LightDrizzle':'LightRain',
             'WintryMix':'Snow',
             'LightSnow':'Snow',
             'ThunderintheVicinity':'T-Storm',
             'Fog':'ImpartedVisibility',
             'LightRainwithThunder':'T-Storm',
             'Thunder':'T-Storm',
             '':'Other',
             'N/A':'Other',
             'Fair':'Cloudy',
             'WidespreadDust':'Other',
             'MostlyCloudy':'Cloudy',
             'PartlyCloudy':'Cloudy',
              }

group2val = {
             'ImpartedVisibility':1,
             'LightRain':1,
             'Snow':2,
             'Snow':2,
             'T-Storm':2,
             'Other':0,
             'Cloudy':0,
             'Rain':2
              }

def format_conditions(x):
    x = x.replace(' ','') if ' ' in x else x
    x = x.replace('Heavy','') if 'Heavy' in x else x
    x = x.replace('N/A','Other') if 'N/A' in x else x
    x = x.split('/') if '/' in x else [x]
    for idx, (item) in enumerate(x):
        if item in groupings.keys():
            x[idx] = groupings[item]
    return x[0]


"""
Weather data info
"""
def predict_vds(flat_arr, model, LTC=True):
    flat = flat_arr.reshape(-1, 1)
    data = pipe.transform(flat)
    data_inc, _ = inc_data_int(data,interval=12)
    data = np.reshape(data_inc, (data_inc.shape[0], data_inc.shape[1], 1))
    save_obj(data, 'data_test')
    prediction_vds = model(data).numpy()
    if LTC == True:
        pred = np.transpose(prediction_vds[:,-1])[0][-288:] # last day only
    else:
        pred = np.transpose(prediction_vds[:,-1])[-288:]
    pred = np.reshape(pred, (1, pred.shape[0]))
    pred = pipe.inverse_transform(pred)
    return pred[0]


# def predict_offline(flat_arr, model, LTC=True):
#     flat = flat_arr.reshape(-1, 1)
#     data = pipe.transform(flat)
#     data_inc, _ = inc_data_int(data,interval=12)
#     data = np.reshape(data_inc, (data_inc.shape[0], data_inc.shape[1], 1))
#     save_obj(data, 'data_test')
#     prediction_vds = model(data).numpy()
#     if LTC == True:
#         pred = np.transpose(prediction_vds[:,-1])[0][-288:] # last day only
#     else:
#         pred = np.transpose(prediction_vds[:,-1])[-288:]
#     pred = np.reshape(pred, (1, pred.shape[0]))
#     pred = pipe.inverse_transform(pred)
#     return pred[0]



def generate_url_change(vds_id, old_hr, old_min, SQL_URL=GDOT_SQL):
    now_ = datetime.datetime.now()
    date, hr, min_, _ = now_.strftime("%Y-%m-%d %H %M %S").split()

    url = SQL_URL['template_url0']
    for i, (var) in enumerate(['Raw', vds_id, date, old_hr, old_min, date, hr, min_]):
        url = url + str(var) + SQL_URL[f'template_url{i+1}']
    return url, hr, min_


def generate_url_todays(vds_id, SQL_URL=GDOT_SQL):
    now_ = datetime.datetime.now()
    timeb4 = now_ + datetime.timedelta(hours=-7)
    hrB4, minB4 = timeb4.strftime('%H %M').split()
    date, hr, min_, _ = now_.strftime("%Y-%m-%d %H %M %S").split()

    url = SQL_URL['template_url0']
    print(hrB4)
    for i, (var) in enumerate(['Raw', vds_id, date, hrB4, 0, date, hr, min_]):
        url = url + str(var) + SQL_URL[f'template_url{i+1}']
    return url, hr, min_

def get_live_full(driver):
    url, old_hr, old_min = generate_url_todays('5917', SQL_URL=GDOT_SQL)
    driver.get(url)
    print(url)
    banner =  interact(driver, GDOT_SQL['banner'],click=False,delay=0.01,count=850,status_rate=300)
    html = driver.page_source
    df_list = pd.read_html(html) # high time cost
    df = df_list[0] # set col names
    col_row = np.where(df.values =='TOT VOL')[0]
    df.columns = df.iloc[col_row,:].values[0]

    cols = df.columns.tolist() # set cols ro list
    cols[2] = 'TIME'
    df.columns = cols # name the time column

    df = df.iloc[(col_row[0]+1):-4,:] # remove bottom
    lst = pd.to_numeric(df['TOT VOL']).to_list()
    full_times = df['TIME'].to_list()

    idx, mean_vols, times = 0, [], []
    while idx < len(lst)-15:
        mean_vols.append(statistics.mean(lst[idx:(idx+15)]))
        times.append(full_times[idx])
        idx+=15

    pred = predict_vds(np.array(mean_vols), LTC_model, LTC=True)
    pred_LSTM = predict_vds(np.array(mean_vols), LSTM_model, LTC=False)

    x = [x for x in range(len(pred))]

    cut = lst[idx:]
    return pred, mean_vols, pred_LSTM, times, cut, old_hr, old_min


def get_weather_url(driver, date, station='KGALILBU42'):
#     station='KGALILBU42'
    url = f'https://www.wunderground.com/history/daily/us/ga/lilburn/{station}/date/{date}'
    print(url)
    driver.get(url)

def targeted_scrape(driver, I, data, K, ID='NA'):
#     station = 'KGALILBU42'
#     url = f'https://www.wunderground.com/history/daily/us/ga/lilburn/{station}/date/{date}'
#     driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html)
    table = '//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]/table'
    username = interact(driver,table,click=False,delay=0.2,count=100)
    data[f'k{K}-{ID}'] = username.text.split('\n') if username else f'missing {station}'
    return data

def get_weather_data(driver, length=288):
#     station = 'KGAATLAN528'
    data = {}
    total = []
    data = targeted_scrape(driver, 'new', data, '1')
    total = [] # get Condition weather data ie t-storm
    for x in data['k1-NA'][10:]:
        x = x.split(' in ')[2]
        total.append(x)
    fill = [] # stretching the smaller interval weather data
    for idx, item in enumerate(total):
        if idx == (len(total)-1):
            num = length - len(fill)
        else:
            num = floor(length/len(total)) if idx%2 == 0 else ceiling(length/len(total))
        fill = fill + [item]*num
    return fill




"""
data space
"""
def create_plot(data, title, full_label='', score=0):
    color = 'white'
    if score > 4.2:
        color = 'yellow'

    image = go.Figure(px.imshow(data))
    image.update_layout(
                    title_text=full_label,
                    title_font_size=20,
                    title_font_color = color,
                    plot_bgcolor = colors['background'],
                    paper_bgcolor = colors['background'],
                    coloraxis_showscale=False,
                    xaxis_title=f'{title}',
                    font=dict(
                                family="Courier New, monospace",
                                size=14,
                                color="white"),)

    return image


def get_im_transf(df_ser, pipeline):
    X = (df_ser.to_numpy()).reshape(1, -1)
    X_transformed = pipeline.transform(X) # reshape and scale
    return X_transformed

zeros_df = pd.DataFrame()
zeros_df['zeros'] = [0 for x in range(288)]
trans_zeros = get_im_transf(zeros_df['zeros'], pipeline_image)


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def getNearestNeighborV2(trainingSet, testInstance, ccsInstance, zeros=zeros):
    # Get an array of the distances between our image and every training image
    distances = np.linalg.norm(trainingSet - testInstance, axis = 1)
    df = pd.DataFrame()
    df['distances'] = distances
    df = df.sort_values(by='distances', ascending=True)[:64] # NN 64

    # Collect data of 64 NN's of CCS space
    pca_comp = 20
    nn_data = np.zeros((64,64))
    for idx, (nn_idx) in enumerate(df.index.values):
        nn_data[idx] = trainingSet[nn_idx]

    # Fit PCA to 64 NN's
    pipe_pca = Pipeline([('reduce_dims', PCA(n_components=pca_comp))])
    pipe_pca.fit(nn_data)
    # project to space and find distance VDS
    testing_point = np.reshape(testInstance, (1, len(testInstance)))
    projected_testing = pipe_pca.transform(testing_point)
    # project to space and find distance CCS
    testing_point_ccs = np.reshape(ccsInstance, (1, len(ccsInstance)))
    projected_testing_ccs = pipe_pca.transform(testing_point_ccs)

    d_pca_vds = (((projected_testing**2).sum(1))**(1/2))
    d_pca_ccs = (((projected_testing_ccs**2).sum(1))**(1/2))

    mean_64nn = np.mean(nn_data, axis=0)

    print(mean_64nn.shape)

    d_hyp_vds = ((((testInstance-mean_64nn)**2).sum())**(1/2))
    d_hyp_ccs = ((((ccsInstance-mean_64nn)**2).sum())**(1/2))

    d_vds = (d_hyp_vds**2-d_pca_vds)**(1/2)
    d_ccs = (d_hyp_ccs**2-d_pca_ccs)**(1/2)

    d3 = ((((testInstance-ccsInstance)**2).sum())**(1/2))

    return d_vds[0], d_ccs[0], d3


def getNearestNeighborV3(trainingSet, testInstance, ccsInstance, zeros=zeros):
    # Get an array of the distances between our image and every training image
    distances = np.linalg.norm(trainingSet - testInstance, axis = 1)
    df = pd.DataFrame()
    df['distances'] = distances
    df = df.sort_values(by='distances', ascending=True)[:64] # NN 64

    # Collect data of 64 NN's of CCS space
    pca_comp = 21
    nn_data = np.zeros((64,64))
    for idx, (nn_idx) in enumerate(df.index.values):
        nn_data[idx] = trainingSet[nn_idx]

    # Fit PCA to 64 NN's
    pipe_pca = Pipeline([('reduce_dims', PCA(n_components=pca_comp))])
    pipe_pca.fit(nn_data)
    # project to space and find distance VDS
    testing_point = np.reshape(testInstance, (1, len(testInstance)))
    projected_testing = pipe_pca.transform(testing_point)
    d_vds = projected_testing[-1]
    d_vds = ((((d_vds)**2).sum())**(1/2))

    # project to space and find distance CCS
    testing_point_ccs = np.reshape(ccsInstance, (1, len(ccsInstance)))
    projected_testing_ccs = pipe_pca.transform(testing_point_ccs)
    d_ccs = projected_testing_ccs[-1]
    d_ccs = ((((d_ccs)**2).sum())**(1/2))

    d3 = ((((testInstance-ccsInstance)**2).sum())**(1/2))

    return d_vds, d_ccs, d3


"""
recurrence
"""

reshape_size_recur = 32
n_scales = 64
reshape_size = 64
interval_len = 288


def recurrence_plot(s, reshape_size, eps=None, steps=None):
    if eps==None: eps=0.1
    if steps==None: steps=100
    d = sklearn.metrics.pairwise.pairwise_distances(s)
    d = np.floor(d / eps)
    d[d > steps] = steps
    d = resize(d, (reshape_size, reshape_size), mode='constant')
    return d


def collect_recurrence_data(X, reshape_size):
    print('generating recurrence images...')
    n_samples = len(X)
    # pre allocate array
    df_X = pd.DataFrame()
    X_recur = np.ndarray(shape=(n_samples, reshape_size, reshape_size), dtype = 'float32')
    for sample in range(n_samples):
        df_X['norm'] = X[sample]
        im = recurrence_plot(df_X, reshape_size, eps=0.5, steps=20)
        X_recur[sample,:,:] = im
    print(colored('Done!','blue'))
    return X_recur

def collect_cwt_data(X, reshape_size, n_scales, interval_len=288, wavelet_name = "morl"):
    print('generating wavelet images...')
    n_samples = len(X)
    scales = np.arange(1, n_scales + 1)
    # pre allocate array
    X_cwt = np.ndarray(shape=(n_samples, reshape_size, reshape_size), dtype = 'float32')
    for sample in range(n_samples):
        serie = X[sample]
        # continuous wavelet transform
        coeffs, freqs = pywt.cwt(serie, scales, wavelet_name)
        coeffs = resize(coeffs, (reshape_size, reshape_size), mode = 'constant')
        X_cwt[sample,:,:] = coeffs
    print(colored('Done!','blue'))
    return X_cwt
