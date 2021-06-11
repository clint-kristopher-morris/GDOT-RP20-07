import os, sys, time
import numpy as np
import sympy
from sympy import *
import pandas as pd
import statistics

# set working dir
# working_dir = '/home/clint/temp/UGA-Masters/VDS_CCS_Project/GUI/GUI'
# os.chdir(working_dir)
from project_tools import load_obj, interact, save_obj
pipe = load_obj('pickel_pipe_minMax_fixed_full')

from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from plotly.subplots import make_subplots

# load model
from kerasncp import wirings
from kerasncp.tf import LTCCell

LSTM_model = keras.models.load_model('models/models_universal/LSTM_Global.epoch01.hdf5')
LTC_model = keras.models.load_model('models/models_universal/LTC_Global_v6.epoch07.hdf5')

"""
VDS data functions
"""
#vpn.dot.ga.gov
private_info = {'username': 'C0007419','password': 'GreatDay2021'}
private_info['password'] = 'GreatDay2021'

"""
load all data_processing helper functions
"""
from data_processing import *
"""
load vae models
"""
from VAE_models import *

"""
dynamic
"""
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
from dash.exceptions import PreventUpdate
import dash_daq as daq

colors = {
    'background': 'rgb(54,54,54)',
    'text': 'rgb(255,255,255)',
    'second': 'rgb(255,255,255)'
}

html_format = {
    'right_col' : '25%',
    'left_col' : '75%'
}

#innit driver
num_workers = 3
drivers = {}
# url, urlb4 = generate_url('5917', '2019-08-04', SQL_URL=GDOT_SQL)
for x in range(num_workers):
    drivers[x] = webdriver.Chrome(
				  #executable_path = '/home/clint/temp/UGA-Masters/VDS_CCS_Project/GUI/GUI/chromedriver',
                                  options=chrome_options)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    [
        html.Img(
        src="https://i.ibb.co/HDDYFLp/logo-gdot.png",
        className='four columns',
        style={
                'height': '9%',
                'width': '15%',
                'float': 'right',
                'position': 'relative',
                'margin-top': 10,'margin-right': 25,}),
        html.Img(
        src="https://i.ibb.co/VHmK0rG/GEORGIA-FS-W-1024x335.png",
        className='four columns',
        style={
                'height': '9%',
                'width': '15%',
                'float': 'right',
                'position': 'relative',
                'margin-top': 10,'margin-right': 25,}),
        
        html.H1("UGA-GDOT Anomaly Detector", style={'color': 'white'}),
        html.Div(html.P([
                        html.Div(html.Div([html.P('Collect Live Data',style={'color': 'white'}),
                                   daq.ToggleSwitch(id='switch',value=False,)],
                                 style={"width": "90%",'background-color': colors['background']}),
                                 style={"textAlign":"center","width": "100%", "float": "center",'marginTop': 130,
                                        'fontColor': 'white','background-color': colors['background']}),
            
                        dcc.Input(id="Start_Date", placeholder="Enter date: YYYY-MM-DD",
                                                    type='text',value='',
                                                    style={'marginTop': 25, "width": "95%",'marginLeft': 25,
                                                           'background-color': colors['second'],}),
            
                        dcc.Loading(id="Start_Date-1",type="default",
                                                    children=html.Div(id="Start_Date-output-1"),),
                         
                        dcc.Dropdown(id='dd_station', options=OptionList,
                                                    value='5917',
                                                    placeholder="Select VDS Station",
                                                    style={"textAlign":"left","width": "95%",
                                                           'marginLeft': 25,'marginTop': 25,
                                                           'background-color': colors['second']})],
            
                                                    style={'color':'black','fontColor': 'white'}),

                         
            style={"textAlign":"right","width": html_format['right_col'], "float": "left",'color':'black'},
        ),
        dcc.Graph(id="graph", style={"width": html_format['left_col'], "display": "inline-block","height": "200%",},
                         figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}}),
        
        html.Div(html.Div(html.Iframe(id='map', srcDoc=open('static/assets/matched.html', 'r').read(), 
                                      title='VDS: Blue, CCS: Red',width='100%', height='320'),
                style={"width": "90%","display":"inline-block",'marginLeft': "7%",'backgroundColor': colors['background']}),
                 style={"width": html_format['right_col'],"display":"inline-block","height": "100%",
                        "float": "left", 'backgroundColor': colors['background']}),
                 
        dcc.Graph(id="graph-re", style={"width": html_format['left_col'], "display": "inline-block","height": "200%", "float": "right"},
                        figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}}),
        dcc.Graph(id="graph-we", style={"width": html_format['left_col'], "display": "inline-block","height": "200%", 'marginLeft': html_format['right_col']},
                        figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}}),
        
        
        # fig row one and two
        html.Div(html.Div([
            
                dcc.Graph(id="image1", style={"display": "inline-block","height":'27%',"width":'24%',"float":"left"},
                         figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}}),
                dcc.Graph(id="image2", style={"display": "inline-block","height": '27%',"width":'24%',"float":"left"},
                         figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}}),
            
                # newly added
                dcc.Graph(id="image3", style={"display": "inline-block","height":'27%',"width":'24%',"float":"left"},
                figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}}),
                dcc.Graph(id="image4", style={"display": "inline-block","height":'27%',"width":'24%',"float":"left"},
                         figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}})
        
        
        ],
                         style={'paper_bgcolor': colors['background'],'marginLeft': '25%'}),
                         style={"width": '100%',"display":"inline-block","height": "100%",
                                'backgroundColor': colors['background']}),

        # fig row one and two
        html.Div(html.Div([
            
                dcc.Graph(id="image5", style={"display": "inline-block","height":'27%',"width":'24%',"float":"left"},
                         figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}}),
                dcc.Graph(id="image6", style={"display": "inline-block","height": '27%',"width":'24%',"float":"left"},
                         figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}}),
            
                # newly added
                dcc.Graph(id="image7", style={"display": "inline-block","height":'27%',"width":'24%',"float":"left"},
                figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}}),
                dcc.Graph(id="image8", style={"display": "inline-block","height":'27%',"width":'24%',"float":"left"},
                         figure={'data': [],'layout': {'plot_bgcolor': colors['background'],
                                'paper_bgcolor': colors['background'],
                                'font': {'color': colors['text']}}})
        
        
        ],
                         style={'paper_bgcolor': colors['background'],'marginLeft': '25%'}),
                         style={"width": '100%',"display":"inline-block","height": "100%",
                                'backgroundColor': colors['background']}),
             
 
        
        dcc.Interval(id='graph-update',interval=5*60*1000),
#         dcc.Interval(id='graph-update',interval=10*1000),
        dcc.Store(id='intermediate_value'),
        
#         html.Footer(html.P("Smart Mobility and Infrastructure Lab", style={'color': 'white',"textAlign":"center"}))
    ],style={'backgroundColor': colors['background'],'color':'white','fontColor': 'white'})


@app.callback([Output("graph", "figure"),
               Output("graph-re", "figure"),
               Output("graph-we", "figure"),
               Output("Start_Date-output-1", "children"),
               Output('image1', 'figure'),
               Output('image2', 'figure'),
               Output('image3', 'figure'),
               Output('image4', 'figure'),
               Output('image5', 'figure'),
               Output('image6', 'figure'),
               Output('image7', 'figure'),
               Output('image8', 'figure'),
               Output('intermediate_value', 'data'),
               
              ],
              
              [Input("switch", "value"),
               Input("Start_Date", "value"), 
               Input('dd_station',"value"),
               Input('graph-update', 'n_intervals'),
               Input('intermediate_value', 'value')])

def get_vds_data(switch, Start_Date, dd_station, input_data, intermediate_value, GDOT_SQL=GDOT_SQL, drivers=drivers):
    
    data = {}
    station=dd_station
    if switch == False:
        print(Start_Date)
        
        weather_station = vds2weatherStation[station]
        print(weather_station)

        now_ = datetime.datetime.now()
        date, hr, min_, _ = now_.strftime("%Y-%m-%d %H %M %S").split()

        url, urlb4 = generate_url(station, Start_Date.strip(), SQL_URL=GDOT_SQL)
        
        # load urls
        get_weather_url(drivers[2], Start_Date, station=weather_station)
        
#         drivers[0].get(urlb4)
#         drivers[1].get(url) # get location

#         # get vds data
#         dfs = {}
#         for x in range(2):
#             banner =  interact(drivers[x], GDOT_SQL['banner'],click=False,delay=0.02,count=350,status_rate=300) 
#             dfs[x] = get_table(drivers[x]) # scrape data from driver 

        weather_data = get_weather_data(drivers[2])

        # preprocessing two days of data for predicting
#         flat_arr = np.array(dfs[0]['TOT VOL'].tolist() + dfs[1]['TOT VOL'].tolist())


        """
        get ccs/vds offline data here
        """
        df_offline_data = pd.read_csv(f'data/historic_ccs_vds_data/{station}.csv')
        
        date_1 = datetime.datetime.strptime(Start_Date, "%Y-%m-%d")
        dayb4 = date_1 + datetime.timedelta(days=-1)
        dayb4 = dayb4.strftime("%Y-%m-%d")
        df_offline_data = df_offline_data[df_offline_data['date'].isin([dayb4,Start_Date])]
        flat_arr = df_offline_data['vds_vol'].to_numpy()
        
        df_offline_data_sub = df_offline_data[df_offline_data['date']==Start_Date]
        time_list = list(df_offline_data_sub['time'])
        true_vds = df_offline_data_sub['vds_vol']
        
        """
        Predict data
        """
        pred = predict_vds(flat_arr, LTC_model, LTC=True)
        pred_LSTM = predict_vds(flat_arr, LSTM_model, LTC=False)

        """
        Get images 
        """
        X_transformed_vds = get_im_transf(true_vds, pipeline_image)
        X_transformed = get_im_transf(df_offline_data_sub['ccs_vol'], pipeline_image)

        """
        Recurrence plots
        """
        reshape_size_recur = 32 # move later
        n_scales = 64
        reshape_size = 64
        interval_len = 288
        
        recur_data = collect_recurrence_data(X_transformed, reshape_size_recur)
        recur_data_re = recur_data.reshape(-1, reshape_size_recur, reshape_size_recur, 1) # ccs data
        recur_data_vds = collect_recurrence_data(X_transformed_vds, reshape_size_recur)
        recur_data_re_vds = recur_data_vds.reshape(-1, reshape_size_recur, reshape_size_recur, 1) # vds data
        
        Recur_Zenc_vds = encoder_model_recur.predict(recur_data_re_vds)[0]
        pred_recur_vds = decoder_model_recur.predict(np.array([Recur_Zenc_vds[0].tolist()])).reshape(reshape_size_recur, -1)
        Recur_Zenc = encoder_model_recur.predict(recur_data_re)[0]
        pred_recur = decoder_model_recur.predict(np.array([Recur_Zenc[0].tolist()])).reshape(reshape_size_recur, -1)
        
        """
        wavelet plots
        """
        cwt_data = collect_cwt_data(X_transformed, reshape_size, n_scales, interval_len=interval_len) # make wavelet images
        cwt_data_re = cwt_data.reshape(-1, reshape_size, reshape_size, 1) # format for CNN
        cwt_data_vds = collect_cwt_data(X_transformed_vds, reshape_size, n_scales, interval_len=interval_len)# param same as CCS
        cwt_data_re_vds = cwt_data_vds.reshape(-1, reshape_size, reshape_size, 1)
        
        Zenc_ccs = encoder_model.predict(cwt_data_re)[0]
        pred_wavelet = decoder_model.predict(np.array([Zenc_ccs[0].tolist()])).reshape(reshape_size, -1)
        Zenc_vds = encoder_model.predict(cwt_data_re_vds)[0]
        pred_wavelet_vds = decoder_model.predict(np.array([Zenc_vds[0].tolist()])).reshape(reshape_size, -1)
        
        """
        get score
        """
        double_vds = np.concatenate((Zenc_vds, Recur_Zenc_vds), axis=1)[0]
        double_ccs = np.concatenate((Zenc_ccs, Recur_Zenc), axis=1)[0]
        
        d_vds, d_ccs, d3 = getNearestNeighborV2(space, double_vds, double_ccs, zeros=0)
        
        
        
        """
        format weather
        """
        simple_weather = [format_conditions(x) for x in weather_data] # simplify cats
        val_weather = [group2val[x] for x in simple_weather] # cat to weather
        weather_data = [f'{time} {weather}' for weather, time in zip(weather_data,time_list)]

        f, indic = [], 0 # frequency and indicator
        for idx, val in enumerate(val_weather):
            if val>indic:
                indic=val
            f = f + [idx]*(1+val)
            

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_list, y=(list(df_offline_data['ccs_vol'].astype(float))),
                    mode='lines+markers',
                    name='CCS'))
        fig.add_trace(go.Scatter(x=time_list, y=(list(true_vds.astype(float))),
                        mode='lines+markers',
                        name='VDS'))
        fig.add_trace(go.Scatter(x=time_list, y=(list(pred.astype(float))),
                        mode='lines+markers',
                        name='VDS LTC      '))
        fig.add_trace(go.Scatter(x=time_list, y=(list(pred_LSTM.astype(float))),
                        mode='lines+markers',
                        name='VDS LSTM'))
        fig.update_layout(title=f"VDS {vds2stationName[station]} Volume Data: {Start_Date}",
                          xaxis_title="Time",
                          yaxis_title="Total Volumn",
                          height=500,
                          #legend_title="",
                          font=dict(
                                family="Courier New, monospace",
                                size=16,
                                color="white"),
                          plot_bgcolor = colors['background'],
                          paper_bgcolor = colors['background'],)



        diff = true_vds.astype(float) - pred.astype(float)
        list_diff = list(diff)
        
        anoms = np.where(abs(diff)>60)[0]
        amoms_vals = [list_diff[x] for x in anoms]
        anoms_times = [time_list[x] for x in anoms]

        
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(x=time_list, y=(list(diff)),
                    mode='lines+markers',name=''))
        fig_resid.add_trace(go.Scatter(x=time_list, y=(list(diff)),
                    mode='lines+markers',name='Residuals    ',))
        
        # anomalies points
        fig_resid.add_trace(go.Scatter(x=(anoms_times), y=(amoms_vals),
                    mode='markers', name='Anomalies',
                    marker=dict(color='yellow',size=16,
                                line=dict(color='black',width=2))))

        fig_resid.update_layout(title=f"VDS Relative Error",
                      xaxis_title="Time",
                      yaxis_title="Total Volumn",
                      height=500,
                      legend_title="       ",
                      font=dict(
                            family="Courier New, monospace",
                            size=16,
                            color="white"),
                      plot_bgcolor = colors['background'],
                      paper_bgcolor = colors['background'],)
        fig_resid.update_yaxes(range=[(-np.max(true_vds.astype(float))), (np.max(true_vds.astype(float)))])
    
    

        group_labels = '             '
        color_ops = {2:'red',1:'orange',0:'blue'}
        color_ = color_ops[indic]
        fig_weather = go.Figure(data=[
        go.Bar(name='             ', x=weather_data, y=val_weather, marker_color=color_),
        go.Bar(name='', x=weather_data, y=[0 for x in range(288)])])
        # Change the bar mode
        fig_weather.update_layout(title_text=f'Potential Impact of Weather on VDS Operation: {weather_station}',
                          yaxis = dict(
                                tickmode = 'array',
                                tickvals = [0,1,2],
                                ticktext = ['Low ', 'High ', 'Severe ']),
                                xaxis_title="Time",
    #                             yaxis_title="Total Volumn",
                                height=500,
                                font=dict(
                                    family="Courier New, monospace",
                                    size=16,
                                    color="white"),
                                plot_bgcolor = colors['background'],
                                paper_bgcolor = colors['background'],)
        fig_weather.update_yaxes(range=[0, 2.1])
        
        
        image1 = create_plot(recur_data[0], f'Recurrence CCS:', full_label='True Plots')
        image2 = create_plot(recur_data_vds[0], f'Recurrence VDS:')
        image3 = create_plot(cwt_data[0], f'Wavelet CCS:')
        image4 = create_plot(cwt_data_vds[0], f'Wavelet VDS:')  
  
        #d_vds, d_ccs, d3
        binary_scores = [0,0,0]
        result, score = '', 0
        for idx, (thresh, val) in enumerate(zip([4.4, 4.4, 3.90],[d_vds, d_ccs, d3])):
            if val > thresh:
                binary_scores[idx] = 1
    
#         if (binary_scores[0] == 1) or (binary_scores[1] == 1) or (binary_scores[2] == 1):
        if (binary_scores[0] == 1) or (binary_scores[1] == 1):
            result, score = 'Anomalous Data', 10
            if (binary_scores[2] == 0): #check false positives
                result, score = 'Organic Anomaly', 0


        image5 = create_plot(pred_recur, f'Predicted Recurrence CCS:', 
                             f'Score VDS: {round(d_vds,2)}',
                             score=score)
        image6 = create_plot(pred_recur_vds, f'Predicted Recurrence VDS:',
                             f'Score CCS: {round(d_ccs,2)}',
                            score=score)
        image7 = create_plot(pred_wavelet, f'Predicted Wavelet CCS:',
                             f'Score Cross: {round(d3,2)}',
                             score=score)
        image8 = create_plot(pred_wavelet_vds, f'Predicted Wavelet VDS:',
                            f'{result}',
                             score=score)
        
        
        
        return [fig, fig_resid, fig_weather, 'loaded', 
                image1, image2, image3, image4,
                image5, image6, image7, image8,
                data,]
        
        
    
#     pred, y_true, pred_LSTM, times, cut, old_hr, old_min = get_live_full(drivers[0])

    Start_Date = '2020-04-04'

    """
    get ccs/vds offline data here
    """
    df_offline_data = pd.read_csv(f'data/historic_ccs_vds_data/{station}.csv')

    date_1 = datetime.datetime.strptime(Start_Date, "%Y-%m-%d")
    dayb4 = date_1 + datetime.timedelta(days=-1)
    dayb4 = dayb4.strftime("%Y-%m-%d")
    df_offline_data = df_offline_data[df_offline_data['date'].isin([dayb4,Start_Date])]
    flat_arr = df_offline_data['vds_vol'].to_numpy()

    df_offline_data_sub = df_offline_data[df_offline_data['date']==Start_Date]
    time_list = list(df_offline_data_sub['time'])[:input_data]
    true_vds = df_offline_data_sub['vds_vol']
    
    """
    Predict data
    """
    pred = predict_vds(flat_arr, LTC_model, LTC=True)
    pred_LSTM = predict_vds(flat_arr, LSTM_model, LTC=False)
    
        
    fig = go.Figure()
#     fig.add_trace(go.Scatter(x=time_list, y=(list(df_offline_data['ccs_vol'].astype(float))[:input_data]),
#                 mode='lines+markers',
#                 name='CCS'))
    fig.add_trace(go.Scatter(x=time_list, y=(list(true_vds.astype(float))[:input_data]),
                    mode='lines+markers',
                    name='VDS'))
    fig.add_trace(go.Scatter(x=time_list, y=(list(pred.astype(float))[:input_data]),
                    mode='lines+markers',
                    name='VDS LTC      '))
    fig.add_trace(go.Scatter(x=time_list, y=(list(pred_LSTM.astype(float))[:input_data]),
                    mode='lines+markers',
                    name='VDS LSTM'))
    fig.update_layout(title=f"VDS {vds2stationName[station]} Volume Data: 2021-05-07",
                      xaxis_title="Time",
                      yaxis_title="Total Volumn",
                      height=500,
                      #legend_title="",
                      font=dict(
                            family="Courier New, monospace",
                            size=16,
                            color="white"),
                      plot_bgcolor = colors['background'],
                      paper_bgcolor = colors['background'],)

    
    fig_weather = go.Figure()
    fig_weather.update_layout(plot_bgcolor = colors['background'],paper_bgcolor = colors['background'],)
    fig_resid = go.Figure()
    fig_resid.update_layout(plot_bgcolor = colors['background'],paper_bgcolor = colors['background'],)
    
    
    image = fig_resid

    
    return [fig, fig_resid, fig_weather, 'loaded', 
            image, image, image, image,
            image, image, image, image,
            data,]


app.run_server(debug=False)


