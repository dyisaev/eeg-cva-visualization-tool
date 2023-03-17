#!/media/st4Tb/anaconda3/envs/di_plotly/bin/python
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_player as player
from dash.dependencies import Input, Output, State,ClientsideFunction
from dash.exceptions import PreventUpdate
from dash import callback_context
from flask_caching import Cache
from config import *
from backend.data import Data
from backend.utils import get_video_duration, get_video_fps
from backend.model import Model
import numpy as np
import copy
import pyarrow as pa
import pandas as pd
import os
import cv2
import plotly.express as px
from scipy.signal import medfilt
import random
import string
import json
from datetime import datetime


#model for machine learning prediction/retrainings  
global model
model=None

#openCV videocapture for active learning
global cap
cap=None

import sys
sys.path.append('models/')

context = pa.default_serialization_context()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,url_base_pathname='/',prevent_initial_callbacks=True)

cache=Cache()
cache.init_app(app.server,config=CACHE_CONFIG)

from layout import generate_layout
events=[]
with open("events.lst", "r") as events_file:
    events = events_file.readlines()
    events = [event.strip() for event in events]
app.layout,signal_figure_stub,trace_ids=generate_layout(events)

app.clientside_callback(
    ClientsideFunction("clientside", "figure"),
    Output(component_id="behav-graph", component_property="figure"),
    [Input("fig-data", "data"), 
    Input("video-display", "currentTime"),
    Input('pred-threshold-slider','value'),
    Input('pred-rejected-events','data')]
)
app.clientside_callback(
    ClientsideFunction("clientside","process_rejection"),
    Output(component_id='pred-rejected-events',component_property='data'),
    [Input('pred-reject','n_clicks'),
    Input('pred-reset-rejections','n_clicks')],
    [State("fig-data", "data"),
    State("video-display", "currentTime"),
    State('pred-rejected-events','data'),
    State('video-fps', 'data')]
)

@app.callback(
    dash.dependencies.Output('video-fps', 'data'),
    [dash.dependencies.Input('datafolder-dropdown', 'value')])
def update_video_fps(value):
    # if video is new - reset videocapture
    global cap
    if not cap is None:
        cap.release()
        cap=None

    url = value+'/video/'+VIDEOFILE_NAME
    return float(get_video_fps(url))

@app.callback(
    dash.dependencies.Output('video-duration', 'data'),
    [dash.dependencies.Input('datafolder-dropdown', 'value')])
def update_video_duration(value):
    url = value+'/video/'+VIDEOFILE_NAME
    return float(get_video_duration(url))

@app.callback(
    dash.dependencies.Output('video-display', 'url'),
    [dash.dependencies.Input('datafolder-dropdown', 'value')])
def update_video_url(value):
    url = value+'/video/'+VIDEOFILE_NAME
    return url

@app.callback(
    [dash.dependencies.Output('al-frame','src'),
    dash.dependencies.Output('al-frames-left-to-label-text','children'),
    dash.dependencies.Output('al-frames-to-label','data'),
    dash.dependencies.Output('al-labeled-frames','data'),
    dash.dependencies.Output('al-current-frame','data'),
    dash.dependencies.Output("video-display", "seekTo")],
    [dash.dependencies.Input('al-start-labeling','n_clicks'),
    dash.dependencies.Input('al-return-to-labeling','n_clicks'),
    dash.dependencies.Input('al-att','n_clicks'),
    dash.dependencies.Input('al-inatt','n_clicks'),
    dash.dependencies.Input('behav-graph', 'clickData')],
    [dash.dependencies.State('al-Nbatch','value'),
    dash.dependencies.State('video-display', 'url'),
    dash.dependencies.State('video-fps','data'),
    dash.dependencies.State('al-frames-left-to-label-text','children'),
    dash.dependencies.State('al-frames-to-label','data'),
    dash.dependencies.State('al-labeled-frames','data'),
    dash.dependencies.State('al-current-frame','data'),
    dash.dependencies.State('al-frame','src')]
)
def active_learning_update(start_labeling,return_to_labeling,al_att,al_inatt,clickData,al_Nbatch,video_url,fps,
                            al_frames_text_state,al_frames_to_label_state,al_labeled_frames_state,al_current_frame_state,current_frame_src):
    global model
    if 'behav-graph.clickData' in callback_context.triggered[0].values():
        if not clickData is None:
            return '',al_frames_text_state,al_frames_to_label_state,al_labeled_frames_state,al_current_frame_state,clickData['points'][0]['x']/fps
        else:
            return '',al_frames_text_state,al_frames_to_label_state,al_labeled_frames_state,al_current_frame_state,None

    elif 'al-start-labeling.n_clicks' in callback_context.triggered[0].values():
        # generate sample.
        # save sample into store
        # output first frame
        if model is None:
            print('model is None, cannot generate sample')
            return
        else:
            frameNums_to_label=model.generate_al_batch(al_Nbatch)
            filenames_to_label=['assets/labeling/'+''.join(random.choices(string.ascii_letters, k=10))+f'_{frameNum}.jpg' for frameNum in frameNums_to_label]
            frames_to_label=list(zip(frameNums_to_label,filenames_to_label))
            for frameNum,filename in frames_to_label:
                get_frame_from_video(video_url,frameNum,filename)
            return frames_to_label[0][1],'labeled 0 frames',frames_to_label,[],frames_to_label[0],frames_to_label[0][0]/fps
    elif 'al-att.n_clicks' in callback_context.triggered[0].values():
        label=1
    elif 'al-inatt.n_clicks' in callback_context.triggered[0].values():
        label=0
    elif 'al-return-to-labeling.n_clicks' in callback_context.triggered[0].values():
        label=-1
    else:
        return
    if al_current_frame_state is None:
        al_frames_text_state=f'No frames left to label'
    if label>=0: # removing frame from array only if att/inatt clicked, not 'return to labeling'.
        al_labeled_frames_state.append((al_current_frame_state[0],label))
        al_frames_to_label_state=[frame_to_label for frame_to_label in al_frames_to_label_state if frame_to_label[0]!=al_current_frame_state[0]]    
        # erase the frame image
        if os.path.isfile(current_frame_src):
            os.remove(current_frame_src)
    al_frames_text_state=f'labeled {len(al_labeled_frames_state)} out of {al_Nbatch} frames'

    if len(al_frames_to_label_state)==0:
       
        
        return '',al_frames_text_state,al_frames_to_label_state,al_labeled_frames_state,al_current_frame_state,al_current_frame_state[0]/fps        
    
    al_current_frame_state=al_frames_to_label_state[0]
    return al_current_frame_state[1],al_frames_text_state,al_frames_to_label_state,al_labeled_frames_state,al_current_frame_state,al_current_frame_state[0]/fps
@app.callback(
    dash.dependencies.Output('pred-threshsold-value','value'),
    [dash.dependencies.Input('pred-threshold-slider','value')]
)
def set_threshold_input_value(value):
    return value

@app.callback(
    dash.dependencies.Output('pred-threshold-slider','value'),
    [dash.dependencies.Input('pred-threshsold-set','n_clicks')],
    [dash.dependencies.State('pred-threshsold-value','value')]
)
def set_threshold_slider_value(button_click,value):
    return float(value)
@app.callback(
    dash.dependencies.Output('pred-issaved','children'),
    [dash.dependencies.Input('pred-save','n_clicks')],
    [dash.dependencies.State('datafolder-dropdown','value'),
     dash.dependencies.State('behav-graph', 'figure')]
)
def save_detections(pred_save_clicked,datafolder_value,fig):
    frames=fig['data'][trace_ids['label']]['x']
    labels=fig['data'][trace_ids['label']]['y']
    det_dict={'frame':frames,'label':labels}
    fname=os.path.basename(datafolder_value)+'.json'
    json.dump(det_dict,open('assets/saved_detections/'+fname,'w'))
    return f'Detections saved at {datetime.now()}'

@app.callback(
    [dash.dependencies.Output('fig-data', 'data'),
    dash.dependencies.Output('model-loading-result', 'children')],
    [dash.dependencies.Input('model-load', 'n_clicks'),
    dash.dependencies.Input('datafolder-dropdown', 'value'),
    dash.dependencies.Input('al-retrain','n_clicks')],
    [dash.dependencies.State('model-filename', 'value'),
     dash.dependencies.State('fig-data', 'data'),
     dash.dependencies.State('al-labeled-frames', 'data'),
     dash.dependencies.State("video-display", "currentTime")]
)
def load_model (n_clicks,datafolder_value,al_retrain,
        model_filename,current_fig_data,al_labeled_frames,currentTime):
    global model
    subj_path=APP_PATH+'/'+datafolder_value
    fps=float(get_video_fps(subj_path+'/video/'+VIDEOFILE_NAME))
    video_duration=float(get_video_duration(subj_path+'/video/'+VIDEOFILE_NAME))
    output=''
    if 'model-load.n_clicks' in callback_context.triggered[0].values():
        model=Model(model_filename)
        if not (model is None):
            output='Model loaded'
        else:
            output = 'Model cannot be loaded'
    elif 'al-retrain.n_clicks' in callback_context.triggered[0].values():
        if not (model is None):
            model.label(al_labeled_frames)
            model.train()
    graph_fig=pa.deserialize(create_traces_annot(subj_path,fps,video_duration,model=model))
    if graph_fig is None or len(graph_fig)==0 or graph_fig['data'] is None:
        raise PreventUpdate
    return graph_fig , output 

@cache.memoize()
def create_traces_annot(subj_path,fps,video_duration,model=None,threshold=0.5,reject_events=[]):
    graph_data = pa.deserialize(load_data(subj_path,fps,video_duration))
    
    gaze=graph_data['gaze']
    nose=graph_data['nose']
    headpose=graph_data['headpose']

    Nframes=int(video_duration*fps)+1
    frames_axis=np.arange(Nframes)

    gazex=np.empty(Nframes)
    gazey=np.empty(Nframes)
    gazex[:]=np.NaN
    gazey[:]=np.NaN
    gazex[gaze['frame']]=np.array(gaze['gazex'])
    gazey[gaze['frame']] = np.array(gaze['gazey'])

    yaw=np.empty(Nframes)
    pitch=np.empty(Nframes)
    roll=np.empty(Nframes)
    yaw[:]=np.NaN
    pitch[:]=np.NaN
    roll[:]=np.NaN
    yaw[headpose['frame']]= np.array([x[1] for x in headpose['headpose']])
    pitch[headpose['frame']]= np.array([x[2] for x in headpose['headpose']])
    roll[headpose['frame']]= np.array([x[0] for x in headpose['headpose']])


    nose_x=np.empty(Nframes)
    nose_y=np.empty(Nframes)
    nose_x[:]=np.NaN
    nose_y[:]=np.NaN
    nose_x[nose['frame']]=np.array(nose['nosex'])
    nose_y[nose['frame']]=np.array(nose['nosey'])

    fig=copy.deepcopy(signal_figure_stub)
    fig['layout']['shapes']=list()
    fig['layout']['xaxis']['range']=[0,Nframes]
    fig['layout']['xaxis']['uirevision']=1


    fig['data'][trace_ids['gazex']]['x']=frames_axis
    fig['data'][trace_ids['gazex']]['y']=gazex
    fig['data'][trace_ids['gazex']]['name']='Gaze X'
    fig['data'][trace_ids['gazey']]['x']=frames_axis
    fig['data'][trace_ids['gazey']]['y']=gazey
    fig['data'][trace_ids['gazey']]['name']='Gaze Y'

    fig['data'][trace_ids['pitch']]['x']=frames_axis
    fig['data'][trace_ids['pitch']]['y']=pitch
    fig['data'][trace_ids['pitch']]['name']='Pitch'
    
    fig['data'][trace_ids['yaw']]['x']=frames_axis
    fig['data'][trace_ids['yaw']]['y']=yaw
    fig['data'][trace_ids['yaw']]['name']='Yaw'
    
    fig['data'][trace_ids['roll']]['x']=frames_axis
    fig['data'][trace_ids['roll']]['y']=roll
    fig['data'][trace_ids['roll']]['name']='Roll'


    fig['data'][trace_ids['nosex']]['x']=frames_axis
    fig['data'][trace_ids['nosex']]['y']=nose_x
    fig['data'][trace_ids['nosex']]['name']='Nose X'
    
    fig['data'][trace_ids['nosey']]['x']=frames_axis
    fig['data'][trace_ids['nosey']]['y']=nose_y
    fig['data'][trace_ids['nosey']]['name']='Nose Y'

    #vertical line representing where we are on the video    
    fig['layout']['shapes']=[]
    fig['layout']['shapes']=fig['layout']['shapes']+[{  # player line
            'type': 'line',
            # x-reference is assigned to the x-values
            'xref': 'x',
            # y-reference is assigned to the plot paper [0,1]
            'yref': 'paper',
            'x0': 0,
            'y0': 0,
            'x1': 0,
            'y1': 1,
            'line': {'width':1,'color':'rgb(0,0,255)'},
            'layer': 'above',
            'opacity': 0.8
        }]

    #process events
    event_timeseries_lst=[]
    if len(graph_data['events'])>0:
        for event in graph_data['events'].keys():
            event_timeseries=np.empty(Nframes)
            event_timeseries[:]=np.NaN
            event_timeseries[graph_data['events'][event]['frame']]=np.array(graph_data['events'][event][event])
            event_timeseries_lst.append(event_timeseries)

    for event,event_ts in zip(graph_data['events'].keys(),event_timeseries_lst):
        fig['data'][trace_ids[event]]['x']=frames_axis
        fig['data'][trace_ids[event]]['y']=event_ts
        fig['data'][trace_ids[event]]['name']=event

    if not model is None:
        dataset = pd.DataFrame([elem for elem in zip(frames_axis,gazex,gazey,yaw,pitch,roll,nose_x,nose_y)],
                            columns=['frame','gazex','gazey','yaw','pitch','roll','nosex','nosey'])
        model.load_dataset(dataset)
        frame_pred,pred=model.predict()

        pred_timeseries=np.empty(Nframes)
        pred_timeseries[:]=np.NaN
        pred_timeseries[frame_pred]=pred
        pred_timeseries=medfilt(pred_timeseries,MEDIAN_FILTERING_LENGTH)
        label_timeseries=(pred_timeseries>threshold).astype(int) #edit this
        

        fig['data'][trace_ids['ml']]['x']=frames_axis
        fig['data'][trace_ids['ml']]['y']=pred_timeseries
        fig['data'][trace_ids['ml']]['name']='predictions'

        fig['data'][trace_ids['label']]['x']=frames_axis
        fig['data'][trace_ids['label']]['y']=label_timeseries
        fig['data'][trace_ids['label']]['name']='labels'
    
    fig['layout']['uirevision']=1

    return pa.serialize(fig).to_buffer()

def get_frame_from_video(videofile,frameNum,filename):
    global cap
    if cap is None:
        cap = cv2.VideoCapture(videofile)
    print('reading frame')
    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # check for valid frame number
    if frameNum >= 0 & frameNum <= totalFrames:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES,frameNum)
    ret, frame = cap.read()
     
    scale_percent = 400/frame.shape[1] # percent of original size
    width = int(frame.shape[1] * scale_percent)
    height = int(frame.shape[0] * scale_percent)
    dim = (width, height)
  
    # resize image
    resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(filename,resized_frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return

@cache.memoize()
def load_data(subj_path,fps,video_duration):
    timeseries = Data(subj_path,events)

    return timeseries.serialize()



if __name__ == '__main__':
    app.run_server(debug=True,dev_tools_hot_reload = False,threaded=True, processes=1, port=20101)
