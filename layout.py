import dash_core_components as dcc
import dash_html_components as html
import dash_player as player
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import os


def generate_signals_figure(events):
    N_SUBPLOTS=int(len(events)>0)+5
    common_signal_fig=make_subplots(rows=N_SUBPLOTS,cols=1,shared_xaxes=True,vertical_spacing=0.02)
    cfdict=common_signal_fig.to_dict()
    cfdict_xaxis=cfdict['layout'][f'xaxis{N_SUBPLOTS}']
    cfdict_xaxis['rangeslider']=dict(visible=True)
    common_signal_fig.update_layout(xaxis6=cfdict_xaxis)
    trace_ids={}
    #--subplot-1--
    #gazex
    common_signal_fig.append_trace(go.Scatter(mode='lines',x=np.arange(1000),y=np.zeros(1000)),row=1,col=1)
    trace_ids['gazex']=0
    #gazey
    common_signal_fig.append_trace(go.Scatter(mode='lines',x=np.arange(1000),y=np.zeros(1000)),row=1,col=1)
    trace_ids['gazey']=1
    #--subplot-2-
    #pitch
    common_signal_fig.append_trace(go.Scatter(mode='lines',x=np.arange(1000),y=np.zeros(1000)),row=2,col=1)
    trace_ids['pitch']=2
    #yaw
    common_signal_fig.append_trace(go.Scatter(mode='lines',x=np.arange(1000),y=np.zeros(1000)),row=2,col=1)
    trace_ids['yaw']=3
    #roll
    common_signal_fig.append_trace(go.Scatter(mode='lines',x=np.arange(1000),y=np.zeros(1000)),row=2,col=1)
    trace_ids['roll']=4
    #nosex
    common_signal_fig.append_trace(go.Scatter(mode='lines',x=np.arange(1000),y=np.zeros(1000)),row=3,col=1)
    trace_ids['nosex']=5
    #nosey
    common_signal_fig.append_trace(go.Scatter(mode='lines',x=np.arange(1000),y=np.zeros(1000)),row=3,col=1)
    trace_ids['nosey']=6

    #ML predictions
    common_signal_fig.append_trace(go.Scatter(mode='lines',x=np.arange(1000),y=np.zeros(1000)),row=4,col=1)
    trace_ids['ml']=7

    #labels
    common_signal_fig.append_trace(go.Scatter(mode='lines',x=np.arange(1000),y=np.zeros(1000)),row=5,col=1)
    trace_ids['label']=8
    for i,event in enumerate(events):
        common_signal_fig.append_trace(go.Scatter(mode='lines',x=np.arange(1000),y=np.zeros(1000)),row=6,col=1)
        trace_ids[event]=8+i+1
    common_signal_fig.update_layout(height=700)
    common_signal_fig=common_signal_fig.to_dict()
    common_signal_fig['layout']['uirevision']=1

    return common_signal_fig,trace_ids


def generate_layout(events):
    common_signal_fig,trace_ids=generate_signals_figure(events)

    layout = html.Div(children=[
        html.H1(children='Video Visualization'),

        html.Div(
            className='main_container',
            children=[
                html.Div(
                    id='left-side-column',
                    className='eight columns',
                    style={'display': 'flex',
                    'flexDirection': 'column',
                    'flex': 1,
                    'height': 'calc(100vh - 5px)',
                    'backgroundColor': '#F2F2F2',
                    'overflow-y': 'scroll',
                    'marginLeft': '0px',
                    'justifyContent': 'flex-start',
                    'alignItems': 'center'
                        }
                        ,
                        
                    children= [
                        html.Div(
                            style={
                                'width':'100%' 
                            },
                                children=[dcc.Graph(id='behav-graph',
                                                    style={
                                                        'width':'100%','marginBottom': '0px',
                                                    }
                                        ),
                                        dcc.Store("fig-data", data=common_signal_fig)]
                        )
                    
                    ]
                ),
                html.Div(
                    id='right-side-column',
                    className='four columns',
                    style={
                        'height': 'calc(100vh - 5px)',
                        'overflow-y': 'scroll',
                        'marginLeft': '1%',
                        'display': 'flex',
                        'backgroundColor': '#F9F9F9',
                        'flexDirection': 'column'
                        
                    },
                    children=[                    
                        html.Div(
                            id='controls-box',
                            children=[
                                html.Div(
                                        children=[player.DashPlayer(
                                            id='video-display',
                                            style={'position': 'relative', 'height':'200px',
                                                'top': '0', 'left': '0', 'bottom': '0', 'right': '0'},
                                            url='',
                                            controls=True,
                                            playing=False,
                                            volume=1,
                                            width='100%',
                                            intervalCurrentTime=500
                #                            height='100%'
                                            ),
                                            dcc.Store("video-fps", data=30),
                                            dcc.Store("video-duration", data=10*60)]

                                    ),
                                html.Div(children='Data Folder:'),
                                dcc.Dropdown(
                                    id='datafolder-dropdown',
                                    options=[  {'label': f.path , 'value': f.path}   for f in os.scandir('assets/data') if f.is_dir()],
                                    #[  {'label': f.path , 'value': f.path}   for f in os.scandir('assets/ACEEEGATT_Link') if f.is_dir()],
                                    value=''
                                ),                                
                                html.Div(children=[
                                    html.Hr(),
                                    html.Div([
                                        'Model filename: ',
                                        dcc.Input(id='model-filename',type='text',placeholder='',style={'width':'300px'}),
                                        html.Button('Load model', id='model-load',style={'margin-left':'20px'}),
                                        html.Div(id='model-loading-result',children='')

                                    ]),
                                    html.Br(),
                                    html.Div([
                                        'Retrained model filename (for saving): ',                                     
                                        dcc.Input(id='model-retrained,filename',type='text',placeholder='',style={'width':'300px'})
                                    ]),
                                    html.Button('Save retrained model', id='model-save',style={'margin-top':'20px',})
                                ]),
                                html.Div(id='active-learning-div',children=[
                                    html.Hr(),
                                    'Label additional ',
                                    dcc.Input(id="al-Nbatch", type="number", value=128, style={'width':'130px', 'margin-left':'10px','marginRight':'10px','marginBottom':'10px'}),
                                    'frames.',
                                    html.Br(),

                                    
                                    html.Button('Start labeling',id='al-start-labeling'),
                                    html.Button('Return to labeling',id='al-return-to-labeling'),
                                    html.Button('Retrain',id='al-retrain',style={'float':'right'}),
                                    html.Img(id='al-frame',width='100%'),
                                    html.Button('Attention',id='al-att'),
                                    html.Button('Inattention',id='al-inatt',style={'float':'right'}),
                                    html.Div(id='al-frames-left-to-label-text',children=''),
                                    dcc.Store(id='al-frames-to-label',data=[]),
                                    dcc.Store(id='al-labeled-frames',data={}),
                                    dcc.Store(id='al-current-frame',data={})
                                    
                                    ]
                                ),
                                html.Div(id='edit-predictions-div',children=[
                                    html.Hr(),
                                    'Threshold:', 
                                    dcc.Input(id='pred-threshsold-value',value=0.5,style={'width':'250px','margin-left':'15px'}),
                                    html.Button('Set threshold',id='pred-threshsold-set'),
                                    dcc.Slider(min=0, max=1, step=0.001,
                                            value=0.5,
                                            id='pred-threshold-slider'
                                    ),
                                    html.Button('Reject event',id='pred-reject'),
                                    html.Button('Reset rejections',id='pred-reset-rejections',style={'float':'right'}),
                                    html.Br(),
                                    html.Button('Save detections', id='pred-save',style={'margin-top':'20px'}),
                                    html.Div('Detections not saved yet',id='pred-issaved'),
                                    dcc.Store(id='pred-rejected-events',data=[])
                                ]),
                                dcc.Interval(
                                        id='interval-component',
                                        interval=int(0.5*1000), # in milliseconds
                                        n_intervals=0,
                                        disabled=True
                                )

                            ]
                        )
                    ]
                )
            ]
        ),
        html.Div(children='',id='current_time')
    ])
    return layout,common_signal_fig, trace_ids