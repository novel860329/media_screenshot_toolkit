# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:46:32 2022

@author: MUILab-VR
"""

import dash
import dash_auth
from dash import callback_context, Dash, dcc, html, Input, Output, callback, dash_table
from dash.dependencies import State
import plotly.graph_objs as go
import pandas as pd
import flask
import datetime
from collections import defaultdict
from textwrap import dedent as s
from PIL import Image
import os
import cv2
import base64
import json
from CreateImage import draw_visible_area
from FigureUpdate import CombineFigUpdate
from FigureUpdate import DeleteFigUpdate
from FigureUpdate import SplitFigUpdate
from FigureUpdate import DiscussFigUpdate
from FigureUpdate import EventFigUpdate
from DB_operation import GetDataframe
from DB_operation import GetHistory
from DB_operation import SaveDataframe
from DB_operation import SaveHistory

# from CoderDataPreprocessingforDev import data_preprocess

def extract_app_name(images):
    image = images.split("\n")[0]
    img = image.split("-")
    if "crop" in image:
        app = img[10][:-4]
    elif "ESM" in image:
        app = img[9][:-4]
    else:
        app = img[8][:-4]
    return app

def extract_time_from_answer(answer):  
#    print(answer)
    temp = answer.split("-")
    date = temp[1] + "-" + temp[2] + "-" + temp[3]
    time = temp[4] + ":" + temp[5] + ":" + temp[6]
    return date + " " + time

#----- config ------
# ubuntu path = "/home/ubuntu/News Consumption/"
# windows path = "D:/Users/MUILab-VR/Desktop/News Consumption/"
ROOT_PATH = "D:/Users/MUILab-VR/Desktop/News Consumption/CHI2022/media_screenshot_toolkit/"
data_path = ROOT_PATH + "Analysis/Visualization/data/"

# data_preprocess()

global_user = 0
event_list = ["news", "external link", 'comment', "external&news", "comment&news"]
source = {'post':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png"),
        'news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/news.png"),
        'external link':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/link.png"),
        'external&news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/link.png"),
        'comment':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png"),
        'comment&news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png")}
event_map = {0: 'external link', 1: 'comment', 2: 'news'}
scatter_layout = []
stacked_layout = []
prev_selection = {}
history_column = ["Time", "Correction step", "Row number", "Button click"]
description = '''
* **We use different background color to represent apps**
    1. Blue tone : Facebook
    2. Red tone : Youtube
    3. Purple : Instagram
* **Click screenshot can enlarge it and click again can reduce to original size**
* **Different mark on the bar represent different event and you can hover on it to see what evnet it stand for**
'''

User_ID = "user"
questionnaire_id = "qid"
image_appeared = "images"
image_content = "context"
visible_time = "visible time"
port_number = 8087
stacked_fig = ""
port_file = "port"+str(port_number) + "/" +"port_" + str(port_number)
Select_post_num = []
Select_picture_num = []
Select_row_index = []
width = 0
height = 0
image_width = 5
try:
   os.makedirs("port_" + str(port_number))
except FileExistsError:
   pass

User_data = os.listdir(data_path)
User_data = [filename.split(".")[0] for filename in User_data]
User_data = sorted(User_data, key = lambda user : (user.split("-")[0], int(user.split("-")[1])))

#-------------------

VALID_USERNAME_PASSWORD_PAIRS = {
    'NewsConsumption': 'test123'
}

static_image_route = '/static/'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

server = app.server
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

col_cate = {'app':["Facebook", "Instagram", "Youtube", "PTT", "Messenger", "LineChat","googleNews" ,"LineToday", "NewsApp" ,"Chrome","-1"], '新聞主題':['生活','運動', '娛樂', '政治', '健康','-1']}

for uid in User_data:
    try:
       os.makedirs("port_" + str(port_number) + "/" + uid)
    except FileExistsError:
       pass

Extra_column = ["app", "detect_time", "date", "discuss", "pid"]

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

coding_layout = html.Div(className="row", children=[
    html.Div([
        html.Div(
            html.Div([
                dcc.Dropdown(
                    id='users-dropdown',
                    options=[{'label': i, 'value': i} for i in User_data],
                    value=""
                )
            ])
            , style={'width': '10%', 'display': 'inline-block', 'vertical-align':'top','margin-left': '10px', 'margin-top': '10px'}
        ),

        html.Div([
            html.Button('Combine', id='Merge_button', n_clicks=0, style={"margin-left": "10px"}),
            html.Button('Delete', id='Delete_button', n_clicks=0, style={"margin-left": "10px"}),
            html.Button('Split', id='Split_button', n_clicks=0, style={"margin-left": "10px"}),
            html.Button('News', id='News_button', n_clicks=0, style={"margin-left": "10px"}),
        ], style={'display': 'inline-block', 'vertical-align':'top', "margin-left": "10px", 'margin-top': '10px'}),

        html.Details([
            html.Summary('Facebook events'),          
            html.Div([                               
                html.Button('Comment', id='Comment_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('External Link', id='Click_button', n_clicks=0, style={"margin-left": "10px"}),           
                html.Button('Like', id='Like_button', n_clicks=0, style={"margin-left": "10px"}),                 
                html.Button('Typing', id='Typing_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('Share', id='Share_button', n_clicks=0, style={"margin-left": "10px"}),     
            ]),            
        ], style={'display': 'inline-block', 'vertical-align':'top', "margin-left": "10px", 'margin-top': '5px'}),

        html.Details([
            html.Summary('Instagram events'),
            html.Div([
                html.Button('Comment', id='Comment2_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('External Link', id='Click2_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('Like', id='Like2_button', n_clicks=0, style={"margin-left": "10px"}),                 
                html.Button('Typing', id='Typing2_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('Share', id='Share2_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('Story', id='Story2_button', n_clicks=0, style={"margin-left": "10px"}),     
            ]),            
        ], style={'display': 'inline-block', 'vertical-align':'top', "margin-left": "10px", 'margin-top': '5px'}),

        html.Details([
            html.Summary('Youtube events'),
            html.Div([
                html.Button('Comment', id='Comment3_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('Watch', id='Watch3_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('Typing', id='Typing3_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('Click on video', id='Click3_button', n_clicks=0, style={"margin-left": "10px"}),     
            ]),            
        ], style={'display': 'inline-block', 'vertical-align':'top', "margin-left": "10px", 'margin-top': '5px'}),

        html.Div(
            html.Div([
                html.P("Discuss Reason：", style={'fontSize':18}),
                dcc.Input(
                    id='Discuss_text',
                    type='text',
                    style={'width':'100px'}
                ),
                html.Button('Record', id='Discuss_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('Previous Step', id='Recovery_button', n_clicks=0, style={"margin-left": "20px"}),
                dcc.Download( id="download_file"),
                html.Button('Output File', id='Output_file', n_clicks=0, style={"margin-left": "10px"}), 
                html.B(id="button_result", style={'fontSize':16, "margin-left": "10px"}),
            ], style={'display': 'flex', 'flex-direction': 'row','margin-left': '10px', 'margin-top': '10px'}),           
        ),
    ]),
    
    html.Div(
        dcc.Link('Go to coding mode', href='/coding')
    ,style={'display': 'none', 'vertical-align':'bottom','margin-left': '20px', 'margin-top': '10px'}),
    html.Div(
        dcc.Graph(
            id='stacked-bar',
            config={"displayModeBar": False}
            # style={'height': '70vh'}
        )
    ,style={ 'width': '100%','margin-top': '10px'}),       

    html.Div([
        html.Div(
            dash_table.DataTable(id='correct_history',
            style_cell={
                'overflow-y': 'hidden',
                'textOverflow': 'ellipsis',
                'fontSize':16,
                'minWidth': '50px', 'width': '50px', 'maxWidth': '50px',
                'textAlign': 'center'
            },
            style_table={'minWidth': '100%', 'width': '100%', 'maxWidth': '100%', 'maxHeight': '500px', 'overflowY': 'scroll'},
            style_data={
                'whiteSpace': 'normal'
            },
        )
        , style={'width': '50%', 'margin-left': '10px', 'display': 'inline-block'}),
        html.Div(dcc.Markdown(s(description)), style={'width': '50%', "height": "500px", 'margin-left': '20px', 'display': 'inline-block', 'fontSize': 20})]
    ,style={'display': 'flex', 'flex-direction': 'row',  "margin-bottom": "15px", "margin-top": "15px"}),  

    html.P(id='placeholder'),
    html.P(id='ph_for_select')
])

def nonvisual_bottombar(picture_number, non_visual_costomdata):
    bottom_bar = go.Bar(
            name="-1",
            y=[0.13],
            x=[int(picture_number)],
            marker=dict(color="rgba(0,0,0,0)"),
            offset=0,
            customdata=[non_visual_costomdata],
            hovertemplate="<br>".join([
                "Non Visible Area",
                "detect time=%{customdata[1]}",
                "questionnaire id=%{customdata[2]}",
                "row index=%{customdata[3]}", 
                "picture number=%{customdata[4]}",                       
           ])
        )
    return bottom_bar

def nonvisual_topbar(picture_number, non_visual_costomdata):
    top_bar = go.Bar(
            name="-1",
            y=[0.13],
            x=[int(picture_number)],
            marker=dict(color="rgba(0,0,0,0)"),
            offset=0,
            customdata=[non_visual_costomdata],
            hovertemplate="<br>".join([
               "Non Visible Area",
                "detect time=%{customdata[1]}",
                "questionnaire id=%{customdata[2]}",
                "row index=%{customdata[3]}", 
                "picture number=%{customdata[4]}"                
           ])
        )
    return top_bar

def draw_barchart(df, sliderrange, uid):
    global width, height
    if not df.empty:
        scatter_dataframe = df.drop(df.loc[df['visible time'] == -1].index, inplace=False)
        scatter_dataframe.reset_index(inplace = True, drop = True)
    pos_x = []

    fig = go.Figure()
    
    ### read image and display
    uid = uid.split("-")[0] if "-" in uid else uid
    temp_path = os.listdir(ROOT_PATH + "/" + uid)
    images = list(dict.fromkeys(scatter_dataframe['images'].tolist()))
    images.reverse()
    for i, image in enumerate(images):
        q_id = scatter_dataframe[scatter_dataframe['images'] == image].iloc[0]['qid']
        picture_number = scatter_dataframe[scatter_dataframe['images'] == image].iloc[0]['picture_number']       
        img = Image.open(ROOT_PATH + uid + "/" + str(q_id) + "/" + image)
        width, height = img.size
        y_size = height * image_width / width
        x_axis = picture_number + (int(picture_number) - 1) * image_width

        fig.add_layout_image(
            x=x_axis,
            y=y_size/2,
            source=img,
            xref="x",
            yref="y",
            sizex=image_width,
            sizey=y_size,
            xanchor="center",
            yanchor="middle",
            sizing="contain",
            layer="below",
            visible=True,
        )
        row_list = scatter_dataframe.loc[scatter_dataframe['images'] == image]['row_index'].tolist()
        row_list.reverse()

        for j, row_index in enumerate(row_list):
            code_id = scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['code_id']
            percent = scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['percent']
            picture_number = scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['picture_number']
            color = scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['color']
            detect_time = scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['detect_time']
            qid = scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['qid']
            comment = scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['comment']
            link = scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['outer_link']
            news = scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['news']

            event = "comment&news"
            shape_2 = int(str(news) + str(comment) + str(link), 2)

            if shape_2 == 2:
                event = "comment"
            elif shape_2 == 1:
                event = "external link"
            elif shape_2 == 4:
                event = "news"
            elif shape_2 == 6:
                event = "comment&news"
            elif shape_2 == 5:
                event = "external&news"
            elif shape_2 == 0:
                event = "post"
            elif shape_2 == 3:
                event = "external&comment"
            else:
                event = "all event"

            non_visual_costomdata = [code_id, detect_time, qid, row_index, picture_number]

            if j == 0:
                bottom_bar = go.Bar(
                    name="-1",
                    y=[y_size * 0.13],
                    x=[x_axis],
                    width=[image_width],
                    marker=dict(color="rgba(180,180,180,0.8)"),
                    customdata=[non_visual_costomdata],
                    hovertemplate="<br>".join([
                        "Non Visible Area",
                        "detect time=%{customdata[1]}",
                        "questionnaire id=%{customdata[2]}",
                        "row index=%{customdata[3]}", 
                        "picture number=%{customdata[4]}",                       
                    ])
                )
                fig.add_trace(bottom_bar)

            bar = go.Bar(
                name=str(code_id),
                y=[y_size * percent],
                x=[x_axis],
                marker=dict(color="rgba(0,0,0,0)",line=dict(
                    color=color,
                    width=3.5
                )),
                width=[image_width],
                customdata=[[code_id, detect_time, qid, row_index, picture_number, event, round(percent, 2)]],
                hovertemplate="<br>".join([
                    "post_number=%{customdata[0]}",
                    "detect time=%{customdata[1]}",
                    "questionnaire id=%{customdata[2]}",
                    "percent=%{customdata[6]}",
                    "row index=%{customdata[3]}", 
                    "picture number=%{customdata[4]}", 
                    "event=%{customdata[5]}",                           
                    ]),
                unselected=dict(marker=dict(opacity=0.5))
            )
            fig.add_trace(bar)

            if j == len(row_list) - 1:
                top_bar = go.Bar(
                    name="-1",
                    y=[y_size * 0.13],
                    x=[x_axis],
                    width=[image_width],
                    marker=dict(color="rgba(180,180,180,0.8)"),
                    customdata=[non_visual_costomdata],
                    hovertemplate="<br>".join([
                        "Non Visible Area",
                        "detect time=%{customdata[1]}",
                        "questionnaire id=%{customdata[2]}",
                        "row index=%{customdata[3]}", 
                        "picture number=%{customdata[4]}"                
                    ])
                )
                fig.add_trace(top_bar)

    x_dict = defaultdict(list)
    for j, row in scatter_dataframe.iterrows():
        shape_2 = int(str(row['news']) + str(row['comment']) + str(row['outer_link']), 2)
        # row_index = row['row_index']
        picture_number = row['picture_number']
        x_dict[int(picture_number) + (int(picture_number) - 1) * image_width].append(shape_2)
    for x_axis, shape_events in x_dict.items():
        shape = FindImgEvent(shape_events)
        shape_bin = bin(shape)[2:][::-1]
        binary1_index = []
        for j, binary in enumerate(shape_bin):
            if binary == '1':
                binary1_index.append(j)

        if shape != 0:
            y_axis = y_size + 0.5
            for j in range(len(binary1_index)):
                event = event_map[binary1_index[j]]
                fig.add_layout_image(
                        x=x_axis,
                        y=y_axis,
                        source=source[event],
                        xref="x",
                        yref="y",
                        sizex=0.5,
                        sizey=0.5,
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        layer="above",
                        visible=True,
                    )
                y_axis += 0.5

    if sliderrange is None:
        figure_x = int(fig.layout['images'][0]['x'])
        sliderrange = [0, figure_x / 20]

    fig.update_layout(
        barmode="stack",
        uniformtext=dict(mode="hide", minsize=10),
        showlegend=False,
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.1),
            range = sliderrange,
            type="linear",
            scaleanchor="y",
            showgrid=False
        ),
        clickmode="event+select",
        dragmode='select',
    )

    # fig.update_yaxes(showticklabels=True, range=[0, y_size + 5], scaleanchor="x", constrain="domain", constraintoward = "top", showgrid=False)
    fig.update_yaxes(showticklabels=True, scaleanchor="x", showgrid=False)

    fig.update(layout = go.Layout(margin=dict(t=0,r=0,b=0,l=0)))

    qid_list = scatter_dataframe['qid'].tolist()
    post_list = scatter_dataframe['picture_number'].tolist()
    j = 0
    for i in range(len(qid_list) - 1):
        if qid_list[i] != qid_list[i + 1]:
            pos_x.append((post_list[i] + (post_list[i] - 1) * image_width + post_list[i] + (post_list[i] - 1) * image_width) / 2 + image_width / 2)
        j += 1
    for x in pos_x:
        fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="black")

    return fig

def FindImgEvent(events):
    shape = 0
    
    for event in events:
        shape = shape | event

    return shape

@callback(
    [Output('button_result', 'children'),
    Output('stacked-bar', 'figure'),
    Output('users-dropdown', 'value'),
    Output(component_id='correct_history', component_property='data'),
    Output(component_id='correct_history', component_property='columns')],
    [Input('users-dropdown', 'value'),
    Input('Discuss_text', 'value'),
    Input('Merge_button', 'n_clicks'),
    Input('Delete_button', 'n_clicks'),
    Input('Split_button', 'n_clicks'),
    Input('Discuss_button', 'n_clicks'),
    Input('Output_file', 'n_clicks'),
    Input('Recovery_button', 'n_clicks'),
    Input('Comment_button', 'n_clicks'),
    Input('Click_button', 'n_clicks'),
    Input('News_button', 'n_clicks'),
    Input('stacked-bar', 'selectedData')],
    [State('stacked-bar', 'relayoutData')]
)
def ButtonClick(uid, discuss_text, merge_btn, delete_btn, split_btn, discuss_btn, file_btn, recovery_btn, comment_btn, click_btn,  news_btn, stacked_select, stacked_relayout): 
    global global_user, stacked_layout, stacked_fig, prev_selection, Select_row_index, width, height

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    if 'stacked-bar' in changed_id:
        bar_list = {}
        if stacked_select is None:
            return "Not select bar yet", dash.no_update, dash.no_update, dash.no_update, dash.no_update

        Select_row_index = []
        for bar in stacked_select['points']:
            bar_id = bar['curveNumber'] 
            if bar['customdata'][0] in event_list:
                return "Don't select event mark", dash.no_update, dash.no_update, dash.no_update, dash.no_update
            Select_row_index.append(int(bar['customdata'][3]))
            bar_list[bar_id] = stacked_fig.data[bar_id].marker.line.color
            stacked_fig.data[bar_id].marker.line.width = 3.5
            stacked_fig.data[bar_id].marker.line.color = "#FFD700"

        Select_row_index = sorted(Select_row_index)

        max_data_index = len(stacked_fig.data) - 1

        for bar_id, color in prev_selection.items():
            if bar_id > max_data_index:
                continue
            if bar_id not in bar_list.keys():
                stacked_fig.data[bar_id].marker.line.color = color

        if stacked_layout is None:
            figure_x = int(stacked_fig.layout['images'][0]['x'])
            sliderrange = [0, figure_x / 20]
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True, thickness=0.1),
                    range = sliderrange,
                    type="linear",
                    scaleanchor="y",
                    showgrid=False
                )
            )
        elif 'xaxis.range' in stacked_layout:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True, thickness=0.1),
                    range = [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]],
                    type="linear",
                    scaleanchor="y",
                    showgrid=False
                )
            )
        else:
            figure_x = int(stacked_fig.layout['images'][0]['x'])
            sliderrange = [0, figure_x / 20]
            stacked_fig.update_layout(
                xaxis=dict(        
                    rangeslider=dict(visible=True, thickness=0.1),
                    range = sliderrange,
                    type="linear",
                    scaleanchor="y",
                    showgrid=False
                )
            )        
        prev_selection = bar_list
        return "You select " + str(len(Select_row_index)) + " bars", stacked_fig, dash.no_update, dash.no_update, dash.no_update 

    if stacked_layout == []:
        stacked_layout = stacked_relayout
    
    if (uid == '' or uid is None) and global_user == 0: #最一開始開這個網頁的時候
        fig = {
            "layout": {
                "xaxis": {
                    "visible": False
                },
                "yaxis": {
                    "visible": False
                },
                "annotations": [
                    {
                        "text": "No data found",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {
                            "size": 24
                        }
                    }
                ]
            }
        }
        columns = [{'name': col, 'id': col} for col in history_column]
        return "Choose one user number", fig, dash.no_update, [{}], columns
    elif (uid == '' or uid is None) and global_user != 0 and global_user != "": #從visual mode回來的時後
        msg = 'Select some bars to start'

        stacked_dataframe = GetDataframe(global_user, port_file)

        if stacked_layout is None: 
            stacked_fig = draw_barchart(stacked_dataframe, None, global_user)
        elif 'xaxis.range' in stacked_layout:
            stacked_fig = draw_barchart(stacked_dataframe, [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]], global_user)
        else:
            stacked_fig = draw_barchart(stacked_dataframe, None, global_user)
        
        Record_table, columns = GetHistory(global_user, port_file)
        
        return msg, stacked_fig, global_user, Record_table, columns
    
    if uid != '' and uid is not None:
        global_user = uid
    if global_user != 0 and global_user != "":
        stacked_dataframe = GetDataframe(global_user, port_file)

    else:
        columns = [{'name': col, 'id': col} for col in history_column]
        fig = {
            "layout": {
                "xaxis": {
                    "visible": False
                },
                "yaxis": {
                    "visible": False
                },
                "annotations": [
                    {
                        "text": "No data found",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {
                            "size": 24
                        }
                    }
                ]
            }
        }
        return "Choose one user number", fig, global_user, [{}], columns
    
    stacked_dataframe.code_id = stacked_dataframe.code_id.astype(int)

    if 'Merge_button' in changed_id:
        if Select_row_index == []:
             return "Must select some bars", dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            msg = "Bars has been combined"
        print(msg)
        
        merge_to = Select_row_index[0]
        BaseDataframe = stacked_dataframe[stacked_dataframe['row_index'] == merge_to].iloc[0]
        for row_index in Select_row_index:
            if row_index != merge_to:           
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, 'color'] = BaseDataframe['color']
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, 'code_id'] = BaseDataframe['code_id']
            post_number = stacked_dataframe[stacked_dataframe['row_index'] == row_index].iloc[0]['code_id']
        update_color_dataframe = stacked_dataframe[stacked_dataframe['code_id'] == int(BaseDataframe['code_id'])]
        for p_id in update_color_dataframe['picture_number'].tolist():
            i = stacked_dataframe[stacked_dataframe['picture_number'] == int(p_id)]['percent'].idxmax()
            stacked_dataframe.loc[i, 'color'] = ",".join(stacked_dataframe.loc[i, 'color'].split(",")[:3]) + ",1)"

        SaveDataframe(global_user, port_file, stacked_dataframe)
        
        stacked_fig, prev_selection = CombineFigUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout)

        HistoryData, columns = SaveHistory(global_user, port_file, Select_row_index, "Combine")
        
        return msg, stacked_fig, dash.no_update, HistoryData, columns
    elif 'Delete_button' in changed_id:
        if Select_row_index == []:
             return "Must select some bars", dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            msg = "Bars has been deleted"
        print(msg)

        for row_index in Select_row_index:
            stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, 'visible time'] = -1

        SaveDataframe(global_user, port_file, stacked_dataframe)

        stacked_fig = DeleteFigUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout, False)
        
        HistoryData, columns = SaveHistory(global_user, port_file, Select_row_index, "Delete")
        return msg, stacked_fig, dash.no_update, HistoryData, columns
    elif 'Split_button' in changed_id:
        if Select_row_index == []:
            return "Must select some bars", dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            msg = "post has been splited"
        print(msg)
  
        Selected_post_number = []
        for row_index in Select_row_index:
            post_id = stacked_dataframe[stacked_dataframe['row_index'] == row_index].iloc[0]['code_id']
            if not post_id in Selected_post_number:
                Selected_post_number.append(post_id)
        if len(Selected_post_number) >= 2:
            return "Two posts are selected, split failed", dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        for post_number in Selected_post_number:
            stacked_dataframe.code_id = stacked_dataframe.code_id.astype(int)

            base_index = stacked_dataframe[stacked_dataframe['code_id'] == int(post_number)].iloc[0]['picture_number']
            
            row_picture_number = []
            for row_index in Select_row_index:
                pic_id = stacked_dataframe[stacked_dataframe['row_index'] == row_index].iloc[0]['picture_number']
                if not pic_id in row_picture_number:
                    row_picture_number.append(pic_id)

            max_row_index = stacked_dataframe['row_index'].max()
            
            code_id_dict = {}
            max_code_id = stacked_dataframe.loc[(stacked_dataframe['row_index'] >= 0) & (stacked_dataframe['row_index'] < Select_row_index[0]), 'code_id'].max()

            for j in range(Select_row_index[0], max_row_index + 1):
                row_index = stacked_dataframe[stacked_dataframe['row_index'] == j].iloc[0]['row_index']
                code_id = stacked_dataframe[stacked_dataframe['row_index'] == j].iloc[0]['code_id']
                if j in Select_row_index:
                    origin_color = stacked_dataframe[stacked_dataframe['row_index'] == j].iloc[0]['color']
                    changed_color = ",".join(origin_color.split(",")[0:-1]) + ",0.3)"
                    stacked_dataframe.loc[stacked_dataframe['row_index'] == j, "color"] = changed_color
                if code_id < int(post_number):
                    continue
                if base_index == row_picture_number[0]:
                    if row_index in Select_row_index:
                        max_code_id = max(code_id, max_code_id)
                        if code_id not in code_id_dict and code_id != int(post_number):
                            code_id_dict[code_id] = max_code_id
                    else:
                        if code_id in code_id_dict:
                            stacked_dataframe.loc[stacked_dataframe['row_index'] == j, 'code_id'] = code_id_dict[code_id]
                        else:
                            max_code_id = max_code_id + 1
                            code_id_dict[code_id] = max_code_id
                            stacked_dataframe.loc[stacked_dataframe['row_index'] == j, 'code_id'] = max_code_id
                else:
                    if row_index in Select_row_index:
                        if code_id in code_id_dict:
                            stacked_dataframe.loc[stacked_dataframe['row_index'] == j, 'code_id'] = code_id_dict[code_id]
                            continue
                        if code_id == int(post_number):
                            max_code_id = max_code_id + 1
                            code_id_dict[code_id] = max_code_id
                            stacked_dataframe.loc[stacked_dataframe['row_index'] == j, 'code_id'] = max_code_id
                        else:
                            if code_id >= max_code_id:
                                max_code_id = max_code_id + 1
                                if code_id not in code_id_dict:
                                    code_id_dict[code_id] = max_code_id
                                stacked_dataframe.loc[stacked_dataframe['row_index'] == j, 'code_id'] = max_code_id
                            else:
                                code_id_dict[code_id] = code_id
                    else:
                        if code_id != int(post_number):
                            if code_id in code_id_dict:
                                stacked_dataframe.loc[stacked_dataframe['row_index'] == j, 'code_id'] = code_id_dict[code_id]
                                continue
                            if code_id >= max_code_id:
                                max_code_id = max_code_id + 1
                                if code_id not in code_id_dict:
                                    code_id_dict[code_id] = max_code_id  
                                stacked_dataframe.loc[stacked_dataframe['row_index'] == j, 'code_id'] = max_code_id
                            else:
                                code_id_dict[code_id] = code_id                 
        print(msg)

        stacked_fig, prev_selection = SplitFigUpdate(stacked_fig, stacked_dataframe, post_number, stacked_layout)
        
        SaveDataframe(global_user, port_file, stacked_dataframe)

        HistoryData, columns = SaveHistory(global_user, port_file, Select_row_index, "Split")
        return msg, stacked_fig, dash.no_update, HistoryData, columns

    elif 'Discuss_button' in changed_id:

        if Select_row_index == []:
            return "Must select some bars", dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            msg = "Bars has been mark as discussion"

        for row_index in Select_row_index:
            stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, 'discuss'] = 1
            stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, 'discuss_reason'] = discuss_text
            stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, 'color'] = "rgba(190,190,190,0.4)"
        print(msg)

        stacked_fig, prev_selection = DiscussFigUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout)

        SaveDataframe(global_user, port_file, stacked_dataframe)

        HistoryData, columns = SaveHistory(global_user, port_file, Select_row_index, "Discussion")
        return msg, stacked_fig, dash.no_update, HistoryData, columns

    elif 'Comment_button' in changed_id or 'Click_button' in changed_id or 'News_button' in changed_id:
        if Select_row_index == []:
            return "Must select some bars", dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            if 'Comment_button' in changed_id:
                msg = "Bars has been marked as comment"
                col = 'comment'
                button_name = "Comment"
            elif 'Click_button' in changed_id:
                msg = "Bars has been mark as external link"
                col = 'outer_link'
                button_name = "External link"
            else:
                msg = "Bars has been mark as news"
                col = 'news'
                button_name = "News"

        # row_index_list = []
        for row_index in Select_row_index:
            if stacked_dataframe[stacked_dataframe['row_index'] == row_index].iloc[0]["comment"] == 1 and col == "outer_link":
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, col] = 1
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, "comment"] = 0
            elif stacked_dataframe[stacked_dataframe['row_index'] == row_index].iloc[0]["outer_link"] == 1 and col == "comment":
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, col] = 1
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, "outer_link"] = 0
            elif stacked_dataframe[stacked_dataframe['row_index'] == row_index].iloc[0][col] == 0:
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, col] = 1
            else:
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, col] = 0
            # row_index_list.append(row_index)
        # print("from_no_event_to_event", from_no_event_to_event)
        print(msg)

        SaveDataframe(global_user, port_file, stacked_dataframe)

        y_size = height * image_width / width

        stacked_fig, prev_selection = EventFigUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout, y_size)

        HistoryData, columns = SaveHistory(global_user, port_file, Select_row_index, button_name)
        return msg, stacked_fig, dash.no_update, HistoryData, columns
    elif 'Recovery_button' in changed_id: 
        cache_file_list = sorted(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user))
        file_len = len(cache_file_list)
        if file_len > 1:
            now_path = cache_file_list[-1]                           
            previous_path = cache_file_list[-2]
            try:
                stacked_dataframe = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + previous_path, encoding="utf_8_sig")
            except:
                stacked_dataframe = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + previous_path, engine='python')            
            os.remove(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + now_path)
            
            if os.path.exists(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json"): 
                with open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json") as f:
                    history = json.load(f)
                    if global_user in history:
                        history_user = history[global_user]
                        step_number = history_user[0]["Correction step"]
                        row_number_list = [int(x) for x in history_user[0]["Row number"].split(",")]
                        action = history_user[0]["Button click"]
                        # print(row_number_list, action)
                        history_user.pop(0)
                        history[global_user] = history_user
                        if step_number == "1":
                            msg = "has resumed 1st step"
                        elif step_number == "2":
                            msg = "has resumed 2nd step"
                        elif step_number == "3":
                            msg = "has resumed 3rd step"
                        else:
                            msg = "has resumed " + step_number + "th step"
                    else:
                        msg = "this is original file"
                        return msg, dash.no_update, dash.no_update, dash.no_update, dash.no_update
                with open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json", "w") as f:
                    json.dump(history, f)
                columns = [{'name': col, 'id': col} for col in history_column]
                if action == "Combine":
                    stacked_fig, prev_selection = CombineFigUpdate(stacked_fig, stacked_dataframe, row_number_list, stacked_layout)
                elif action == "Delete":
                    stacked_fig = DeleteFigUpdate(stacked_fig, stacked_dataframe, row_number_list, stacked_layout, True)
                elif action == "Split":
                    post_number = stacked_dataframe[stacked_dataframe['row_index'] == row_number_list[0]].iloc[0]['code_id']
                    stacked_fig, prev_selection = SplitFigUpdate(stacked_fig, stacked_dataframe, int(post_number), stacked_layout)
                elif action == "Discussion":
                    stacked_fig, prev_selection = DiscussFigUpdate(stacked_fig, stacked_dataframe, row_number_list, stacked_layout)
                elif action == "Comment" or action == "External link" or action == "News":
                    y_size = height * image_width / width
                    stacked_fig, prev_selection = EventFigUpdate(stacked_fig, stacked_dataframe, row_number_list, stacked_layout, y_size)

                return msg, stacked_fig, dash.no_update, history[global_user] , columns
                
            else:
                msg = "this is original file"
                return msg, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            msg = "this is original file"
            return msg, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if 'users-dropdown' in changed_id:
        print("users-dropdown")        
        msg = 'Select some bars to start'

        stacked_fig = draw_barchart(stacked_dataframe, None, global_user)

        if len(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "")) == 0:
            SaveDataframe(global_user, port_file, stacked_dataframe)
        
        print(msg)

        Record_table, columns = GetHistory(global_user, port_file)

        return msg, stacked_fig, global_user, Record_table, columns
    elif 'Output_file' in changed_id:
        msg = global_user + " has been downloaded"
        return msg, dash.no_update, global_user, dash.no_update, dash.no_update
    else:
        msg = 'Not click any button yet'
        return msg, dash.no_update, global_user, dash.no_update, dash.no_update

@callback(
    Output('download_file', 'data'),
    Input('users-dropdown', 'value'),
    Input('Output_file', 'n_clicks'),
    prevent_initial_call=True,
)
def DownloadClick(uid, file_btn):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if "Output_file" in changed_id:
        stacked_dataframe = GetDataframe(global_user, port_file)
        stacked_dataframe = stacked_dataframe.drop(stacked_dataframe[(stacked_dataframe.code_id == -2) | (stacked_dataframe['visible time'] == -1)].index)

        return dcc.send_data_frame(stacked_dataframe.to_excel, global_user + "_PostCodingData.xlsx", sheet_name="Sheet1")
    
@app.server.route('{}<image_path>.jpg'.format(static_image_route))
def serve_image(image_path):
    path = image_path.replace(" ", "/") + ".jpg"
    temp =  path.split("/")
    image_name = temp[len(temp) - 1]
    path = '/'.join(temp[:len(temp) - 1]) 
    img_exits = False
    
    image_directory = ROOT_PATH + path
    full_image_path = image_directory + "/" + image_name
    if os.path.exists(full_image_path): 
        img_exits = True
    if not img_exits:
        print('"{}" is excluded from the allowed static files'.format(image_name))
    return flask.send_from_directory(image_directory, image_name)

@callback(
    Output('placeholder', 'children'),
    Input('stacked-bar', 'relayoutData')
)
def RecordBarChartLayout(stacked_relayout):
    global stacked_layout
    if stacked_relayout is not None:
        if 'autosize' not in stacked_relayout: 
            stacked_layout = stacked_relayout
    return dash.no_update

#Update the index
@callback(Output('page-content', 'children'),
        [Input('url', 'pathname')])
def index_page(pathname):
    print("index page name:", pathname)
    return coding_layout

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=port_number, threaded=True, debug=True)
