# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:46:32 2022

@author: MUILab-VR
"""

import dash
import dash_auth
from dash import callback_context, Dash, dcc, html, Input, Output, callback, dash_table
from dash.dependencies import State
from matplotlib.cbook import Stack
import plotly.graph_objs as go
import pandas as pd
import datetime as dt
from flask import make_response
from plotly.validators.scatter.marker import SymbolValidator
import plotly.express as px
import numpy as np
import flask
import datetime
import calendar
import time
import io
from collections import defaultdict
from textwrap import dedent as s
from PIL import Image
import re
import copy
import os
import cv2
import base64
from PIL import ImageColor
import json
from CreateImage import draw_visible_area
from CoderDataPreprocessingforDev import data_preprocess

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
ROOT_PATH = 'D:/Users/MUILab-VR/Desktop/News Consumption/CHI2022/'
data_path = ROOT_PATH + "Analysis/Visualization/data/"

# data_preprocess()

global_user = 0
event_list = ["external link", 'comment', "news", "external&news", "comment&news"]
source = {'post':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png"),
        'news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/news.png"),
        'external link':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/link.png"),
        'external&news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/link.png"),
        'comment':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png"),
        'comment&news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png")}
scatter_layout = []
stacked_layout = []
prev_selection = []
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
            html.Div([
                html.Button('Combine', id='Merge_button', n_clicks=0, style={"margin-left": "20px"}),
                html.Button('Delete', id='Delete_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('Split', id='Split_button', n_clicks=0, style={"margin-left": "10px"}),
                html.Button('Comment', id='Comment_button', n_clicks=0, style={"margin-left": "10px", "margin-top": "5px"}),
                html.Button('External Link', id='Click_button', n_clicks=0, style={"margin-left": "10px", "margin-top": "5px"}),
                html.Button('News', id='News_button', n_clicks=0, style={"margin-left": "10px", "margin-top": "5px"}),                 
            ]),            
        ], style={'display': 'inline-block', 'vertical-align':'top','margin-left': '10px', 'margin-top': '10px'}),

        html.Div(
            html.Div([
                html.P(id="button_result", style={'fontSize':16}),
            ])
            , style={'display': 'inline-block', 'vertical-align':'top','margin-left': '10px', 'margin-top': '10px'}
        ),

        html.Div(
            html.Div([
                html.P(id="accuracy"),
            ])
            , style={'display': 'none', 'vertical-align':'bottom','margin-left': '20px', 'margin-top': '10px'}
        ),
    ]),
    html.Div(
        html.Div([
            html.P("Discuss Reason：", style={'fontSize':18}),
        ]),
        style={'display': 'inline-block','margin-left': '10px', 'margin-top': '10px'}
    ),
    html.Div(
        html.Div([
            dcc.Input(
                id='Discuss_text',
                type='text',
                style={'width':'200px'}
            ),
            html.Button('Record the reason', id='Discuss_button', n_clicks=0, style={"margin-left": "10px"}),
            html.Button('Resume Previous Step', id='Recovery_button', n_clicks=0, style={"margin-left": "20px"}),
            dcc.Download( id="download_file"),
            html.Button('Output File', id='Output_file', n_clicks=0, style={"margin-left": "10px"}), 
        ])
        , style={ 'display': 'inline-block','margin-left': '10px', 'margin-top': '10px'}
    ),
    html.Div(
        dcc.Link('Go to coding mode', href='/coding')
    ,style={'display': 'none', 'vertical-align':'bottom','margin-left': '20px', 'margin-top': '10px'}),
    html.Div(
        dcc.Graph(
            id='stacked-bar',
            config={"displayModeBar": False},
            style={'height': '35vh'}
        )
    ,style={ 'width': '100%','margin-top': '10px'}),
        
    html.Div(
       html.Div([
                html.B(id="post_num", style={'fontSize':16}),
            ])
        , style={'width': '20%', 'display': 'inline-block', 'margin-left': '10px', 'margin-top': '10px'}
    ),
    
    html.Div([
            html.Div(html.P("", id = "post_content"), style={'width': '15%', "height": "60vh", "overflow": "scroll", 'margin-left': '10px', 'display': 'inline-block', 'flex':1, "margin-bottom": "15px",'fontSize':18}),
            html.Div(id='img-content', style={"margin-bottom": "15px", "height": "60vh", "overflow": "scroll", 'flex':5, 'text-align':'center'})
        ]
    ,style={'display': 'flex', 'flex-direction': 'row'}),        

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

def RecordHistory(user, row_index_list, action):
    if os.path.exists(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json"): 
        with open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json") as f:
            history = json.load(f)
            if user in history:
                history_user = history[user]
                if history_user != []:
                    current_step = int(history_user[0]["Correction step"]) + 1
                    now_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                    history_user.insert(0, {"Time": now_time, "Correction step":str(current_step), "Row number": ",".join(str(e) for e in row_index_list), "Button click": action})
                    history[user] = history_user
                else:
                    now_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                    history[user] = [{"Time": now_time, "Correction step":"1", "Row number": ",".join(str(e) for e in row_index_list), "Button click": action}]
            else:
                now_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                history[user] = [{"Time": now_time, "Correction step":"1", "Row number": ",".join(str(e) for e in row_index_list), "Button click": action}]
        with open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json", "w") as f:
            json.dump(history, f)
    else:
        with open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json", "w") as f:
            now_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            history = {user:[{"Time": now_time, "Correction step":"1", "Row number": ",".join(str(e) for e in row_index_list), "Button click": action}]}
            json.dump(history, f)
    columns = [{'name': col, 'id': col} for col in history_column]
    return history[user], columns

def BarChart_Preprocessing(df):
    facebook_color = ["rgba(2,22,105,1)",  "rgba(5,61,180,1)", 
                    "rgba(0,68,255,1)", "rgba(4,134,219,1)", "rgba(64, 187, 213,1)",
                    "rgba(87,226,255,1)", "rgba(135,206,255,1)", "rgba(46, 180, 237,1)"]
    youtube_color = ["rgba(255, 20, 147,1)",  "rgba(248, 0, 0,1)", "rgba(255, 100, 0,1)",
                    "rgba(255, 150, 0,1)", "rgba(255, 126, 106,1)", "rgba(204, 75, 101,1)",
                     "rgba(205, 140, 149,1)", "rgba(202, 9, 53,1)"]
    instagram_color = ["rgba(139, 0, 139,1)",  "rgba(85, 26, 139,1)", "rgba(115, 35, 189,1)",
                      "rgba(171, 156, 255,1)", "rgba(213, 172, 255,1)", "rgba(255, 187, 255,1)"]
    All_color = facebook_color + youtube_color + instagram_color
    
    max_postID = df['code_id'].max()
    color_scale_dict = {}
    for i in range(max_postID):
        color = facebook_color[i % len(facebook_color)]
        color_scale_dict[i + 1] = color
        
    fig_dataframe = []    
    pic_num = 0
    for index in range(len(df)):
        image = df.at[index, 'images']
        if image not in fig_dataframe: #一篇貼文的第一張照片
            pic_num += 1 
            df.loc[index, 'picture_number'] = pic_num 
            fig_dataframe.append(image)
        else:
            df.loc[index, 'picture_number'] = pic_num  
    
    fig_dataframe = []
    df_seperate = []
    
    scatter_dataframe = pd.DataFrame(columns=df.columns)
    for dataf in df_seperate:
        scatter_dataframe = scatter_dataframe.append(dataf, ignore_index=True)
    df = df.reset_index(drop=True)

    for index in range(len(df)):
        df.at[index, 'color'] = color_scale_dict[df.at[index, 'code_id']]

    df["code_id"] = df["code_id"].astype(str)
    print("end BarChart_Preprocessing")
    return df

def nonvisual_bottombar(picture_number, non_visual_costomdata):
    bottom_bar = go.Bar(
            name="-1",
            y=[0.13],
            x=[int(picture_number)],
            marker=dict(color="rgba(180,180,180,0.8)"),
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
            marker=dict(color="rgba(180,180,180,0.8)"),
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

def draw_barchart(df, sliderrange):
    if not df.empty:
        scatter_dataframe = df.drop(df.loc[df['visible time'] == -1].index, inplace=False)
        scatter_dataframe.reset_index(inplace = True, drop = True)
    pos_x = []

    fig = go.Figure()

    prev_img = "none"
    prev_shape = -1
    for i in range(len(scatter_dataframe) - 1, -1, -1):
        img = scatter_dataframe['images'][i] 
        code_id = scatter_dataframe['code_id'][i]  
        percent = scatter_dataframe['percent'][i]
        picture_number = scatter_dataframe['picture_number'][i]
        color = scatter_dataframe['color'][i]
        detect_time = scatter_dataframe['detect_time'][i]
        qid = scatter_dataframe['qid'][i]
        row_index = scatter_dataframe['row_index'][i]
        comment = scatter_dataframe['comment'][i]
        link = scatter_dataframe['outer_link'][i]
        news = scatter_dataframe['news'][i]

        event = "comment"
        shape_2 = int(str(comment) + str(link) + str(news), 2)

        if shape_2 == 4:
            event = "comment"
        elif shape_2 == 2:
            event = "external link"
        elif shape_2 == 1:
            event = "news"
        elif shape_2 == 5:
            event = "comment&news"
        elif shape_2 == 3:
            event = "external&news"
        elif shape_2 == 0:
            event = "post"

        if prev_img != img:
            non_visual_costomdata = [code_id, detect_time, qid, row_index, picture_number]
            bottom_bar = nonvisual_bottombar(picture_number, non_visual_costomdata)
            fig.add_trace(bottom_bar)
            prev_shape = -1

        if shape_2 != 0:
            if (prev_shape == 1 or prev_shape == 2 or prev_shape == 4) and prev_img == img:
                if prev_shape != shape_2:
                    if prev_shape == 2 and shape_2 == 3 :
                        source_event = "news"
                    elif prev_shape == 4 and shape_2 == 5:
                        source_event = "news"
                    elif prev_shape == 1 and shape_2 == 2:
                        source_event = "external link"
                    elif prev_shape == 1 and shape_2 == 4:
                        source_event = "comment"
                    else:
                        source_event = event
                    fig.add_layout_image(
                        x=int(picture_number)+0.4,
                        y=1.25,
                        source=source[source_event],
                        xref="x",
                        yref="y",
                        sizex=0.6,
                        sizey=0.6,
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        layer="above",
                        visible=True,
                    )
            else:
                fig.add_layout_image(
                    x=int(picture_number)+0.4,
                    y=1.1,
                    source=source[event],
                    xref="x",
                    yref="y",
                    sizex=0.6,
                    sizey=0.6,
                    xanchor="center",
                    yanchor="middle",
                    sizing="contain",
                    layer="above",
                    visible=True,
                )

                if shape_2 == 3 or shape_2 == 5:
                    fig.add_layout_image(
                        x=int(picture_number)+0.4,
                        y=1.25,
                        source=source["news"],
                        xref="x",
                        yref="y",
                        sizex=0.6,
                        sizey=0.6,
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        layer="above",
                        visible=True,
                    )

        bar = go.Bar(
                    name=str(code_id),
                    y=[percent],
                    x=[int(picture_number)],
                    marker=dict(color=color),
                    offset=0,
                    customdata=[[code_id, detect_time, qid, row_index, picture_number, event]],
                    hovertemplate="<br>".join([
                    "post_number=%{customdata[0]}",
                    "detect time=%{customdata[1]}",
                    "questionnaire id=%{customdata[2]}",
                    "percent=%{y}",
                    "row index=%{customdata[3]}", 
                    "picture number=%{customdata[4]}", 
                    "event=%{customdata[5]}",                           
                    ]),
                    unselected=dict(marker=dict(opacity=0.5))
                )
        fig.add_trace(bar)  

        if i - 1 >= 0 and scatter_dataframe['images'][i - 1] != img:
            non_visual_costomdata = [code_id, detect_time, qid, row_index, picture_number]
            top_bar = nonvisual_topbar(picture_number, non_visual_costomdata)
            fig.add_trace(top_bar)  
        
        if i == 0:
            non_visual_costomdata = [code_id, detect_time, qid, row_index, picture_number]
            top_bar = nonvisual_topbar(picture_number, non_visual_costomdata)
            fig.add_trace(top_bar)

        prev_shape = shape_2
        prev_img = img

    fig.update_layout(
        barmode="stack",
        uniformtext=dict(mode="hide", minsize=10),
        showlegend=False,
        xaxis=dict(
            rangeslider=dict(visible=True),
            range = sliderrange,
            type="linear"
        ),
        clickmode="event+select",
        dragmode='select'
    )

    fig.update_yaxes(showticklabels=False, range=[0,1.35])
    fig.update(layout = go.Layout(margin=dict(t=0,r=0,b=0,l=0)))
    qid_list = scatter_dataframe['qid'].tolist()
    post_list = scatter_dataframe['picture_number'].tolist()
    for i in range(len(qid_list) - 1):
        if qid_list[i] != qid_list[i + 1]:
            pos_x.append(post_list[i + 1])
    for x in pos_x:
        fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="black")

    return fig

def FindImgEvent(events):
    shape = 0
    
    for event in events:
        shape = shape | event
    if shape == 0:
        event = "post"
    elif shape == 4:
        event = "comment"
    elif shape == 2:
        event = "external link"
    elif shape == 1:
        event = "news"
    elif shape == 5:
        event = "comment&news"
    elif shape == 3:
        event = "external&news"
    else :
        event = "comment&news"

    return event

@callback(
    Output('img-content', 'children'),
    Input('stacked-bar', 'hoverData'),
    Input('users-dropdown', 'value'),
    [Input('button_result', 'children')])
def update_image(stacked_hover, userid, button_result):

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'users-dropdown' in changed_id:
        return dash.no_update

    if stacked_hover is None:
        return dash.no_update 

    result = []
    if "button_result" in changed_id:
        if "拆開" in button_result or "上一步" in button_result:
            return dash.no_update
        hoverData = stacked_hover
        if hoverData['points'][0]['customdata'][0] in event_list:
            picture_number = int(hoverData['points'][0]['customdata'][3])
        else:
            picture_number = int(hoverData['points'][0]['customdata'][4])
        
    elif 'stacked-bar' in changed_id:
        hoverData = stacked_hover
        if hoverData['points'][0]['customdata'][0] in event_list:
            picture_number = int(hoverData['points'][0]['customdata'][3])
        else:
            picture_number = int(hoverData['points'][0]['customdata'][4])

    ### used for searching comment and outer_link value
    stacked_file_path = sorted(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + userid))[-1]
    try:
        cache = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + userid + "/" + stacked_file_path, encoding="utf_8_sig")
    except:
        cache = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + userid + "/" + stacked_file_path, engine='python')
    max_picture_number = int(cache['picture_number'].max())

    picture_range = [0, 0]
    if picture_number <= 2:
        # print("picture_number smaller than 2")
        picture_range[0] = 1
        picture_range[1] = picture_number + 2
    elif picture_number + 2 >= max_picture_number:
        # print("picture_number larger than max")
        picture_range[0] = picture_number - 2
        picture_range[1] = max_picture_number
    else:
        # print("else")
        picture_range[0] = picture_number - 2
        picture_range[1] = picture_number + 2
    
    p_num = 0
    bias = 0
    if picture_range[1] == 3:
        bias = 2
        for i in range(bias):
            result.append(html.Img(id= "img"+str(i)))
    elif picture_range[1] == 4:
        bias = 1
        for i in range(bias):
            result.append(html.Img(id= "img"+str(i)))        

    for i, pic_num in enumerate(range(picture_range[0], picture_range[1] + 1)):
        p_num += 1
        img = cache[cache['picture_number'] == pic_num].iloc[0]['images']
        q_id = cache[cache['picture_number'] == pic_num].iloc[0]['qid']
        code_id = cache[cache['picture_number'] == pic_num].iloc[0]['code_id']
        userid = userid.split("-")[0] if "-" in userid else userid

        temp_path = os.listdir(ROOT_PATH + "/" + userid)
        if cache.loc[(cache["images"] == img) & (cache["code_id"] == code_id), "comment"].shape[0] == 0:
            return dash.no_update
        # print(ROOT_PATH + userid + "/" + temp_path[0] + "/NewInterval/" + str(q_id) + "/" + img)
        img_cv2 = cv2.imread(ROOT_PATH + userid + "/" + temp_path[0] + "/NewInterval/" + str(q_id) + "/" + img)
        img_cv2 = draw_visible_area(img_cv2)
        _, buffer = cv2.imencode('.jpg', img_cv2)
        img_64 = base64.b64encode(buffer).decode('utf-8')

        if pic_num == picture_number:
            width = "15%"
        else:
            width = "10%"
        result.append(html.Img(id= "img"+str(i + bias), src='data:image/jpg;base64,{}'.format(img_64), style={'width': width, 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}))    
    if picture_range[0] == max_picture_number - 2:
        for i in range(3, 5):
            result.append(html.Img(id= "img"+str(i)))
    elif picture_range[0] == max_picture_number - 3:
        for i in range(4, 5):
            result.append(html.Img(id= "img"+str(i)))
    # print("-------------------------------------------------------------------")
    return result

@app.callback(Output('img0', 'style'),
    Output('img1', 'style'),
    Output('img2', 'style'),
    Output('img3', 'style'),
    Output('img4', 'style'),
    Input('img0', 'n_clicks'),
    Input('img1', 'n_clicks'),
    Input('img2', 'n_clicks'),
    Input('img3', 'n_clicks'),
    Input('img4', 'n_clicks')
    )
def display_image(n_click0, n_click1, n_click2, n_click3, n_click4):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'img0' in changed_id:
        if n_click0 % 2 == 0:
            return {'width': "10%", 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            return {'width': "18%", 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif 'img1' in changed_id:
        if n_click1 % 2 == 0:
            return dash.no_update, {'width': "10%", 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}, dash.no_update, dash.no_update, dash.no_update
        else:
            return dash.no_update, {'width': "18%", 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}, dash.no_update, dash.no_update, dash.no_update
    elif 'img2' in changed_id:
        if n_click2 % 2 == 0:
            return dash.no_update, dash.no_update, {'width': "15%", 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}, dash.no_update, dash.no_update
        else:
            return dash.no_update, dash.no_update, {'width': "18%", 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}, dash.no_update, dash.no_update
    elif 'img3' in changed_id:
        if n_click3 % 2 == 0:
            return dash.no_update, dash.no_update, dash.no_update, {'width': "10%", 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}, dash.no_update
        else:
            return dash.no_update, dash.no_update, dash.no_update, {'width': "18%", 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}, dash.no_update
    elif 'img4' in changed_id:
        if n_click4 % 2 == 0:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, {'width': "10%", 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, {'width': "18%", 'margin-right': '40px', 'margin-bottom': '20px', 'vertical-align':'top'}
    return dash.no_update

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
    global global_user, stacked_layout, stacked_fig, prev_selection, Select_row_index

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    if 'stacked-bar' in changed_id:
        bar_list = []
        if stacked_select is None:
            return "Not select bar yet", dash.no_update, dash.no_update, dash.no_update, dash.no_update

        Select_row_index = []
        for bar in stacked_select['points']:
            bar_id = bar['curveNumber'] 
            if bar['customdata'][0] in event_list:
                return "Don't select event mark", dash.no_update, dash.no_update, dash.no_update, dash.no_update
            Select_row_index.append(int(bar['customdata'][3]))
            stacked_fig.data[bar_id].marker.line.width = 3.5
            stacked_fig.data[bar_id].marker.line.color = "#FFD700"
            bar_list.append(bar_id)

        Select_row_index = sorted(Select_row_index)

        for i in prev_selection:
            if i not in bar_list:
                stacked_fig.data[i].marker.line.width = 0

        if stacked_layout is None:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )
        elif 'xaxis.range' in stacked_layout:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]],
                    type="linear"
                )
            )
        else:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )        
        prev_selection = bar_list
        return "You select " + str(len(Select_row_index)) + " bars", stacked_fig, dash.no_update, dash.no_update, dash.no_update

    if stacked_layout == []:
        stacked_layout = stacked_relayout
    
    if (uid == '' or uid is None) and global_user == 0: #最一開始開這個網頁的時候
        return "Choose one user number", dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif (uid == '' or uid is None) and global_user != 0 and global_user != "": #從visual mode回來的時後
        msg = 'Select some bars to start'
        stacked_file_path = sorted(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user))[-1]

        try:
            stacked_dataframe = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + stacked_file_path, encoding="utf_8_sig")
        except:
            stacked_dataframe = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + stacked_file_path, engine='python')

        if stacked_layout is None: 
            stacked_fig = draw_barchart(stacked_dataframe, None)
        elif 'xaxis.range' in stacked_layout:
            stacked_fig = draw_barchart(stacked_dataframe, [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]])
        else:
            stacked_fig = draw_barchart(stacked_dataframe, None)
        
        if os.path.exists(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json"): 
            f = open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json")
            history = json.load(f)
            columns = [{'name': col, 'id': col} for col in history_column]
            if global_user in history:
                return msg, stacked_fig, global_user, history[global_user], columns
            else:
                return msg, stacked_fig, global_user, [{}], columns
        else:
            columns = [{'name': col, 'id': col} for col in history_column]
            return msg, stacked_fig, global_user, [{}], columns
    
    if uid != '' and uid is not None:
        global_user = uid
    if global_user != 0 and global_user != "":
        if len(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user)) != 0:
            stacked_file_path = sorted(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user))[-1]
            try:
                stacked_dataframe = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + stacked_file_path, encoding="utf_8_sig")
            except:
                stacked_dataframe = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + stacked_file_path, engine='python')
        else:
            try:
                scatter_df = pd.read_csv(data_path + global_user + ".csv", encoding="utf_8_sig")
            except:
                scatter_df = pd.read_csv(data_path + global_user + ".csv", engine='python')
            # real data
            scatter_df = scatter_df.drop(scatter_df[scatter_df.code_id == -2].index)
            scatter_df = scatter_df.drop(columns=['n_main', 'keyword', 'picture_number'])
            scatter_df['code_id'] = scatter_df['code_id'].apply(lambda x:int(x) + 1)
            scatter_df.reset_index(inplace=True,drop=True)

            stacked_dataframe = copy.deepcopy(scatter_df)
            stacked_dataframe['detect_time'] = stacked_dataframe[image_appeared].apply(lambda x:extract_time_from_answer(x))
            stacked_dataframe['row_index'] = stacked_dataframe.index
            stacked_dataframe['visible time'] = 0
            stacked_dataframe['discuss'] = 0
            stacked_dataframe['discuss_reason'] = " "
            stacked_dataframe = BarChart_Preprocessing(stacked_dataframe)
    else:
        columns = [{'name': col, 'id': col} for col in history_column]
        return "Choose one user number", dash.no_update, global_user, [{}], columns
    
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

        timestamp = calendar.timegm(time.gmtime())
        stacked_dataframe.to_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file  + "/" + global_user + "/_" + str(timestamp) + ".csv", encoding='utf_8_sig', index=False)
        
        
        updatefig_dataframe = stacked_dataframe[stacked_dataframe['row_index'].isin(Select_row_index)]
        
        update_fig_index = []
        for i, trace in enumerate(stacked_fig.data):
            if trace['customdata'][0][0] not in event_list:
                row_id = trace['customdata'][0][3]
                if trace['name'] != "-1" and row_id in Select_row_index:
                    update_fig_index.append(i)
        update_fig_index = sorted(update_fig_index, reverse=True)

        for index, (i, row) in zip(update_fig_index, updatefig_dataframe.iterrows()):           
            code_id = row['code_id']  
            percent = row['percent']
            picture_number = row['picture_number']
            color = row['color']
            detect_time = row['detect_time']
            qid = row['qid']
            row_index = row['row_index']
            shape = int(str(row['comment']) + str(row['outer_link']) + str(row['news']), 2)
            event = FindImgEvent([shape])

            stacked_fig.data[index].update(
                name=str(code_id),
                y=[percent],
                x=[int(picture_number)],
                marker=dict(color=color),
                offset=0,
                customdata=[[code_id, detect_time, qid, row_index, picture_number, event]],
                hovertemplate="<br>".join([
                    "post_number=%{customdata[0]}",
                    "detect time=%{customdata[1]}",
                    "questionnaire id=%{customdata[2]}",
                    "percent=%{y}",
                    "row index=%{customdata[3]}", 
                    "picture number=%{customdata[4]}", 
                    "event=%{customdata[5]}"                           
                    ])
            )

        if stacked_layout is None:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )
        elif 'xaxis.range' in stacked_layout:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]],
                    type="linear"
                )
            )
        else:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )
        HistoryData, columns = RecordHistory(global_user, Select_row_index, "Combine")
        
        return msg, stacked_fig, dash.no_update, HistoryData, columns
    elif 'Delete_button' in changed_id:
        if Select_row_index == []:
             return "Must select some bars", dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            msg = "Bars has been deleted"
        print(msg)

        for row_index in Select_row_index:
            stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, 'visible time'] = -1

        timestamp = calendar.timegm(time.gmtime())
        stacked_dataframe.to_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file  + "/" + global_user + "/_" + str(timestamp) + ".csv", encoding='utf_8_sig', index=False)

        updatefig_dataframe = stacked_dataframe[stacked_dataframe['row_index'].isin(Select_row_index)]

        update_fig_index = []
        update_for_event = defaultdict(list)
        for i, trace in enumerate(stacked_fig.data):
            row_index = trace['customdata'][0][3]
            if trace['customdata'][0][0] in event_list:
                row_id = trace['customdata'][0][2]
                if trace['name'] != "-1" and row_id in Select_row_index:
                    update_for_event[row_id].append(i)
            else:
                row_id = trace['customdata'][0][3]
                if trace['name'] != "-1" and row_id in Select_row_index:
                    update_fig_index.append(i)
                if trace['name'] == "-1":
                    update_for_event[row_id].append(i)

        update_fig_index = sorted(update_fig_index, reverse=True)

        for index, (i, row) in zip(update_fig_index, updatefig_dataframe.iterrows()):
            row_index = row['row_index']
            picture_number = row['picture_number']
            if row_index in update_for_event:
                for j in update_for_event[row_index]:
                    stacked_fig.data[j].visible = False
                for j, layout_img in enumerate(stacked_fig.layout['images']):
                    if layout_img['x'] == int(picture_number) + 0.4:
                        stacked_fig.layout['images'][j].visible=False
            stacked_fig.data[index].visible = False

        if stacked_layout is None:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )
        elif 'xaxis.range' in stacked_layout:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]],
                    type="linear"
                )
            )
        else:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )
        HistoryData, columns = RecordHistory(global_user, Select_row_index, "Delete")
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

        updatefig_dataframe = stacked_dataframe[stacked_dataframe['code_id'] >= int(post_number)]
        
        update_fig_index = []
        for i, trace in enumerate(stacked_fig.data):
            if trace['customdata'][0][0] not in event_list:
                row_id = trace['customdata'][0][3]
                code_id = trace['customdata'][0][0]
                if trace['name'] != "-1" and code_id >= int(post_number):
                    update_fig_index.append(i)
        update_fig_index = sorted(update_fig_index, reverse=True)

        for index, (i, row) in zip(update_fig_index, updatefig_dataframe.iterrows()):
            code_id = row['code_id']  
            percent = row['percent']
            picture_number = row['picture_number']
            color = row['color']
            detect_time = row['detect_time']
            qid = row['qid']
            row_index = row['row_index']
            shape = int(str(row['comment']) + str(row['outer_link']) + str(row['news']), 2)
            event = FindImgEvent([shape])

            stacked_fig.data[index].update(
                name=str(code_id),
                y=[percent],
                x=[int(picture_number)],
                marker=dict(color=color),
                offset=0,
                customdata=[[code_id, detect_time, qid, row_index, picture_number, event]],
                # marker_pattern_shape=pattern_sahpe[shape],
                hovertemplate="<br>".join([
                    "post_number=%{customdata[0]}",
                    "detect time=%{customdata[1]}",
                    "questionnaire id=%{customdata[2]}",
                    "percent=%{y}",
                    "row index=%{customdata[3]}", 
                    "picture number=%{customdata[4]}",   
                    "event=%{customdata[5]}"                         
                    ])
            )

        if stacked_layout is None:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )
        elif 'xaxis.range' in stacked_layout:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]],
                    type="linear"
                )
            )
        else:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )
        timestamp = calendar.timegm(time.gmtime())
        stacked_dataframe.to_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file  + "/" + global_user + "/_" + str(timestamp) + ".csv", encoding='utf_8_sig', index=False)

        HistoryData, columns = RecordHistory(global_user, Select_row_index, "Split")
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

        updatefig_dataframe = stacked_dataframe[stacked_dataframe['row_index'].isin(Select_row_index)]
        
        update_fig_index = []
        update_for_event = {}
        for i, trace in enumerate(stacked_fig.data):
            if trace['customdata'][0][0] in event_list:
                row_id = trace['customdata'][0][2]
                if trace['name'] != "-1" and row_id in Select_row_index:
                    update_for_event[row_id] = i
            else:
                row_id = trace['customdata'][0][3]
                if trace['name'] != "-1" and row_id in Select_row_index:
                    update_fig_index.append(i)
        update_fig_index = sorted(update_fig_index, reverse=True)

        for index, (i, row) in zip(update_fig_index, updatefig_dataframe.iterrows()):           
            color = row['color']
            row_index = row['row_index']

            stacked_fig.data[index].update(
                marker=dict(color=color)
            )

        if stacked_layout is None:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )
        elif 'xaxis.range' in stacked_layout:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]],
                    type="linear"
                )
            )
        else:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )

        timestamp = calendar.timegm(time.gmtime())
        stacked_dataframe.to_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file  + "/" + global_user + "/_" + str(timestamp) + ".csv", encoding='utf_8_sig', index=False)
        HistoryData, columns = RecordHistory(global_user, Select_row_index, "Discussion")
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

        updatefig_dataframe = stacked_dataframe[stacked_dataframe['row_index'].isin(Select_row_index)]
        # row_index_list = updatefig_dataframe['row_index'].tolist()
        update_fig_index = []
        update_img_index = []
        update_for_event = {}
        for i, trace in enumerate(stacked_fig.data):
            row_id = trace['customdata'][0][3]
            if trace['name'] != "-1" and row_id in Select_row_index:
                update_fig_index.append(i)
                if trace['customdata'][0][5] in event_list:
                    update_for_event[row_id] = i
        update_fig_index = sorted(update_fig_index, reverse=True)

        # for k, v in update_for_event.items():
        #     print(stacked_fig.data[v])

        for index, (i, row) in zip(update_fig_index, updatefig_dataframe.iterrows()):
            shape_2 = int(str(row['comment']) + str(row['outer_link']) + str(row['news']), 2)

            event = "comment"
            if shape_2 == 0:
                event = "post"
            elif shape_2 == 4:
                event = "comment"
            elif shape_2 == 2:
                event = "external link"
            elif shape_2 == 1:
                event = "news"
            elif shape_2 == 5:
                event = "comment&news"
            elif shape_2 == 3:
                event = "external&news"

            code_id = row['code_id']  
            percent = row['percent']
            picture_number = row['picture_number']
            color = row['color']
            detect_time = row['detect_time']
            qid = row['qid']
            row_index = row['row_index']

            if int(picture_number) not in update_img_index:
                update_img_index.append(int(picture_number))
            #         # scatter = go.Scatter(name="", x=[int(picture_number)+0.4], y=[y_value],
            #         #     mode='markers', 
            #         #     marker_color='rgba(0,0,0,0)',
            #         #     # marker_symbol=mark_symbol[shape_2], 
            #         #     marker_size=7,
            #         #     customdata=[[event, code_id, row_index, picture_number]],
            #         #     hovertemplate="<br>".join([
            #         #         "Event=%{customdata[0]}",
            #         #         "post_number=%{customdata[1]}",
            #         #         "row index=%{customdata[2]}", 
            #         #         "picture number=%{customdata[3]}",                            
            #         #         ])
            #         #     )
            #         # stacked_fig.add_trace(scatter)
            
            # print("---") # stacked bar用picture number來index不準
            stacked_fig.data[index].update(
                name=str(code_id),
                y=[percent],
                x=[int(picture_number)],
                marker=dict(color=color),
                offset=0,
                customdata=[[code_id, detect_time, qid, row_index, picture_number, event]],
                # marker_pattern_shape=pattern_sahpe[shape],
                hovertemplate="<br>".join([
                    "post_number=%{customdata[0]}",
                    "detect time=%{customdata[1]}",
                    "questionnaire id=%{customdata[2]}",
                    "percent=%{y}",
                    "row index=%{customdata[3]}", 
                    "picture number=%{customdata[4]}",  
                    "event=%{customdata[5]}"                          
                    ])
            )

        updatefig_dataframe = stacked_dataframe[stacked_dataframe['picture_number'].isin(update_img_index)]
        x_dict = defaultdict(list)
        for i, row in updatefig_dataframe.iterrows():
            shape_2 = int(str(row['comment']) + str(row['outer_link']) + str(row['news']), 2)
            row_index = row['row_index']
            picture_number = row['picture_number']
            x_dict[int(picture_number) + 0.4].append(shape_2)
        has_exist = []
        for x_axis, shape_events in x_dict.items():
            event = FindImgEvent(shape_events)
            # print(x_axis, event)
            event_candidate_index = []
            for j, layout_img in enumerate(stacked_fig.layout['images']):
                if layout_img['x'] == x_axis: 
                    if j not in event_candidate_index:
                        event_candidate_index.append(j)
                    if x_axis not in has_exist:
                        has_exist.append(x_axis)
            event_candidate_index = sorted(event_candidate_index)
            if x_axis in has_exist:
                for j in event_candidate_index:
                    stacked_fig.layout['images'][j].visible = False
                if event != 'post':
                    if len(event_candidate_index) == 1:
                        stacked_fig.layout['images'][event_candidate_index[0]].source = source[event]
                        stacked_fig.layout['images'][event_candidate_index[0]].visible = True
                        if event == "comment&news" or event == "external&news":
                            stacked_fig.add_layout_image(
                                x=x_axis,
                                y=1.25,
                                source=source["news"],
                                xref="x",
                                yref="y",
                                sizex=0.6,
                                sizey=0.6,
                                xanchor="center",
                                yanchor="middle",
                                sizing="contain",
                                layer="above",
                                visible=True,
                            )
                    else:
                        stacked_fig.layout['images'][event_candidate_index[0]].source = source[event]
                        stacked_fig.layout['images'][event_candidate_index[0]].visible = True
                        if event == "comment&news" or event == "external&news":
                            stacked_fig.layout['images'][event_candidate_index[-1]].source = source["news"]
                            stacked_fig.layout['images'][event_candidate_index[-1]].visible = True
            else:
                stacked_fig.add_layout_image(
                        x=x_axis,
                        y=1.1,
                        source=source[event],
                        xref="x",
                        yref="y",
                        sizex=0.6,
                        sizey=0.6,
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        layer="above",
                        visible=True,
                    )
                if event == "comment&news" or event == "external&news":
                    stacked_fig.add_layout_image(
                        x=x_axis,
                        y=1.25,
                        source=source["news"],
                        xref="x",
                        yref="y",
                        sizex=0.6,
                        sizey=0.6,
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        layer="above",
                        visible=True,
                    )
        if stacked_layout is None:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )
        elif 'xaxis.range' in stacked_layout:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]],
                    type="linear"
                )
            )
        else:
            stacked_fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    range = None,
                    type="linear"
                )
            )

        timestamp = calendar.timegm(time.gmtime())
        stacked_dataframe.to_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file  + "/" + global_user + "/_" + str(timestamp) + ".csv", encoding='utf_8_sig', index=False)
        # if stacked_layout is None:
        #     stacked_fig = draw_barchart(stacked_dataframe, None)
        # elif 'xaxis.range' in stacked_layout:
        #     stacked_fig = draw_barchart(stacked_dataframe, [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]])
        # else:
        #     stacked_fig = draw_barchart(stacked_dataframe, None)
        HistoryData, columns = RecordHistory(global_user, Select_row_index, button_name)
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
            
            if stacked_layout is None:
                stacked_fig = draw_barchart(stacked_dataframe, None)
            elif 'xaxis.range' in stacked_layout:
                stacked_fig = draw_barchart(stacked_dataframe, [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]])
            else:
                stacked_fig = draw_barchart(stacked_dataframe, None)
            
            if os.path.exists(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json"): 
                with open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json") as f:
                    history = json.load(f)
                    if global_user in history:
                        history_user = history[global_user]
                        step_number = history_user[0]["Correction step"]
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

        stacked_fig = draw_barchart(stacked_dataframe, None)

        if len(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "")) == 0:
            timestamp = calendar.timegm(time.gmtime())
            stacked_dataframe.to_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file  + "/" + global_user + "/_" + str(timestamp) + ".csv", encoding='utf_8_sig', index=False)
        
        print(msg)

        if os.path.exists(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json"): 
            f = open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json")
            history = json.load(f)
            columns = [{'name': col, 'id': col} for col in history_column]
            if global_user in history:
                return msg, stacked_fig, global_user, history[global_user], columns
            else:
                return msg, stacked_fig, global_user, [{}], columns
        else:
            print("No history")
            columns = [{'name': col, 'id': col} for col in history_column]
            return msg, stacked_fig, global_user, [{}], columns
    elif 'Output_file' in changed_id:
        msg = global_user + " has been downloaded"
        return msg, dash.no_update, global_user, dash.no_update, dash.no_update
    else:
        msg = 'Not click any button yet'
        return msg, dash.no_update, global_user, dash.no_update, dash.no_update

@callback(
    Output('visual_placeholder', 'children'),
    Input('scatter-graph', 'relayoutData')
)
def RecordScatterLayout(scatter_relayout):
    global scatter_layout
    if scatter_relayout is not None:
        if 'autosize' not in scatter_relayout: 
            scatter_layout = scatter_relayout

    return dash.no_update

@callback(
    Output('download_file', 'data'),
    Input('users-dropdown', 'value'),
    Input('Output_file', 'n_clicks'),
    prevent_initial_call=True,
)
def DownloadClick(uid, file_btn):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if "Output_file" in changed_id:
        stacked_file_path = sorted(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user))[-1]
        
        try:
            stacked_dataframe = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + stacked_file_path, encoding="utf_8_sig")
        except:
            stacked_dataframe = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + stacked_file_path, engine='python')
        # real data
        stacked_dataframe = stacked_dataframe.drop(stacked_dataframe[(stacked_dataframe.code_id == -2) | (stacked_dataframe['visible time'] == -1)].index)
        stacked_dataframe = stacked_dataframe.drop(columns=['percent', 'row_index', 'picture_number', 'color'])
        scatter_dataframe = pd.DataFrame(columns=['user', 'qid', 'images', 'pid', 'context', 'visible time', 'post_number', 'comment', 'outer_link', 'discuss', 'discuss_reason'])
        stacked_dataframe.code_id = stacked_dataframe.code_id.astype(int)
        postID = stacked_dataframe['code_id'].unique()
        for _id in postID: #把相同code id的接成同一列               
            post = stacked_dataframe[stacked_dataframe['code_id'] == _id] 
            comment = 1 if 1 in post['comment'].tolist() else 0
            link =  1 if 1 in post['outer_link'].tolist() else 0
            insert_list = [post['user'].iloc[0], post['qid'].iloc[0], '\n'.join(post['images'].tolist()), post['pid'].iloc[0], post['context'].iloc[0], len(post), _id, comment, link, post['discuss'].iloc[0], post['discuss_reason'].iloc[0]]
            scatter_dataframe = scatter_dataframe.append(pd.DataFrame([insert_list], columns=scatter_dataframe.columns))
        scatter_dataframe['app'] = scatter_dataframe[image_appeared].apply(lambda x:extract_app_name(x))   
        scatter_dataframe['qid'] = scatter_dataframe["qid"].apply(lambda x:int(x))                
        scatter_dataframe['detect_time'] = scatter_dataframe[image_appeared].apply(lambda x:extract_time_from_answer(x))
        scatter_dataframe['date'] = scatter_dataframe['detect_time'].apply(lambda x:x.split(" ")[0])
        output_df = copy.deepcopy(scatter_dataframe)
        output_df = output_df.drop(columns=Extra_column)       
        output_df = output_df.drop(output_df.loc[output_df['visible time'] == -1].index, inplace=False)
        output_df = output_df.reset_index(drop=True)

        return dcc.send_data_frame(output_df.to_excel, global_user + "_PostCodingData.xlsx", sheet_name="Sheet1")

@callback(
    Output('post_content', 'children'),
    [Input('stacked-bar', 'hoverData')],
    [Input('users-dropdown', 'value')],
    Input('button_result', 'children'),
    prevent_initiall_call=True)
def update_content(stacked_hover, userid, result_btn):
    global global_user

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'users-dropdown' in changed_id:
        return dash.no_update
    if stacked_hover is None:
        return dash.no_update
    result = []
    
    if 'stacked-bar' in changed_id:
        hoverData = stacked_hover
        if hoverData['points'][0]['customdata'][0] in event_list:
            row_index = int(hoverData['points'][0]['customdata'][2])
        else:
            row_index = int(hoverData['points'][0]['customdata'][3])

        if global_user == None or global_user == 0:
            return dash.no_update

        stacked_file_path = sorted(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user))[-1]
        try:
            cache = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + stacked_file_path, encoding="utf_8_sig")
        except:
            cache = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + stacked_file_path, engine='python')
        content_split = cache.loc[cache['row_index'] == row_index]
        if content_split.empty:
            return dash.no_update
        content_split = str(content_split['context'].tolist()[0]).split('\n')
        for i, sentence in enumerate(content_split):
            result.append(sentence)
            if i != len(content_split) - 1:
                result.append(html.Br())  
        return result
    else:
        return dash.no_update

@callback(
    Output('post_num', 'children'),
    Input('stacked-bar', 'hoverData')
    )
def whichPost(stacked_hover):
    if stacked_hover is None:
        return dash.no_update
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'stacked-bar' in changed_id:
        if stacked_hover['points'][0]['customdata'][0] in event_list:
            post_number = int(stacked_hover['points'][0]['customdata'][1])
        else:
            post_number = int(stacked_hover['points'][0]['customdata'][0])
        return "Post number: " + str(post_number)
    else:
        return dash.no_update
    
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
