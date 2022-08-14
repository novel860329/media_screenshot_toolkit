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
    date = temp[1] + "/" + temp[2] + "/" + temp[3]
    time = temp[4] + ":" + temp[5] + ":" + temp[6]
    return date + " " + time

#----- config ------
# ubuntu path = "/home/ubuntu/News Consumption/"
# windows path = "D:/Users/MUILab-VR/Desktop/News Consumption/"
ROOT_PATH = "D:/Users/MUILab-VR/Desktop/News Consumption/CHI2022/media_screenshot_toolkit/"
data_path = ROOT_PATH + "Analysis/Visualization/data/"

# data_preprocess()

global_user = 0
metadata_table_col = ["Picture_no", "Time", "OCR result", "Event"]
event_col = ['share', 'like', 'typing', 'news', 'comment', 'outer_link']
event_map = {0: 'external link', 1: 'comment', 2: 'news', 3: 'typing', 4: 'like', 5: 'share'}
Button_dict = {'Comment_button': ('comment', "Comment"), 'Click_button': ('outer_link', "External link"),
                'Typing_button': ('typing', 'Typing'), 'News_button': ('news', 'News'), 'Like_button': ('like', 'Like'), 
                'Share_button': ('share', 'Share')}
Event = ["Comment", "External link", "News", "Typing", 'Like', 'Share']
source = {'post':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png"),
        'news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/news.png"),
        'external link':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/link.png"),
        'comment':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png"),
        'typing':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/typing.png"),
        'like':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/like.png"),
        'share':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/share.png")}
scatter_layout = []
stacked_layout = []
prev_selection = {}
history_column = ["Time", "Correction step", "Row number", "Button click"]
description = '''
* **We use different background color to represent apps**
    1. Blue tone : Facebook
    2. Red tone : Youtube
    3. Purple : Instagram
* **Different mark on the bar represent different event and you can hover on it to see what evnet it stand for**
'''

User_ID = "user"
questionnaire_id = "qid"
image_appeared = "images"
image_content = "context"
visible_time = "visible time"
port_number = 28025
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

# from flask_caching import Cache
# cache = Cache(app.server, config={
#     # try 'filesystem' if you don't want to setup redis
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': ROOT_PATH + "Analysis/Visualization/port" + str(port_number) + "/cache"
# })
# app.config.suppress_callback_exceptions = True
# timeout = 60

# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )

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
        html.Div([
            dcc.Dropdown(
                id='users-dropdown',
                options=[{'label': i, 'value': i} for i in User_data],
                value=""
            )
        ], style={'width': '7%', 'display': 'inline-block', 'vertical-align':'top','margin-left': '10px', 'margin-top': '10px'}),

        html.Div([
            html.Button('Combine', id='Merge_button', n_clicks=0, style={"margin-left": "10px"}),
            html.Button('Delete', id='Delete_button', n_clicks=0, style={"margin-left": "10px"}),
            html.Button('Split', id='Split_button', n_clicks=0, style={"margin-left": "10px"}),
            html.Button('News', id='News_button', n_clicks=0, style={"margin-left": "10px"}),
        ], style={'display': 'inline-block', 'vertical-align':'top', "margin-left": "10px", 'margin-top': '10px'}),

        html.Div([
            html.P("Discuss Reason：", style={'fontSize':18}),
            dcc.Input(
                id='Discuss_text',
                type='text',
                style={'width':'20%'}
            ),
            html.Button('Record', id='Discuss_button', n_clicks=0, style={"margin-left": "10px"}),
            html.Button('Previous Step', id='Recovery_button', n_clicks=0, style={"margin-left": "20px"}),
            dcc.Download( id="download_file"),
            html.Button('Output File', id='Output_file', n_clicks=0, style={"margin-left": "10px"}), 
        ], style={'display': 'flex', 'flex-direction': 'row','margin-left': '10px', 'margin-top': '10px'}),           
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    
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
    ], open=True, style={'display': 'inline-block', 'vertical-align':'top', "margin-left": "10px", 'margin-top': '5px'}),

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
        html.B(id="button_result", style={'fontSize':16, "margin-left": "10px"}),
        style={'display': 'inline-block', 'vertical-align':'top', "margin-left": "10px", 'margin-top': '5px'},
    ),

    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(
            dcc.Graph(
                id='stacked-bar',
                config={"displayModeBar": False}
                # style={'height': '70vh'}
            )
        ,style={ 'width': '100%','margin-top': '10px'}),    
    ),

    html.Div(
       html.Div([
                html.B("Images Metadata", id="post_num", style={'fontSize':16}),
            ])
        , style={'width': '20%', 'display': 'inline-block', 'margin-left': '10px', 'margin-top': '20px'}
    ),
    
    html.Div([
        html.Div(
            dash_table.DataTable(id='hover_metadata',
            data=[{metadata_table_col[0]: "", metadata_table_col[1]:"", metadata_table_col[2]: "", metadata_table_col[3]: ""} for i in range(5)],
            columns=[{"name": i, "id": i} for i in metadata_table_col],
            style_cell={
                # 'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'fontSize':14,
                # 'minWidth': '20px', 'width': '20px', 'maxWidth': '50px',
                'maxWidth': '50px',
                'textAlign': 'center'
            },
            css=[{
                'selector': '.dash-spreadsheet td div',
                'rule': '''
                    line-height: 15px;
                    max-height: 45px; min-height: 45px; height: 45px;
                    display: block;
                    overflow-y: hidden;
                    margin-bottom: 5px;
                    margin-top: 5px;
                    font-weight: bold;
                    font-family: sans-serif;
                '''
            },{'selector': '.dash-table-tooltip', 'rule': "white-space: pre-wrap;"}],
            style_table={'minWidth': '100%', 'width': '100%', 'maxWidth': '100%', 'maxHeight': '600px'},
            style_data={
                'whiteSpace': 'normal'
            },
            tooltip_delay=0,
            tooltip_duration=None,
            style_data_conditional=[{'if': {'row_index': 2}, 'color': 'tomato'},
                {'if': {'column_id': metadata_table_col[0]},'width': '20%'},
                {'if': {'column_id': metadata_table_col[1]},'width': '15%'},
                {'if': {'column_id': metadata_table_col[3]},'width': '15%'}], 
            )
        , style={'width': '30%', 'margin-left': '10px', 'display': 'inline-block'}),
        # html.Div(html.P("", id = "post_content"), style={'width': '15%', "height": "60vh", "overflow": "scroll", 'margin-left': '10px', 'display': 'flex', 'flex-direction': 'row', 'flex':1, "margin-bottom": "15px",'fontSize':18}),
        html.Div(id='img-content', style={"margin-bottom": "15px", "overflowY": "scroll", 'flex':5, 'text-align':'center', 'maxHeight': '600px'})
    ]
    ,style={'display': 'flex', 'flex-direction': 'row'}),  

    html.Div([
            html.B("Correct History", style={'fontSize':16}),
        ]
    , style={'width': '20%', 'display': 'inline-block', 'margin-left': '10px', 'margin-top': '20px'}),
    
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
        , style={'width': '100%', 'margin-left': '10px', 'display': 'inline-block'})]
        # html.Div(dcc.Markdown(s(description)), style={'width': '50%', "height": "500px", 'margin-left': '20px', 'display': 'inline-block', 'fontSize': 20})]
    ,style={'display': 'flex', 'flex-direction': 'row',  "margin-bottom": "15px"}),  

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
        img = Image.open(ROOT_PATH + uid + "/" + temp_path[0] + "/NewInterval/" + str(q_id) + "/" + image)
        width, height = img.size
        y_size = height * image_width / width
        x_axis = picture_number + (int(picture_number) - 1) * image_width

        fig.add_layout_image(
            x=x_axis,
            y=y_size/2 - y_size*0.13,
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
            detect_time = extract_time_from_answer(scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['images'])
            qid = scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0]['qid']

            shape = ""
            for col in event_col:
                shape = shape + str(scatter_dataframe[scatter_dataframe['row_index'] == row_index].iloc[0][col])    
            
            shape = int(shape, 2) ### "6"

            shape_bin = bin(shape)[2:][::-1] ### [0,1,1]
            binary1_index = []
            for j, binary in enumerate(shape_bin):
                if binary == '1':
                    binary1_index.append(j) ### [1, 2]

            event = ""
            bin_index_len = len(binary1_index)
            if bin_index_len == 0:
                event = "post"
            for j in range(bin_index_len):
                if j == bin_index_len - 1:
                    event = event + event_map[binary1_index[j]]
                else:
                    event = event + event_map[binary1_index[j]] + "&"

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

    x_dict = defaultdict(list)
    for j, row in scatter_dataframe.iterrows():
        shape = ""
        for col in event_col:
            shape = shape + str(row[col])           
        shape = int(shape, 2) ### "6"
        picture_number = row['picture_number']
        x_dict[int(picture_number) + (int(picture_number) - 1) * image_width].append(shape)
    for x_axis, shape_events in x_dict.items():
        shape = FindImgEvent(shape_events)
        shape_bin = bin(shape)[2:][::-1]
        binary1_index = []
        for j, binary in enumerate(shape_bin):
            if binary == '1':
                binary1_index.append(j)

        if shape != 0:
            y_axis = y_size + 0.5 - y_size*0.13
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
        sliderrange = [0, 100]

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
    fig.update_yaxes(showticklabels=False, scaleanchor="x", showgrid=False)

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
    Output('img-content', 'children'),
    Output(component_id='hover_metadata', component_property='data'),
    Output("hover_metadata", "tooltip_data"),
    Input('stacked-bar', 'hoverData'),
    Input('users-dropdown', 'value'),
    [Input('button_result', 'children')])
def update_image(stacked_hover, userid, button_result):
    global global_user
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'users-dropdown' in changed_id:
        return dash.no_update, dash.no_update, dash.no_update

    if stacked_hover is None:
        return dash.no_update, dash.no_update, dash.no_update

    result = []
    if "button_result" in changed_id or 'stacked-bar' in changed_id:
        hoverData = stacked_hover
        picture_number = int(hoverData['points'][0]['customdata'][4])

    ### used for searching comment and outer_link value
    cache = GetDataframe(global_user, port_file)

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
    metadata_table = []
    tool_tip = []
    for i, pic_num in enumerate(range(picture_range[0], picture_range[1] + 1)):
        p_num += 1
        picture = cache[cache['picture_number'] == pic_num]
        img = picture.iloc[0]['images']
        q_id = picture.iloc[0]['qid']
        detect_time = extract_time_from_answer(img).split(" ")[1]

        ocr_result = ""
        x_dict = defaultdict(list)
        for j, (index, row) in enumerate(picture.iterrows()):
            if str(row['context']) == "nan":
                content = ""
            else:
                content = row['context'].strip()
            ocr_result += content
        
            shape = ""
            for col in event_col:
                shape = shape + str(row[col])           
            shape = int(shape, 2) ### "6"
            x_dict[pic_num].append(shape)
        event = ""
        for x_axis, shape_events in x_dict.items():
            shape = FindImgEvent(shape_events)
            shape_bin = bin(shape)[2:][::-1]
            binary1_index = []
            for k, binary in enumerate(shape_bin):
                if binary == '1':
                    binary1_index.append(k)
            for k in range(len(binary1_index)):
                if k == len(binary1_index) - 1:
                    event = event + event_map[binary1_index[k]]
                else:
                    event = event + event_map[binary1_index[k]] + "&"

        code_id = cache[(cache['picture_number'] == pic_num) & (cache['biggest'] == 1)].iloc[0]['code_id']
        userid = userid.split("-")[0] if "-" in userid else userid

        temp_path = os.listdir(ROOT_PATH + "/" + userid)
        if cache.loc[(cache["images"] == img) & (cache["code_id"] == code_id), "comment"].shape[0] == 0:
            return dash.no_update
        # print(ROOT_PATH + userid + "/" + temp_path[0] + "/NewInterval/" + str(q_id) + "/" + img)
        img_cv2 = cv2.imread(ROOT_PATH + userid + "/" + temp_path[0] + "/NewInterval/" + str(q_id) + "/" + img)
        # img_cv2 = draw_visible_area(img_cv2)
        _, buffer = cv2.imencode('.jpg', img_cv2)
        img_64 = base64.b64encode(buffer).decode('utf-8')

        if pic_num == picture_number:
            width = "18%"
        else:
            width = "15%"

        ocr_result = '     '.join(ocr_result.split("\n"))

        data = {metadata_table_col[0]: pic_num, metadata_table_col[1]: detect_time, metadata_table_col[2]: ocr_result, metadata_table_col[3]: event}
        metadata_table.append(data)
        tool_tip.append({c:{'type': 'text', 'value': str(v)} for c, v in data.items()})
        result.append(html.Img(id= "img"+str(i + bias), src='data:image/jpg;base64,{}'.format(img_64), style={'width': width, 'margin-right': '20px', 'vertical-align':'top'}))    
    if picture_range[0] == max_picture_number - 2:
        for i in range(3, 5):
            result.append(html.Img(id= "img"+str(i)))
    elif picture_range[0] == max_picture_number - 3:
        for i in range(4, 5):
            result.append(html.Img(id= "img"+str(i)))
    # print("-------------------------------------------------------------------")
    return result, metadata_table, tool_tip
    
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
    Input('Typing_button', 'n_clicks'),
    Input('Like_button', 'n_clicks'),
    Input('Share_button', 'n_clicks'),
    Input('stacked-bar', 'selectedData')],
    [State('stacked-bar', 'relayoutData')]
)
def ButtonClick(uid, discuss_text, merge_btn, delete_btn, split_btn, discuss_btn, file_btn, recovery_btn, comment_btn, click_btn,  news_btn, typing_btn, like_btn, share_btn, stacked_select, stacked_relayout): 
    global global_user, stacked_layout, stacked_fig, prev_selection, Select_row_index, width, height

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    if 'stacked-bar' in changed_id:
        bar_list = {}
        if stacked_select is None:
            return "Not select bar yet", dash.no_update, dash.no_update, dash.no_update, dash.no_update

        Select_row_index = []
        for bar in stacked_select['points']:
            bar_id = bar['curveNumber'] 
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
            sliderrange = [0, 100]
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
            sliderrange = [0, 100]
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

    elif 'Comment_button' in changed_id or 'Click_button' in changed_id or 'News_button' in changed_id or 'Typing_button' in changed_id \
        or 'Like_button' in changed_id or 'Share_button' in changed_id:
        if Select_row_index == []:
            return "Must select some bars", dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            for button, tup in Button_dict.items():
                if button in changed_id:
                    col, button_name = tup
                    msg = "Bars has been marked as " + button_name 

        old_value = stacked_dataframe[stacked_dataframe['row_index'] == Select_row_index[0]].iloc[0][col]
        new_value = old_value ^ 1
        
        for row_index in Select_row_index:
            if col == "outer_link":
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, col] = new_value
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, "comment"] = 0
            elif col == "comment":
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, col] = new_value
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, "outer_link"] = 0
            else:
                stacked_dataframe.loc[stacked_dataframe['row_index'] == row_index, col] = new_value
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
                elif action in Event:
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