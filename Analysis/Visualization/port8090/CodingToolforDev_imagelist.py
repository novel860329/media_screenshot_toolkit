# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:46:32 2022

@author: MUILab-VR
"""

import code
import dash
import dash_auth
from dash import callback_context, Dash, dcc, html, Input, Output, callback, dash_table
import pandas as pd
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
    temp = answer.split("-")
    date = temp[1] + "-" + temp[2] + "-" + temp[3]
    time = temp[4] + ":" + temp[5] + ":" + temp[6]
    return date + " " + time


#----- config ------
ROOT_PATH = 'D:/Users/MUILab-VR/Desktop/News Consumption/'
data_path = ROOT_PATH + "Analysis/Visualization/data/"

global_user = 0
pattern_sahpe = ['', '.', 'x', '|']
event_list = ["external link", 'comment', "news", "external&news", "comment&news"]
mark_symbol = ['', 'x', 'circle', 'bowtie', 'diamond', 'hourglass', 'square', 'hexagram']
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
port_number = 8090
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
        html.Div(id='img-content', style={"margin-bottom": "15px", "width": "100%", "overflow-x": "scroll", 'display' : 'flex', 'flex-direction': 'row'})
    ,style={ 'width': '100%','margin-top': '10px'}),   

    html.Div([
        html.Div(
            dash_table.DataTable(id='correct_history',
            style_cell={
                'overflow-y': 'hidden',
                'textOverflow': 'ellipsis',
                'fontSize':16,
                'minWidth': '50px', 'width': '50px', 'maxWidth': '50px',
                # 'maxHeight': '50px',
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

@callback(
    [Output('button_result', 'children'),
    Output('img-content', 'children'),
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
    Input('News_button', 'n_clicks')])
def update_image(uid, discuss_text, merge_btn, delete_btn, split_btn, discuss_btn, file_btn, recovery_btn, comment_btn, click_btn,  news_btn):
    global global_user, stacked_layout, prev_selection, Select_row_index

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if (uid == '' or uid is None) and global_user == 0: #最一開始開這個網頁的時候
        return "Choose one user number", dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif (uid == '' or uid is None) and global_user != 0 and global_user != "": #從visual mode回來的時後
        msg = 'Select some bars to start'
        stacked_file_path = sorted(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user))[-1]

        try:
            stacked_dataframe = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + stacked_file_path, encoding="utf_8_sig")
        except:
            stacked_dataframe = pd.read_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "/" + stacked_file_path, engine='python')

        ### read image and display
        
        if os.path.exists(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json"): 
            f = open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json")
            history = json.load(f)
            columns = [{'name': col, 'id': col} for col in history_column]
            if global_user in history:
                return msg, dash.no_update, global_user, history[global_user], columns
            else:
                return msg, dash.no_update, global_user, [{}], columns
        else:
            columns = [{'name': col, 'id': col} for col in history_column]
            return msg, dash.no_update, global_user, [{}], columns

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

    if 'users-dropdown' in changed_id:
        print("users-dropdown")
        msg = 'Select some bars to start'

        ### read image and display
        result = []
        images = list(dict.fromkeys(stacked_dataframe['images'].tolist()))
        for i, image in enumerate(images):
            q_id = stacked_dataframe[stacked_dataframe['images'] == image].iloc[0]['qid']
            uid = uid.split("-")[0] if "-" in uid else uid
            temp_path = os.listdir(ROOT_PATH + "/" + uid)
            img_cv2 = cv2.imread(ROOT_PATH + uid + "/" + temp_path[0] + "/NewInterval/" + str(q_id) + "/" + image)
            height = img_cv2.shape[0]
            width = img_cv2.shape[1]
            img_cv2 = draw_visible_area(img_cv2)
            percentage = stacked_dataframe.loc[stacked_dataframe['images'] == image]['percent'].tolist()
            colors = stacked_dataframe.loc[stacked_dataframe['images'] == image]['color'].tolist()

            end_point = 0.13
            j = len(colors) - 1
            area = []
            for percent, color in zip(percentage, colors):
                start_point = end_point
                end_point += round(percent, 2)
                c = re.split(r'[(,)]', color)
                c = (int(c[3]), int(c[2]), int(c[1]))
                img_cv2 = cv2.rectangle(img_cv2, (0, int(start_point*height) - 10*j), (width, int(end_point*height) - 10*j), c, 10)
                coords = str(0) + "," + str(int(start_point*height) - 10*j) + "," + str(width) + str(int(end_point*height) - 10*j)
                area.append(html.Area(target='', alt='otimizar', title='otimizar', href='/img_click/' + str(i + 1) + "&" + coords, coords=coords, shape='rect'))
                j -= 1
            _, buffer = cv2.imencode('.jpg', img_cv2)
            img_64 = base64.b64encode(buffer).decode('utf-8')
            
            result.append(
                html.MapEl(
                    area, name='map' + str(i))
            )    
            result.append(
                html.Img(id= "img"+str(i), src='data:image/jpg;base64,{}'.format(img_64), useMap='#map' + str(i), style={'width': '10%', 'margin-left': '10px', 'margin-bottom': '20px', 'vertical-align':'top'})
            )
        if len(os.listdir(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/" + global_user + "")) == 0:
            timestamp = calendar.timegm(time.gmtime())
            stacked_dataframe.to_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file  + "/" + global_user + "/_" + str(timestamp) + ".csv", encoding='utf_8_sig', index=False)
        
        print(msg)
        if os.path.exists(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json"): 
            f = open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json")
            history = json.load(f)
            columns = [{'name': col, 'id': col} for col in history_column]
            if global_user in history:
                return msg, result, global_user, history[global_user], columns
            else:
                return msg, result, global_user, [{}], columns
        else:
            print("No history")
            columns = [{'name': col, 'id': col} for col in history_column]
            return msg, result, global_user, [{}], columns
    else:
        return "Not User dropdown", dash.no_update, dash.no_update, dash.no_update, dash.no_update

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

#Update the index
@callback(Output('page-content', 'children'),
        [Input('url', 'pathname')])
def index_page(pathname):
    print("index page name:", pathname)
    return coding_layout

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=port_number, threaded=True, debug=True)
