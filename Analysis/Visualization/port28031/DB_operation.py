from http import client
import dash
import dash_auth
from dash import callback_context, Dash, dcc, html, Input, Output, callback, dash_table
from dash.dependencies import State
import plotly.graph_objs as go
import pandas as pd
import flask
import datetime
import calendar
import time
from collections import defaultdict
from textwrap import dedent as s
from PIL import Image
import copy
import os
import cv2
import base64
import json
from pymongo import MongoClient
from bson.json_util import dumps, loads
from pandas import json_normalize

ROOT_PATH = '/home/ubuntu/News Consumption/'
data_path = ROOT_PATH + "Analysis/Visualization/data/"
client = MongoClient("mongodb://Hua:Wahahaha443@127.0.0.1:27017/newscons?authSource=admin")
mydb = client['visualtool']
history_column = ["Time", "Correction step", "Row number", "Button click"]
columns = [{'name': col, 'id': col} for col in history_column]

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

def SaveDataframeToDB(collection_name, stacked_dataframe, user):
    now = datetime.datetime.now()
    current_time = now.strftime("%Y/%m/%d %H:%M:%S")
    collection = mydb[collection_name]
    data_dict = stacked_dataframe.to_dict("records")
    collection.insert_one({"User": user, "data": data_dict, "Time": current_time})
    # collection.update_one(
    #     {"$and":[{"type": "data"}, {"User":user}]},
    #     {"$set":
    #         {"time":current_time, "type": "data", "data": data_dict, "User": user}
    #     },
    #     upsert=True
    # )

def GetDataframeFromDB(collection_name, user):
    collection = mydb[collection_name]
    data_count = collection.count_documents({"User": user})
    if data_count != 0:
        record = loads(dumps(list(collection.find({"$query": {"User": user}, "$orderby": {"Time" : -1}}))))
        stacked_dataframe = json_normalize(record[0]['data'])
    else:
        try:
            scatter_df = pd.read_csv(data_path + user + ".csv", encoding="utf_8_sig")
        except:
            scatter_df = pd.read_csv(data_path + user + ".csv", engine='python')
        # real data
        scatter_df = scatter_df.drop(scatter_df[scatter_df.code_id == -2].index)
        scatter_df = scatter_df.drop(columns=['n_main', 'keyword', 'picture_number'])
        scatter_df['code_id'] = scatter_df['code_id'].apply(lambda x:int(x) + 1)
        scatter_df.reset_index(inplace=True,drop=True)

        stacked_dataframe = copy.deepcopy(scatter_df)
        stacked_dataframe['detect_time'] = stacked_dataframe['images'].apply(lambda x:extract_time_from_answer(x))
        stacked_dataframe['row_index'] = stacked_dataframe.index
        stacked_dataframe['visible time'] = 0
        stacked_dataframe['discuss'] = 0
        stacked_dataframe['discuss_reason'] = " "
        stacked_dataframe = BarChart_Preprocessing(stacked_dataframe)
    return stacked_dataframe

def SaveHistoryToDB(collection_name, user, row_index_list, action):
    collection = mydb[collection_name]
    data_count = collection.count_documents({"User": user})
    now_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    Record_table = []
    if data_count != 0:
        record = collection.find({"$query": {"User": user}, "$orderby": {"Time" : -1}})
        Record_table = list(record)
        # print(Record_table)
        last_record = Record_table[0]
        current_step = int(last_record["Correction step"]) + 1
        data = {"User": user, "Time": now_time, "Correction step": str(current_step), "Row number": ",".join(str(e) for e in row_index_list), "Button click": action}
    else:
        data = {"User": user, "Time": now_time, "Correction step":"1", "Row number": ",".join(str(e) for e in row_index_list), "Button click": action}
    collection.insert_one(data)
    Record_table.insert(0, data)
    for record in Record_table:
        del record['User']
        del record['_id']
    return Record_table, columns

def GetHistoryFromDB(collection_name, user):
    collection = mydb[collection_name]
    data_count = collection.count_documents({"User": user})
    Record_table = []
    if data_count != 0:
        record = collection.find({"$query": {"User": user}, "$orderby": {"Time" : -1}})
        Record_table = list(record)
        for record in Record_table:
            del record['User']
            del record['_id']
        return Record_table, columns
    else:
        return [{}], columns
    