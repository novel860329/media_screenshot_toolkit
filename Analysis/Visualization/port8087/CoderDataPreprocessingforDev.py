import pandas as pd
import numpy as np
import time
from collections import defaultdict
import os
import datetime

dateFormatter = "%Y-%m-%d %H-%M-%S"
dateFormatter_timeZone = "%Y/%m/%d %H:%M:%S +0800"
comment_threshold = 4
column_name = ['user', 'images', 'typing', 'like', 'share']
init_data = [0 for x in range(len(column_name))]
TakeWhichApps = ['com.facebook.katana', 'com.instagram.android']

try:
    UserDataframe = pd.read_csv("D:/Users/MUILab-VR/Desktop/News Consumption/Analysis/UserProfile/UserProfileForCode.csv", encoding="utf_8_sig")
except:
    UserDataframe = pd.read_csv("D:/Users/MUILab-VR/Desktop/News Consumption/Analysis/UserProfile/UserProfileForCode.csv", engine='python')

user_id = []
Users = []
First_day = []
End_day = []

root_path = "D:/Users/MUILab-VR/Desktop/News Consumption/Database/SplitPostFile/"
all_file = os.listdir(root_path)
for file in all_file:
    user = file.split("-")[0]
    if user not in user_id:
        user_id.append(user)
        Users.append(UserDataframe[UserDataframe['User ID'] == user].iloc[0]['Device ID'])
        First_day.append(UserDataframe[UserDataframe['User ID'] == user].iloc[0]['實驗第0天'])
        End_day.append(UserDataframe[UserDataframe['User ID'] == user].iloc[0]['實驗最後一天'])

def Data_init(user, images):
    result = {}
    for col in column_name:
        if col == "user":
            result[col] = user
        elif col == "images":
            result[col] = images
        else:
            result[col] = 0
    return result

def ConvertToAction(ImageName):
    if pd.isnull(ImageName):
        return ImageName
        
    img_name = str(ImageName)
    img_date = img_name.split("/")[0]
    img_time = img_name.split("/")[1][0:8]
    timestamp = str(int(time.mktime(datetime.datetime.strptime(img_date + " " + img_time, "%Y-%m-%d %H-%M-%S").timetuple())) * 1000)
    return timestamp + "-" + img_name.replace("/", "-")
 
def data_preprocess():
    user_index = 0
    
    Action_df = pd.DataFrame(columns = column_name)
    for user in Users:
        print("Initial", user_id[user_index])
        if not os.path.isdir("D:/Users/MUILab-VR/Desktop/News Consumption/"  + user_id[user_index]+ "/" + user + "/PhoneStateData/"):
            print(user_id[user_index], "not exist")
            user_index += 1
            continue
        path = "D:/Users/MUILab-VR/Desktop/News Consumption/"  + user_id[user_index]+ "/" + user + "/PhoneStateData/" + user +  "_myAccessibility.csv"        
        try:
            accessibility_data = pd.read_csv(path, usecols=['PackageName', 'EventText', 'Extra', 'EventType', 'ImageName', 'detect_time'], encoding="utf-8")
        except:
            accessibility_data = pd.read_csv(path, usecols=['PackageName', 'EventText', 'Extra', 'EventType', 'ImageName', 'detect_time'], engine="python")
            
        accessibility_data = accessibility_data[accessibility_data['PackageName'].isin(TakeWhichApps)].reset_index(drop=True)

        for index in range(len(accessibility_data)):
            action = False

            ### FB action
            if not pd.isnull(accessibility_data.loc[index, 'ImageName']) and accessibility_data.loc[index, 'PackageName'] == 'com.facebook.katana':
                text = str(accessibility_data.loc[index, 'EventText'])
                event = accessibility_data.loc[index, 'EventType']
                img = ConvertToAction(accessibility_data.loc[index, 'ImageName'])
                row_df = Data_init(user_id[user_index], img)
                
                ### 判斷有沒有打字
                if event == 'TYPE_VIEW_TEXT_CHANGED':
                    row_df['typing'] = 1
                    action = True
                
                ### 按讚                
                if (text == "讚" and event == 'TYPE_VIEW_CLICKED') or (text == "選擇心情" and event == 'TYPE_ANNOUNCEMENT') or ("讚按鈕，已按下" in text and event == 'TYPE_VIEW_CLICKED') or ("讚按鈕，已按下" in text and event == 'TYPE_WINDOW_STATE_CHANGED'):
                    row_df['like'] = 1
                    action = True
                    
                ### 分享   
                if "分享" in text and event == 'TYPE_VIEW_CLICKED':
                    row_df['share'] = 1
                    action = True
            elif not pd.isnull(accessibility_data.loc[index, 'ImageName']) and accessibility_data.loc[index, 'PackageName'] == 'com.instagram.android':
                if event == 'TYPE_VIEW_TEXT_CHANGED':
                    row_df['typing'] = 1
                    action = True
                
                ### 按讚                
                if text == "讚" and event == 'TYPE_VIEW_CLICKED':
                    row_df['like'] = 1
                    action = True
            
            if action and img not in Action_df['images'].tolist():
                Action_df = Action_df.append(row_df, ignore_index=True)
        user_index += 1
    
    for file in all_file:    
        try:
            machine_data = pd.read_csv(root_path + file, encoding="utf-8")
        except:
            machine_data = pd.read_csv(root_path + file, engine="python")
        
        for col in column_name[2:len(column_name)]:
            machine_data[col] = 0

        for index in range(len(machine_data)):
            user = machine_data.loc[index, 'user']
            image = machine_data.loc[index, 'images']
   
            if not Action_df.loc[(Action_df['user'] == user) & (Action_df['images'] == image)].empty:
                ### update three event on mahcine dataframe
                for col in column_name[2:len(column_name)]:
                    machine_data.loc[index, col] = Action_df[(Action_df['user'] == user) & (Action_df['images'] == image)].iloc[0][col]
        machine_data.to_csv("D:/Users/MUILab-VR/Desktop/News Consumption/Analysis/Visualization/data/" + file, index=None, encoding='utf_8_sig')
    print("Complete !!")