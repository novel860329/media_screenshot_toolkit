import pandas as pd
import datetime
import calendar
import time
import copy
import os
import json


ROOT_PATH = "D:/Users/MUILab-VR/Desktop/News Consumption/CHI2022/media_screenshot_toolkit/"
history_column = ["Time", "Correction step", "Row number", "Button click"]
columns = [{'name': col, 'id': col} for col in history_column]
data_path = ROOT_PATH + "Analysis/Visualization/data/"

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

def GetDataframe(global_user, port_file):
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
        stacked_dataframe['repeat'] = 0
        stacked_dataframe['typing'] = 0
        stacked_dataframe['like'] = 0
        stacked_dataframe['share'] = 0
        stacked_dataframe['detect_time'] = stacked_dataframe['images'].apply(lambda x:extract_time_from_answer(x))
        stacked_dataframe['row_index'] = stacked_dataframe.index
        stacked_dataframe['visible time'] = 0
        stacked_dataframe['discuss'] = 0
        stacked_dataframe['discuss_reason'] = " "
        stacked_dataframe = BarChart_Preprocessing(stacked_dataframe)
    return stacked_dataframe

def SaveDataframe(global_user, port_file, stacked_dataframe):
    timestamp = calendar.timegm(time.gmtime())
    stacked_dataframe.to_csv(ROOT_PATH + "/Analysis/Visualization/" + port_file  + "/" + global_user + "/_" + str(timestamp) + ".csv", encoding='utf_8_sig', index=False)

def GetHistory(global_user, port_file):
    if os.path.exists(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json"): 
        f = open(ROOT_PATH + "/Analysis/Visualization/" + port_file + "/CorrectionHistory.json")
        history = json.load(f)
        if global_user in history:
            return history[global_user], columns
        else:
            return [{}], columns
    else:
        return [{}], columns

def SaveHistory(user, port_file, row_index_list, action):
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
    return history[user], columns