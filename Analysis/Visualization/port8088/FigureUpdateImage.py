from collections import defaultdict
from PIL import Image

ROOT_PATH = 'D:/Users/MUILab-VR/Desktop/News Consumption/CHI2022/media_screenshot_toolkit/'

event_col = ['share', 'like', 'typing', 'news', 'comment', 'outer_link']
event_map = {0: 'external link', 1: 'comment', 2: 'news', 3: 'typing', 4: 'like', 5: 'share'}
source = {'post':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png"),
        'news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/news.png"),
        'external link':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/link.png"),
        'comment':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png"),
        'typing':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/typing.png"),
        'like':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/like.png"),
        'share':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/share.png")}

image_width = 5
def UpdateFigureLayout(stacked_fig, stacked_layout):
    if stacked_layout is None:
        figure_x = int(stacked_fig.layout['images'][0]['x']),
        stacked_fig.update_layout(
            xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.1),
            range = [0, 100],
            type="linear",
            scaleanchor="y",
            showgrid=False
        ),
        )
    elif 'xaxis.range' in stacked_layout:
        stacked_fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                range = [stacked_layout['xaxis.range'][0], stacked_layout['xaxis.range'][1]],
                type="linear",
                scaleanchor="y",
                showgrid=False
            )
        )
    else:
        print(stacked_fig.layout['images'])
        figure_x = int(stacked_fig.layout['images'][0]['x']),
        stacked_fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                range = [0, 100],
                type="linear",
                scaleanchor="y",
                showgrid=False
            )
        )
    return stacked_fig

def FindImgEvent(events):
    shape = 0
    
    for event in events:
        shape = shape | event

    return shape

def GetEvent(shape):
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
    return event

def CombineFigImageUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout):
    updatefig_dataframe = stacked_dataframe[stacked_dataframe['row_index'].isin(Select_row_index)]
    
    bar_list = {}
    update_fig_index = []
    for i, trace in enumerate(stacked_fig.data):
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
        shape = ""
        for col in event_col:
            shape = shape + str(row[col])

        event = GetEvent(shape)

        bar_list[index] = color

        stacked_fig.data[index].update(
            name=str(code_id),
            marker=dict(color="rgba(0,0,0,0)",line=dict(
                    color=color,
                    width=3.5
                )),
            customdata=[[code_id, detect_time, qid, row_index, picture_number, event, round(percent, 2)]],
            hovertemplate="<br>".join([
                "post_number=%{customdata[0]}",
                "detect time=%{customdata[1]}",
                "questionnaire id=%{customdata[2]}",
                "percent=%{customdata[6]}",
                "row index=%{customdata[3]}", 
                "picture number=%{customdata[4]}", 
                "event=%{customdata[5]}"                           
                ])
        )

    stacked_fig = UpdateFigureLayout(stacked_fig, stacked_layout)

    return stacked_fig, bar_list

def DeleteFigImageUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout, visible):
    updatefig_dataframe = stacked_dataframe[stacked_dataframe['row_index'].isin(Select_row_index)]

    update_fig_index = []
    update_for_event = defaultdict(list)
    for i, trace in enumerate(stacked_fig.data):
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
                stacked_fig.data[j].visible = visible
        for j, layout_img in enumerate(stacked_fig.layout['images']):
            if layout_img['x'] == int(picture_number) + (int(picture_number) - 1) * image_width:
                stacked_fig.layout['images'][j].visible=visible
        stacked_fig.data[index].visible = visible

    stacked_fig = UpdateFigureLayout(stacked_fig, stacked_layout)

    return stacked_fig

def SplitFigImageUpdate(stacked_fig, stacked_dataframe, post_number, stacked_layout):
    updatefig_dataframe = stacked_dataframe[stacked_dataframe['code_id'] >= int(post_number)]
    
    bar_list = {}
    update_fig_index = []
    for i, trace in enumerate(stacked_fig.data):
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

        shape = ""
        for col in event_col:
            shape = shape + str(row[col])

        event = GetEvent(shape)

        bar_list[index] = color

        stacked_fig.data[index].update(
            name=str(code_id),
            marker=dict(color="rgba(0,0,0,0)",line=dict(
                    color=color,
                    width=3.5
                )),
            customdata=[[code_id, detect_time, qid, row_index, picture_number, event, round(percent, 2)]],
            hovertemplate="<br>".join([
                "post_number=%{customdata[0]}",
                "detect time=%{customdata[1]}",
                "questionnaire id=%{customdata[2]}",
                "percent=%{customdata[6]}",
                "row index=%{customdata[3]}", 
                "picture number=%{customdata[4]}", 
                "event=%{customdata[5]}"                           
                ])
        )

    stacked_fig = UpdateFigureLayout(stacked_fig, stacked_layout)

    return stacked_fig, bar_list

def DiscussFigImageUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout):
    updatefig_dataframe = stacked_dataframe[stacked_dataframe['row_index'].isin(Select_row_index)]
    
    bar_list = {}
    update_fig_index = []
    update_for_event = {}
    for i, trace in enumerate(stacked_fig.data):
        row_id = trace['customdata'][0][3]
        if trace['name'] != "-1" and row_id in Select_row_index:
            update_fig_index.append(i)
    update_fig_index = sorted(update_fig_index, reverse=True)

    for index, (i, row) in zip(update_fig_index, updatefig_dataframe.iterrows()):
        color = row['color']
        bar_list[index] = color
        stacked_fig.data[index].update(
            marker=dict(color="rgba(0,0,0,0)",line=dict(
                    color=color,
                    width=3.5
                )),
        )
    stacked_fig = UpdateFigureLayout(stacked_fig, stacked_layout)

    return stacked_fig, bar_list

def EventFigImageUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout, y_size):
    updatefig_dataframe = stacked_dataframe[stacked_dataframe['row_index'].isin(Select_row_index)]
    
    bar_list = {}
    update_fig_index = []
    update_img_index = []
    for i, trace in enumerate(stacked_fig.data):
        row_id = trace['customdata'][0][3]
        if trace['name'] != "-1" and row_id in Select_row_index:
            update_fig_index.append(i)

    update_fig_index = sorted(update_fig_index, reverse=True)

    for index, (i, row) in zip(update_fig_index, updatefig_dataframe.iterrows()):
        shape = ""
        for col in event_col:
            shape = shape + str(row[col])
        
        event = GetEvent(shape)

        code_id = row['code_id']  
        percent = row['percent']
        picture_number = row['picture_number']
        color = row['color']
        detect_time = row['detect_time']
        qid = row['qid']
        row_index = row['row_index']

        bar_list[index] = color
        if int(picture_number) not in update_img_index:
            update_img_index.append(int(picture_number))
        
        stacked_fig.data[index].update(
            name=str(code_id),
            marker=dict(color="rgba(0,0,0,0)",line=dict(
                    color=color,
                    width=3.5
                )),
            customdata=[[code_id, detect_time, qid, row_index, picture_number, event, round(percent, 2)]],
            hovertemplate="<br>".join([
                "post_number=%{customdata[0]}",
                "detect time=%{customdata[1]}",
                "questionnaire id=%{customdata[2]}",
                "percent=%{customdata[6]}",
                "row index=%{customdata[3]}", 
                "picture number=%{customdata[4]}", 
                "event=%{customdata[5]}"                           
                ])
        )

    updatefig_dataframe = stacked_dataframe[stacked_dataframe['picture_number'].isin(update_img_index)]
    x_dict = defaultdict(list)
    for i, row in updatefig_dataframe.iterrows():
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
        event_candidate_index = []
        for j, layout_img in enumerate(stacked_fig.layout['images']):
            if layout_img['x'] == x_axis and layout_img['y'] >= y_size - y_size*0.13: 
                if j not in event_candidate_index:
                    event_candidate_index.append(j)

        event_candidate_index = sorted(event_candidate_index)

        for j in event_candidate_index:
            stacked_fig.layout['images'][j].visible = False

        if shape != 0:
            y_axis = y_size + 0.5 - y_size*0.13         
            for j in range(len(binary1_index)):
                event = event_map[binary1_index[j]]
                if j < len(event_candidate_index):
                    stacked_fig.layout['images'][event_candidate_index[j]].source = source[event]
                    stacked_fig.layout['images'][event_candidate_index[j]].visible = True
                else:
                    stacked_fig.add_layout_image(
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

    stacked_fig = UpdateFigureLayout(stacked_fig, stacked_layout)

    return stacked_fig, bar_list
