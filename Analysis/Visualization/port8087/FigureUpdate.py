from collections import defaultdict
from PIL import Image

ROOT_PATH = 'D:/Users/MUILab-VR/Desktop/News Consumption/'

event_list = ["external link", 'comment', "news", "external&news", "comment&news"]

source = {'post':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png"),
        'news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/news.png"),
        'external link':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/link.png"),
        'external&news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/link.png"),
        'comment':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png"),
        'comment&news':Image.open(ROOT_PATH + "Analysis/Visualization/EventIcon/comment.png")}

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

def CombineFigUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout):
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
    return stacked_fig

def DeleteFigUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout, visible):
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
                stacked_fig.data[j].visible = visible
            for j, layout_img in enumerate(stacked_fig.layout['images']):
                if layout_img['x'] == int(picture_number) + 0.4:
                    stacked_fig.layout['images'][j].visible=visible
        stacked_fig.data[index].visible = visible

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
    return stacked_fig

def SplitFigUpdate(stacked_fig, stacked_dataframe, post_number, stacked_layout):
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
    return stacked_fig

def DiscussFigUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout):
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
    return stacked_fig

def EventFigUpdate(stacked_fig, stacked_dataframe, Select_row_index, stacked_layout):
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

        event = "comment&news"
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
        # print(x_axis, shape_events, event)
        event_candidate_index = []
        for j, layout_img in enumerate(stacked_fig.layout['images']):
            if layout_img['x'] == x_axis: 
                if j not in event_candidate_index:
                    event_candidate_index.append(j)
                    # print("event_candidate_index", j)
                if x_axis not in has_exist:
                    has_exist.append(x_axis)
                    # print("has_exist", x_axis)

        event_candidate_index = sorted(event_candidate_index)
        # print("sorted list: ", event_candidate_index)
        # print("has_exist list: ", has_exist)

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
    return stacked_fig
