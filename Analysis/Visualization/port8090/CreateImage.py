# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:05:26 2022

@author: MUILab-VR
"""

import os
import numpy as np
import cv2

#----- config ------
ROOT_PATH = 'D:/Users/MUILab-VR/Desktop/News Consumption/CHI2022/media_screenshot_toolkit/'
data_path = ROOT_PATH + "Database/SplitPostFile/"
profile_path = ROOT_PATH + "Analysis/UserProfile/UserProfileForCode.csv"

#UID = ["U16"]
FolderName = "AttentionImage"

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

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20): 
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5 
    pts= [] 
    for i in np.arange(0,dist,gap): 
        r=i/dist 
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5) 
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5) 
        p = (x,y) 
        pts.append(p) 

    if style=='dotted': 
        for p in pts: 
            cv2.circle(img,p,thickness,color,-1) 
    else: 
        s=pts[0] 
        e=pts[0] 
        i=0 
        for p in pts: 
            s=e 
            e=p 
            if i%2==1: 
                cv2.line(img,s,e,color,thickness) 
            i+=1 
    return img

def draw_visible_area(img):
    height, width, channel = img.shape
    height -= 1
    width -= 1
    
    if height >= width:
        top_pixel = int(height * 0.13)
        bottom_pixel = height - int(height * 0.13)
        
        temp_img = img.copy()
        alpha = 0.8 
        cv2.rectangle(temp_img, (0, 0), (width, top_pixel), (128, 128, 128), -1)       
        img = cv2.addWeighted(temp_img, alpha, img, 1 - alpha, 0)
        
        temp_img = img.copy()
        cv2.rectangle(temp_img, (width, bottom_pixel), (0, height), (128, 128, 128), -1)
        img = cv2.addWeighted(temp_img, alpha, img, 1 - alpha, 0)
    else:
        top_pixel = int(width * 0.13)
        bottom_pixel = width - int(width * 0.13)

        img = cv2.rectangle(img, (0, 0), (width, top_pixel), (128, 128, 128), -1)
        img = cv2.rectangle(img, (width, bottom_pixel), (0, height), (128, 128, 128), -1)   
    return img

def draw_rectangle(img, pt0, pt1, color):
    height, width, channel = img.shape
    height -= 1
    width -= 1
    return cv2.rectangle(img, pt0, pt1, color, 15)
