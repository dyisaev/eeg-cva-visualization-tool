import os
import json
import numpy as np
import pandas as pd

def init_gaze(filename):
    try:  # if the data exists ....
        with open(filename) as json_file:
            g = json.load(json_file)
        gaze = {'gazex': g['gazeX'], 'gazey': g['gazeY'], 'frame': g['frame']}
    except FileNotFoundError:
        gaze = None 
        print('\r warning: no gaze data for this task. sbj: {}'.format(filename), end='')
    return gaze
def init_nose(filename):
    try:  # if the data exists ....
        with open(filename) as json_file:
            n = json.load(json_file)
        nose = {'nosex': n['noseX'], 'nosey': n['noseY'], 'frame': n['frame']}
    except FileNotFoundError:
        nose = None 
        print('\r warning: no gaze data for this task. sbj: {}'.format(filename), end='')
    return nose
    
def init_landmarks(filename):
    # === Init. Landmarks and head pose === #
    try:
        with open(filename) as json_file:
            facial_data = json.load(json_file)
        landmarks = {'landmarks': facial_data['landmarks'],'frame': facial_data['frame'] } 
    except FileNotFoundError:
        landmarks = None
        print('\r warning: no landmarks or headpose data for this task. sbj: {}'.format(filename), end='')
    return landmarks

def init_headpose(filename):
    # === Init. Landmarks and head pose === #
    try:
        with open(filename) as json_file:
            facial_data = json.load(json_file)
        headpose = {'frame': facial_data['frame'], 'headpose': facial_data['headpose']}
    except FileNotFoundError:
        headpose = None 
        print('\r warning: no landmarks or headpose data for this task. sbj: {}'.format(filename), end='')
    return headpose
def init_event(path,event):
    filename=path+'/json/'+event+'.json'
    try:  # if the data exists ....
        with open(filename) as json_file:
            e = json.load(json_file)
        events = {event: e[event], 'frame': e['frame']}
    except FileNotFoundError:
        events = None 
        print('\r warning: no gaze data for this task. sbj: {}'.format(filename), end='')
    return events
