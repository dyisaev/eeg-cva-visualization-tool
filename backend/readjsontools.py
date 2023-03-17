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

'''    
def init_predictions(fpath,subj_id,prefix='cva_pred',postfix='raw'):
    # === Init. Landmarks and head pose === #
    filename = os.path.join(fpath, 'cva_predictions', prefix+'_'+postfix+'.json' )
    try:
        with open(filename) as json_file:
            pred_data = json.load(json_file)
    except FileNotFoundError:
        landmarks = None
        headpose = None 
        print('\r warning: no ML predictions data for this task. sbj: {}'.format(fpath), end='')
    return pred_data

def init_eye_aspect_ratio(landmarks):
    if landmarks is not None:
        eyear = compute_eye_aspect_ratio(landmarks)
    else:
        eyear = None
    return eyear
    
def compute_eye_aspect_ratio(landmarks):
    """ 
    The eye aspect ratio measures the openness of the eye by computing the ratio between a) the distance between the upper and lower eye lid, 
    and b) the distance between the two eye corners. The eye aspect ratio can be measured independently for each eye. Here we return the mean 
    of the two eyes since similar behavior is expected under normal circumstances.

    Arguments:
        landmarks {dic} -- Landmarks dic with the keys: 'frame', 'landmarks'
    
    Returns:
        eye_aspect_ratio {float}: the aspect ratio between the vertical and horizontal distance of the eye landmarks
    """
    lm = np.array(landmarks['landmarks'], dtype=float)

    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    h1l = np.sqrt(np.sum((lm[:,20,:] - lm[:,24,:]) ** 2, axis=1))
    h2l = np.sqrt(np.sum((lm[:,21,:] - lm[:,23,:]) ** 2, axis=1))
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    wl = np.sqrt(np.sum((lm[:,19,:] - lm[:,22,:]) ** 2, axis=1))
    # compute the eye aspect ratio
    earl = (h1l + h2l) / (2.0 * wl)

    # Do the same for the right eye
    h1r = np.sqrt(np.sum((lm[:,26,:] - lm[:,30,:]) ** 2, axis=1))
    h2r = np.sqrt(np.sum((lm[:,27,:] - lm[:,29,:]) ** 2, axis=1))
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    wr = np.sqrt(np.sum((lm[:,25,:] - lm[:,28,:]) ** 2, axis=1))
    # compute the eye aspect ratio
    earr = (h1r + h2r) / (2.0 * wr)

    # compute the minimum eye aspect ratio of the two eyes
    # ear = np.minimum(earl, earr)
    ear = 0.5 * (earl + earr)

    return {'frame': landmarks['frame'], 'eye_aspect_ratio': ear}

'''