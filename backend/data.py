import pickle
import pyarrow as pa

from backend.readjsontools import init_gaze,init_nose, init_headpose, init_landmarks,init_event

class Data:
    def __init__(self,path,events=None) -> None:
        self.gaze=init_gaze(path+'/json/gaze.json')
        self.nose=init_nose(path+'/json/nose.json')
        self.headpose=init_headpose(path+'/json/headpose.json')
        self.events={}
        if not events is None:
            for event in events:
                self.events[event]=init_event(path,event)
    def serialize(self):

        return pa.serialize({'gaze':self.gaze,'nose':self.nose,'headpose':self.headpose,'events':self.events}).to_buffer()

    