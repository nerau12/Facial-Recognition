import face_recognition as frm
import pickle
import numpy
import sys
import os
from numba import jit
import datetime
import threading
import time
import cv2
from PIL import Image
# pip install dlib==19.7.0, make sure that dlib is version 19.7, otherwise there are problems
# this class captures video from a file on initialization
# getFrame method sets the current frame global variable to the next frame then returns true/false
#   true/false indicates if there was a next frame (false if image is None)
class Video:
    # constructor
    def __init__(self, videoPath):
        self.currentFrame = None
        self.videoPath = videoPath
        self.video = cv2.VideoCapture(self.videoPath)
    # places next frame in self.video and returns true or false if an image was found
    def getFrame(self):
        frameRead, self.currentFrame = self.video.read()
        if not frameRead:
            self.video.release()
        return frameRead
    # closes video.
    def forceClose(self):
        self.video.release()

    # returns 1 frame spaced apart by the sample rate.
    # This should be used when paired with facial recognition and not used as the default.
    # returns True/False
    def getSampleFrame(self, sampleRate = 1):
        for i in range(0,sampleRate):
            frameRead, self.currentFrame = self.video.read()
        if not frameRead:
            self.video.release()
        return frameRead
    # counts number of frames of the video. then reloads the video
    # Note: do NOT use this in the middle of normal operations
    #       Use this before or after what you're trying to do. Not in the middle.
    def countFrames(self):
        count = 0
        while self.getFrame():
            count += 1
        self.video = cv2.VideoCapture(self.videoPath)
        return count
    
    def getTime(self):
        pass
    #self.video.
        #CV_CAP_PROP_POS_MSEC 

# read/write, organize, and manage all data.
class Database:
    def __init__(self):
        pass
    def readEncoding(self, performer='RyanReynolds'):
        with open(f'{performer}.fr', 'rb') as f:
            return numpy.array(pickle.load(f))
    def writeEncoding(self,encoding, performer='RyanReynolds'):
        with open(f'{performer}.fr') as f:
            pickle.dump(encoding, f)


class FaceDetector:
    # constructor
    # self.frame is where we're trying to ID faces
    # self.database is where queries are made regarding faces
    def __init__(self):
        pass

    # Accept an image and an encoding, then compares them
    @jit(nopython=True)
    def Identify(self,img, enc):
        loc = frm.face_locations(img)
        t_enc = frm.face_encodings(img, loc)
        if True in frm.compare_faces(t_enc,enc):
            return True
        return False


class Main:
    def __init__(self):
        self.running = True
        self.c_thread = None
        self.numFrames = 0
        self.debug_list = []                                     # TBD: removed later. this is for debugging purposes
    def programStart(self):
        self.c_thread = threading.Thread(target=self.clock)      # c_thread to count program run time
        self.c_thread.start()                                                # c_thread
        video = Video('testclip.mp4')                            # load video
        dbase = Database()                                       # load database
        faceDet = FaceDetector()                                 #
        testEnc = dbase.readEncoding()
        s_frames = []
        # get frames at a sample rate of 1 frame per 50
        while (video.getSampleFrame(50)):
            self.numFrames = self.numFrames + 1
            img = video.currentFrame
            cv2.imwrite(f'debugImages/image{self.numFrames}.png',img)       # tbd: remove later
            self.debug_list.append((faceDet.Identify(img, testEnc)))
        self.programEnd()
    # runs at end of programStart() to release resources
    def programEnd(self):
        self.running = False                                            # ends run condition in c_thread while loop
        self.c_thread.join()                                                 # c_thread joins main thread
        deb_log = open('debug.txt', 'w')    # TBD: remove later
        for example in self.debug_list:
            deb_log.write(f'{example}\n')       # TBD: remove later
        #deb_log.write(str(self.debug_list)) # TBD: remove later
        deb_log.close()                     # TBD: remove later
        print('Program End')                                            # end message
    # prints and counts the time elapsed for the program
    def clock(self):
        seconds = 0
        while(self.running):
            time.sleep(1)
            seconds += 1
            print(f'Time Elapsed ({seconds})')
main = Main()
main.programStart()
