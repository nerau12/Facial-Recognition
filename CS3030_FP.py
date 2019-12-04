import face_recognition as frm
import pickle
import numpy
import sys
import os
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
    # empty constructor
    def __init__(self):
        pass
    # compares all known encodings to all encodings in an image
    # returns a tuple of unknown encodings and found encodings

    def FindAll(self, img, knownEncs):
        allEncs = frm.face_encodings(img)
        foundEncs = []
        unknownEncs = []
        found = False
        # check all encodings
        for enc in allEncs:
            # check known encodings
            for known in knownEncs:
                result = frm.compare_faces([known], enc)
                # a known encoding was found. so append to found and then break out of loop
                if True in result[0]:
                    foundEncs.append(known)
                    found = True
                    break
            # if nothing was found then add to list of unknown encodings

            if found == False:
                knownEncs.append(enc)
            else:
                found = False
        return (unknownEncs, foundEncs)

    # Accept an image and an encoding, then compares them
    def Identify(self,img, enc):
        loc = frm.face_locations(img)
        t_enc = frm.face_encodings(img, loc)
        #t_enc = img
        if True in frm.compare_faces(t_enc,enc):
            return True
        return False


class Main:
    def __init__(self):
        self.running = True
        self.c_thread = None
        self.numFrames = 0
        self.video = Video('testclip.mp4')                                       # load video
        self.dbase = Database()                                                  # load database
        self.faceDet = FaceDetector()#

    def programStart(self):
        self.c_thread = threading.Thread(target=self.clock)                 # c_thread to count program run time
        self.c_thread.start()                                               # c_thread

        testEnc = [self.dbase.readEncoding()]
        # get frames at a sample rate of 1 frame per 50
        while (self.video.getSampleFrame(50)):
            self.numFrames = self.numFrames + 1
            img = self.video.currentFrame
            results = self.faceDet.FindAll(img, testEnc)

        self.programEnd()

    # runs at end of programStart() to release resources
    def programEnd(self):
        self.running = False                                                # ends run condition in c_thread while loop
        self.c_thread.join()                                                # c_thread joins main thread
        print('Program End')                                                # end message

    # prints and counts the time elapsed for the program
    def clock(self):
        seconds = 0

        while self.running:
            time.sleep(1)
            seconds += 1
            print(f'Time Elapsed ({seconds})')


main = Main()
main.programStart()
