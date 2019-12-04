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
    def __init__(self, video_path):
        self.currentFrame = None
        self.videoPath = video_path
        self.video = cv2.VideoCapture(self.videoPath)
    # places next frame in self.video and returns true or false if an image was found

    def get_frame(self):
        frame_read, self.currentFrame = self.video.read()
        if not frame_read:
            self.video.release()
        return frame_read
    # closes video.

    def force_close(self):
        self.video.release()

    # returns 1 frame spaced apart by the sample rate.
    # This should be used when paired with facial recognition and not used as the default.
    # returns True/False
    def get_sample_frame(self, sample_rate = 1):
        for i in range(0,sample_rate):
            frameRead, self.currentFrame = self.video.read()
        if not frameRead:
            self.video.release()
        return frameRead
    # counts number of frames of the video. then reloads the video
    # Note: do NOT use this in the middle of normal operations
    #       Use this before or after what you're trying to do. Not in the middle.

    def count_frames(self):
        count = 0
        while self.get_frame():
            count += 1
        self.video = cv2.VideoCapture(self.videoPath)
        return count


# read/write, organize, and manage all data.
class Database:
    def __init__(self):
        pass

    def read_encoding(self, performer='RyanReynolds'):
        with open(f'{performer}.fr', 'rb') as f:
            return numpy.array(pickle.load(f))

    def write_encoding(self, encoding, performer='RyanReynolds'):
        with open(f'{performer}.fr') as f:
            pickle.dump(encoding, f)


class FaceDetector:
    # empty constructor
    def __init__(self):
        pass
    # compares all known encodings to all encodings in an image
    # returns a tuple of unknown encodings and found encodings

    def find_all(self, img, known_encs):
        all_encs = frm.face_encodings(img)
        found_encs = []
        unknown_encs = []
        found = False
        # check all encodings
        for enc in all_encs:
            # check known encodings
            for known in known_encs:
                result = frm.compare_faces([known], enc)
                # a known encoding was found. so append to found and then break out of loop
                if True in result[0]:
                    found_encs.append(known)
                    found = True
                    break
            # if nothing was found then add to list of unknown encodings

            if not found:
                known_encs.append(enc)
            else:
                found = False

        return (unknown_encs, found_encs)

    # Accept an image and an encoding, then compares them
    def identify(self, img, enc):
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
        self.video = Video('testclip.mp4')                                       # load video
        self.data_base = Database()                                                  # load database
        self.faceDet = FaceDetector()#

    def program_start(self):
        self.c_thread = threading.Thread(target=self.clock)                 # c_thread to count program run time
        self.c_thread.start()                                               # c_thread

        test_enc = [self.data_base.read_encoding()]
        
        # get frames at a sample rate of 1 frame per 50
        while self.video.get_sample_frame(50):
            self.numFrames = self.numFrames + 1
            img = self.video.currentFrame
            results = self.faceDet.find_all(img, test_enc)

        self.program_end()

    # runs at end of programStart() to release resources
    def program_end(self):
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
main.program_start()
