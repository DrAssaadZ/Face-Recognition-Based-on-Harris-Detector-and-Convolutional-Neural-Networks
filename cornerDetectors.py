'''picking the best corner detector (between fast, sift and shi-tomas
for our model'''

import cv2 as cv
import numpy as np


def shi_tomas_detector(img, nb_points, distance):
    gray = np.float32(img)

    # detecting corners
    corners = cv.goodFeaturesToTrack(gray, nb_points, 0.01, distance)
    return corners
   

def sift_detector(img, n_corn):
    # initi the sift detector
    sift = cv.xfeatures2d.SIFT_create(n_corn)
    corners = sift.detect(img, None)
    return corners
        

def fast_detector(img):

    # Initiate FAST object
    fast = cv.FastFeatureDetector_create()

    corners = fast.detect(img, None)
    return corners


def surf_detector(img):
    surf = cv.xfeatures2d.SURF_create(10, 5)
    corners = surf.detect(img, None)
    return corners
	

    













