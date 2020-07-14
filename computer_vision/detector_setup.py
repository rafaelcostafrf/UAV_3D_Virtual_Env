import cv2 as cv
import numpy as np

def detection_setup(render):
    checker_scale = render.checker_scale
    checker_sqr_size = render.checker_sqr_size
    fast = cv.FastFeatureDetector_create()
    fast.setThreshold(20)   
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 1, 0.0001) 
    nCornersCols = 9
    nCornersRows = 6
    objp = np.zeros((nCornersCols*nCornersRows, 3), np.float32)
    objp[:,:2] = (np.mgrid[0:nCornersCols, 0:nCornersRows].T.reshape(-1,2))*checker_scale*checker_sqr_size
    
    return fast, criteria, nCornersCols, nCornersRows, objp, checker_scale, checker_sqr_size