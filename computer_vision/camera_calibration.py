import cv2 as cv
import numpy as np
from computer_vision.detector_setup import detection_setup

class calibration():
    def __init__(self, render, cv_cam):
        self.cam = cv_cam
        self.render = render
        self.fast, self.criteria, self.nCornersCols, self.nCornersRows, self.objp, self.checker_scale, self.checker_sqr_size = detection_setup(render) 
        self.path = './config/camera_calibration_' + self.cam.name + '.npz'
        try: 
            npzfile = np.load(self.path)
            self.mtx = npzfile[npzfile.files[0]]
            self.dist = npzfile[npzfile.files[1]]
            self.calibrated = True
            print('Calibration File Loaded')
        except:
            print('Could Not Load Calibration File, Calibrating... ')
            self.render.taskMgr.add(self.calibrate, 'Camera Calibration')   
            self.calibrated = False
            self.render.cam_pos = []
            self.objpoints = []
            self.imgpoints = []
            
    def calibrate(self, task): 
            rand_pos = (np.random.random(3)-0.5)*5
            rand_pos[2] = np.random.random()*3+2
            cam_pos = tuple(rand_pos)
            
            self.cam.cam.reparentTo(self.render.render)
            self.cam.cam.setPos(*cam_pos)
            self.cam.cam.lookAt(self.render.checker)

            self.render.quad_model.setPos(10,10,10)
            
            ret, image = self.cam.get_image()
            if ret:
                img = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
                self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                
                ret, corners = cv.findChessboardCorners(self.gray, (self.nCornersCols, self.nCornersRows), 
                                                        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FILTER_QUADS+ cv.CALIB_CB_FAST_CHECK)
                if ret:
                    self.objpoints.append(self.objp)             
                    self.imgpoints.append(corners)
                    img = cv.drawChessboardCorners(img, (self.nCornersCols, self.nCornersRows), corners, ret)
                    cv.imshow('img',img)
                    cv.waitKey(1)
            
            if len(self.objpoints) > 50:
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
                if ret:
                    h,  w = img.shape[:2]
                    newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
                    dist = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]) #Camera Perfeita (Simulada), logo não há distorção
                    dst = cv.undistort(img, mtx, dist, None, newcameramtx)    
                    cv.imshow('img', dst)
                    self.mtx = mtx
                    self.dist = dist
                    print('Calibration Complete')
                    self.calibrated = True
                    np.savez(self.path, mtx, dist)
                    print('Calibration File Saved')
                    print('Calibration Complete! Running the Algorithm...')

                    self.render.quad_model.setPos(0, 0, 0)
                    self.cam.cam.setPos(0,0,0.01)
                    self.cam.cam.reparentTo(self.render.quad_model)
                    self.render.run_setup()
                    cv.destroyWindow('img')
                    return task.done
            return task.cont