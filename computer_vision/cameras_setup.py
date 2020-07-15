import panda3d
from computer_vision.img_2_cv import opencv_camera
from computer_vision.camera_calibration import calibration

class cameras():
    def __init__(self, render, frame_interval, cam_names):
        self.render = render
        
        # CAMERA NEUTRAL POSITION
        self.cam_neutral_pos = panda3d.core.LPoint3f(5, 5, 7)       
        
        self.opencv_cameras = []
        self.opencv_cam_cal = []
        for i, name in enumerate(cam_names):
            self.opencv_cameras.append( opencv_camera(self.render, name, frame_interval) )
            # CALIBRATION ALGORITHM
            self.opencv_cam_cal.append( calibration(self.render, self.opencv_cameras[i]) )        
        self.render.taskMgr.add(self.calibration_test)   
    
    def calibration_test(self, task):
            for i in self.opencv_cam_cal:
                if i.calibrated == False:                
                    return task.cont
            self.render.run_setup()
            return task.done