import cv2 as cv


class computer_vision():
    def __init__(self, render, quad_model, cv_cam, cv_cam_2, camera_cal):
        
        self.mtx = camera_cal.mtx
        self.dist = camera_cal.dist

        self.render = render  
        
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        
        self.cv_cam = cv_cam
        self.cv_cam.cam.setPos(0, 0, 0.01)
        self.cv_cam.cam.lookAt(0, 0, 0)
        self.cv_cam.cam.reparentTo(self.render.quad_model)
        
        
        self.cv_cam_2 = cv_cam_2
        self.cv_cam_2.cam.setPos(0, 0, 1)
        self.cv_cam_2.cam.lookAt(0, 0, 0)
        self.cv_cam_2.cam.reparentTo(self.render.quad_model)

        self.render.taskMgr.add(self.img_show, 'OpenCv Image Show')
    
    def img_show(self, task):
        if task.frame % self.cv_cam.frame_int == 1:           
            ret, image = self.cv_cam.get_image()
            ret, image2 = self.cv_cam_2.get_image()
            if ret:
                cv.imshow('Drone Camera',image)
                cv.imshow('Drone Camera 2 ',image2)
                cv.waitKey(1)
        return task.cont