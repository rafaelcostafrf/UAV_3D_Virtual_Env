import numpy as np
import cv2 as cv

class opencv_camera():
    def __init__(self, render, name):
        self.render = render   
        window_size = (self.render.win.getXSize(), self.render.win.getYSize())     
        self.buffer = self.render.win.makeTextureBuffer(name, *window_size, None, True)
        self.cam = self.render.makeCamera(self.buffer)
        self.cam.setName(name)     
        self.cam.node().getLens().setFilmSize(36, 24)
        self.cam.node().getLens().setFocalLength(45)
        self.name = name
        
    def get_image(self):
        tex = self.buffer.getTexture()  
        img = tex.getRamImage()
        image = np.frombuffer(img, np.uint8)
        
        if len(image) > 0:
            image = np.reshape(image, (tex.getYSize(), tex.getXSize(), 4))
            image = cv.resize(image, (0,0), fx=0.5, fy=0.5)
            image = cv.flip(image, 0)
            return True, image
        else:
            return False, None