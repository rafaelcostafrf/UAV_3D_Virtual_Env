import numpy as np
import panda3d
class camera_control():
    
    def __init__(self, env, render):
        self.camera_init = True
        self.render = render
        env.taskMgr.add(self.camera_move, 'Camera Movement')
        env.accept('r', self.camera_reset)
        env.accept('c', self.camera_change)
        self.env = env
        self.camera_change()
        
    def camera_reset(self):
        self.env.cam.reparentTo(self.render)
        self.env.cam.setPos(self.env.cam_neutral_pos)
        self.camera_init = True
        
    def camera_change(self):
        if self.camera_init:
            self.env.cam.reparentTo(self.env.quad_model)
            self.env.cam.setPos(0, 0, 0.01)
            self.env.cam.setHpr(0, 270, 0)           
            self.camera_init = False
        else:
            self.camera_reset()
            self.camera_init = True
        
    def camera_move(self, task):
        if self.camera_init:
            '''
            Moves the camera about the quadcopter
            '''
            keys = ('d', 'a', 'e', 'q', 'w', 's')
            
            moves = ((1, 0, 0),
                     (-1, 0, 0),
                     (0, 1, 0),
                     (0, -1, 0),
                     (0, 0, 1),
                     (0, 0, -1))
            
            mat = np.array(self.env.cam.getMat())[0:3,0:3]
            move_total = panda3d.core.LPoint3f(0, 0, 0)
            for key, move in zip(keys, moves):
                pressed_key = self.env.mouseWatcherNode.is_button_down(panda3d.core.KeyboardButton.asciiKey(key))
                if pressed_key:
                    move = panda3d.core.LPoint3f(move)
                    move_total = move_total + move 
                    
            # ROTATE COORDINATE SYSTEM (TO CAMERA)                
            move_total = panda3d.core.LPoint3f(*tuple(np.dot(mat.T, np.array(move_total))))
            
            # MULTIPLIER IS PROPORTIONAL TO QUAD TO CAMERA DISTANCE 
            multiplier = np.linalg.norm(np.array(self.env.cam.getPos()-self.env.quad_model.getPos()))*0.02
            
            cam_pos = self.env.cam.getPos()
            cam_pos += move_total*multiplier
            self.env.cam.setPos(cam_pos)
            self.env.cam.lookAt(self.env.quad_model)        
        return task.cont