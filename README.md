# UAVs Visual Simulation, A Python Approach

For full documentation of the project, please refer to: https://osf.io/a9s54/

The presented software is a solution to a common problem in machine learning and computer vision applied to UAVS. 

In the early phases of an algorithm, it is necessary to be able to simulate it and observe if its running as intended, making necessary corrections and bug catching before downloading and running it in a real platform, enabling a faster development speed and lowering costs of Implementation.

Simulating cameras with good performance is a particularly difficult task, even more so on python, developing one from scratch would take a lot of time, effort, and a good performance wasn't garanteed, so a python game engine was chosen, called Panda3D. 

With this engine and its integration on the presented software, it is possible to simulate commom UAVs missions, as well as its dynamics, on-board or off-board cameras, object visualisation and colision detection. 


At this moment, the presented software is capable of:

1. Fully functional quadrotor
2. On-board and off-board cameras
3. Direct OpenCv integration. 
4. Fully Customizable Dynamics
5. Fully Customizable Scenario Model
6. Fully Customizable UAV Model

In future versions we plan to include:

1. Object Colision Detection


# How To Use

### Basic Usage
1. Needed Packages:
    1. Panda3D Version 1.10.6.post2
    2. Numpy Version 1.19.0
    3. OpenCv Version 4.2.0
    4. Scipy Version 1.5.0
    5. Pytorch Version 1.5.1
1. Download and unpack the following repository: https://github.com/rafaelcostafrf/UAV_3d_virtual_env
2. Running the file ./main.py will render the world and show an example of the software capabilities. Keep in mind to run the command from terminal. 
3. In this example, there are two OpenCV cameras, cam_1 and cam_2. Both cameras are facing down, on-board the UAV. 
4. It is possible to add more OpenCV cameras, just adding another name to cam_names variable in ./main.py
5. It is advisable to run the camera calibration algorithm, just delete the camera calibration files in ./config/camera_calibration_*.npz The software will detect the abscence of files and will automatically run the calibration algorithm.
6. In ./computer_vision/quadrotor_cv.py it is possible to observe an example of camera manipulation in ./computer_vision/quadrotor_cv.computer_vision.init(). 
7. Setting the camera position is easily done by the cam.setPos(). setPos command is a (X, Y, Z) coordinate system in meters. 
9. To set a camera parent the reparentTo() command is used. The argument must be a model present in the environment or the render itself. If you parent a camera to a model, the camera will be connected by translation to that model. Parenting a camera to the render keeps it in the same place.  
10. In computer_vision.img_show() it is possible to observe a simple OpenCv use, showing the camera image with OpenCv commands. 
11. Any OpenCv algorithm should be usable at this point. 

### Render Camera Controls

1. C - Changes camera
2. WASD - Changes external camera angle
3. QE - Changes external camera distance
4. R - Resets Camera

### Provided Controller
 The quadrotor provided as example is controlled by a neural network trained by a machine learning algorithm. It may be run with simulated states by the flag REAL_CTRL = True or else by MEMS simulation with REAL_CTRL = False. This flag is set in ./main.py

The controller is capable of hovering in the same position, or else initializing in a random state, by changing the flag HOVER = True or False. 

### Usefull Functions

./computer_vision/img_2_cv.py:
    
    opencv_camera(render, name, frame_interval) - Creates an OpenCv Camera.
        1. render is self in main.py (Panda3D render)
        2. name is the camera name
        3. frame_interval is the capture interval (related to rendered frames) e.g a frame interval of 10 means one OpenCv camera capture every 10 rendered frames.
        
    get_image(target_frame=True) - returns a True and an Image if the algorithm was able to receive an image, returns False and None if no image was found. 
        1. target_frame=True - Sets the buffer inactive for the next (frame_interval-1) frames (increasing render performance)
        2. target_frame=False - Keeps the buffer active, reducing performance, but may be usefull in some scenarios.
        
./computer_vision/camera_calibration.py
    
    calibration() - Calibrates a camera, given its class and the render. The calibration is based on pictures of a ChessBoard pattern in various angles and distances.
    
./computer_vision/cameras_setup.py
    
    cameras(): Automates the process of creating the camera and calibrating it, the user only concern is naming the cameras, uses both opencv_camera() and calibration() functions. 
    
        1. To access the camera calibration matrixes, the user should call cameras.opencv_cam_cal[i].mtx for intrinsic matrix and cameras.opencv_cam_cal[i].dist for the distortion matrix, being [i] the index of the camera. 
        2. To access the camera image the user should call cameras_setup.opencv_cameras[i].get_image() being [i] the camera index.
    
./environment/position.py
    
    Everything in this file is about the dynamics and control of the presented quadrotor. The discussion of its intricacies isn't the scope of this project. If you wish to change the UAV dynamics, this file should be used just as a rough reference. 
    The most important functions are:
        1. self.setPos(X, Y, Z) in meters
        2. self.setHpr(Yaw, Pitch, Roll) in degrees
    Your UAV model should output that information. The functions above update the 3D model position and attitude in the render. 
    
### Warnings and Engine Functionality
1. The engine runs entirely by rendered frames. When developing an algorithm, keep that in mind. The developed function will be run once per frame. For more information please refer to Panda3D taskMgr (task manager) funcion, particularly useful commands are task.cont and task.done. 
2. The frame_interval variable is heavily performance related. Its function is to limit the OpenCv camera capture to 1 in each 10 of the Panda3D frames. Faster tasks might need more OpenCv frames, slower taks might need less. This variable impacts performance because all the OpenCv frames are being sent to RAM, slowing the process down, compared to running it in a dedicated GPU. 
