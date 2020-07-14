# UAVS 3D simulation and visualization using python

## How to use the 3D environment:

	1. Download the whole repository
	2. Install the following packages:
		Panda3d 1.10.6.post2
		OpenCv 4.2.0
		Numpy 1.19.0
		Scipy 1.5.0
		PyTorch 1.5.1
	3. Run ./Main.py (does not work on Spyder console)
	4. Basic Controls:
		C - Changes camera
		WASD - Changes external camera angle
		QE - Changes external camera distance
    R - Resets Camera
	5. Controller:
		Machine Learning Based Controller, trained by a PPO algorithm. 
	6. Estimation Algorithm:
		MEMS - Simulates an onboard IMU, with gyroscope, accelerometer and magnetometer. TRIAD algorithm is used to estimate attitude, retangular integrator is used to estimate position. 
		True State - Uses the exact simulated state.
