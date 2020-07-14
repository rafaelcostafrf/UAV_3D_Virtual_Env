from scipy import integrate
import numpy as np
from environment.quaternion_euler_utility import euler_quat, quat_euler, deriv_quat, quat_rot_mat
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
"""""
QUADROTOR ENVIRONMENT
DEVELOPED BY: 
    RAFAEL COSTA FERNANDES
    PROGRAMA DE PÓS GRADUAÇÃO EM ENGENHARIA MECÂNICA, UNIVERSIDADE FEDERAL DO ABC
    SP - SANTO ANDRÉ - BRASIL

FURTHER DOCUMENTATION ON README.MD
"""""

## QUADROTOR PARAMETERS ##

## SIMULATION BOUNDING BOXES ##
BB_POS = 5 
BB_VEL = 10
BB_CONTROL = 9
BB_ANG = np.pi/2

# QUADROTOR MASS AND GRAVITY VALUE
M, G = 1.03, 9.82

# AIR DENSITY
RHO = 1.2041

#DRAG COEFFICIENT
C_D = 1.1

# ELETRIC MOTOR THRUST AND MOMENT
K_F = 1.435e-5
K_M = 2.4086e-7
I_R = 5e-5
T2WR = 2

## INDIRECT CONTROL CONSTANTS ##
IC_THRUST = 6
IC_MOMENTUM = 0.8


# INERTIA MATRIX
J = np.array([[16.83e-3, 0, 0],
              [0, 16.83e-3, 0],
              [0, 0, 28.34e-3]])

# ELETRIC MOTOR DISTANCE TO CG
D = 0.26

#PROJECTED AREA IN X_b, Y_b, Z_b
BEAM_THICKNESS = 0.05
A_X = BEAM_THICKNESS*2*D
A_Y = BEAM_THICKNESS*2*D
A_Z = BEAM_THICKNESS*2*D*2
A = np.array([[A_X,A_Y,A_Z]]).T


## REWARD PARAMETERS ##

# CONTROL REWARD PENALITIES #
P_C = 0.2
P_C_D = 0.3

## TARGET STEADY STATE ERROR ##
TR = [0.01, 0.1]
TR_P = [100, 10]


class quad():

    def __init__(self, t_step, n, euler=0, direct_control=0, T=1):
        
        """"
        inputs:
            t_step: integration time step 
            n: max timesteps
            euler: flag to set the states return in euler angles, if off returns quaternions
            deep learning:
                deep learning flag: If on, changes the way the env. outputs data, optimizing it to deep learning use.
                T: Number of past history of states/actions used as inputs in the neural network
                debug: If on, prints a readable reward funcion, step by step, for a simple reward weight debugging.
        
        """
        
        self.i = 0
        self.T = T                                              #Initial Steps
        
        self.bb_cond = np.array([BB_POS, BB_VEL,
                                 BB_POS, BB_VEL,
                                 BB_POS, BB_VEL,
                                 BB_ANG, BB_ANG, 4,
                                 BB_VEL, BB_VEL, BB_VEL])       #Bounding Box Conditions Array
        
        
        self.state_size = 13                                   #Quadrotor states dimension
        self.action_size = 4                                    #Quadrotor action dimension
        
        
        self.done = True                                        #Env done Flag
    
        self.n = n+self.T                                       #Env Maximum Steps
        self.t_step = t_step
        
        self.inv_j = np.linalg.inv(J)
        self.zero_control = np.ones(4)*(2/T2WR - 1)             #Neutral Action (used in reset and absolute action penality) 
        self.direct_control_flag = direct_control
        
    def seed(self, seed):
        
        """"
        Set random seeds for reproducibility
        """
        
        np.random.seed(seed)       
    
    
    
    def f2w(self,f,m):
        """""
        Translates F (Thrust) and M (Body x, y and z moments) into eletric motor angular velocity (rad/s)
        """""
        x = np.array([[1, 1, 1, 1],
                      [-D, 0, D, 0],
                      [0, D, 0, -D],                      
                      [-K_F/K_M, +K_F/K_M, -K_F/K_M, +K_F/K_M]])      
        
        y = np.array([f, m[0,0], m[1,0], m[2,0]])
        
        u = np.linalg.solve(x,y)
        u = np.clip(u,0,T2WR*M*G/4)
        
        w_1 = np.sqrt(u[0]/K_F)
        w_2 = np.sqrt(u[1]/K_F)
        w_3 = np.sqrt(u[2]/K_F)
        w_4 = np.sqrt(u[3]/K_F)        
        w = np.array([w_1,w_2,w_3,w_4])

        FM_new = np.dot(x,u)
        
        F_new = FM_new[0]
        M_new = FM_new[1:4]
        
        return u, w, F_new, M_new
        
    def f2F(self,f_action):
        f = (f_action+1)*T2WR*M*G/8

        
        w = np.array([[np.sqrt(f[0]/K_F)],
                      [np.sqrt(f[1]/K_F)],
                      [np.sqrt(f[2]/K_F)],
                      [np.sqrt(f[3]/K_F)]])
        
        F_new = np.sum(f)
        M_new = np.array([[(f[2]-f[0])*D],
                          [(f[1]-f[3])*D],
                          [(-f[0]+f[1]-f[2]+f[3])*K_M/K_F]])
        return w, F_new, M_new
    
    def drone_eq(self, t, x, action):
        
        """"
        Main differential equation, not used directly by the user, rather used in the step function integrator.
        Dynamics based in: 
            MODELAGEM DINÂMICA E CONTROLE DE UM VEÍCULO AÉREO NÃO TRIPULADO DO TIPO QUADRIRROTOR 
            by ALLAN CARLOS FERREIRA DE OLIVEIRA
            BRASIL, SP-SANTO ANDRÉ, UFABC - 2019
        Incorporates:
            Drag Forces, Gyroscopic Forces
            In indirect mode: Force clipping (preventing motor shutoff and saturates over Thrust to Weight Ratio)
            In direct mode: maps [-1,1] to forces [0,T2WR*G*M/4]
        """
        self.w, f_in, m_action = self.f2F(action)
        
        
        #BODY INERTIAL VELOCITY                
        vel_x = x[1]
        vel_y = x[3]
        vel_z = x[5]
        
        #QUATERNIONS
        q0 = x[6]
        q1 = x[7]
        q2 = x[8]
        q3 = x[9]
        
        #BODY ANGULAR VELOCITY
        w_xx = x[10]
        w_yy = x[11]
        w_zz = x[12]      

        #QUATERNION NORMALIZATION (JUST IN CASE)
        q = np.array([[q0, q1, q2, q3]]).T
        q = q/np.linalg.norm(q)
        
        # DRAG FORCES ESTIMATION (BASED ON BODY VELOCITIES)
        self.mat_rot = quat_rot_mat(q)
        v_inertial = np.array([[vel_x, vel_y, vel_z]]).T
        v_body = np.dot(self.mat_rot.T, v_inertial)
        f_drag = -0.5*RHO*C_D*np.multiply(A,np.multiply(abs(v_body),v_body))
        
        # DRAG MOMENTS ESTIMATION (BASED ON BODY ANGULAR VELOCITIES)
        
        #Discretization over 10 steps (linear velocity varies over the body)
        d_xx = np.linspace(0,D,10)
        d_yy = np.linspace(0,D,10)
        d_zz = np.linspace(0,D,10)
        m_x = 0
        m_y = 0
        m_z = 0
        for xx,yy,zz in zip(d_xx,d_yy,d_zz):
            m_x += -RHO*C_D*BEAM_THICKNESS*D/10*(abs(xx*w_xx)*(xx*w_xx))
            m_y += -RHO*C_D*BEAM_THICKNESS*D/10*(abs(yy*w_yy)*(yy*w_yy))
            m_z += -2*RHO*C_D*BEAM_THICKNESS*D/10*(abs(zz*w_zz)*(zz*w_zz))
        
        m_drag = np.array([[m_x],
                           [m_y],
                           [m_z]])
        
        #GYROSCOPIC EFFECT ESTIMATION (BASED ON ELETRIC MOTOR ANGULAR VELOCITY)                
        omega_r = (-self.w[0]+self.w[1]-self.w[2]+self.w[3])[0]
        
        m_gyro = np.array([[-w_xx*I_R*omega_r],
                           [+w_yy*I_R*omega_r],
                           [0]])

        #BODY FORCES
        self.f_in = np.array([[0, 0, f_in]]).T
        self.f_body = self.f_in+f_drag
        
        #BODY FORCES ROTATION TO INERTIAL
        self.f_inertial = np.dot(self.mat_rot, self.f_body)
        
        #INERTIAL ACCELERATIONS        
        accel_x = self.f_inertial[0, 0]/M        
        accel_y = self.f_inertial[1, 0]/M        
        accel_z = self.f_inertial[2, 0]/M-G
        self.accel = np.array([[accel_x, accel_y, accel_z]]).T
        
        #BODY MOMENTUM
        W = np.array([[w_xx],
                 [w_yy],
                 [w_zz]])

        m_in = m_action + m_gyro + m_drag - np.cross(W.flatten(), np.dot(J, W).flatten()).reshape((3,1))

        #INERTIAL ANGULAR ACCELERATION        
        accel_ang = np.dot(self.inv_j,m_in).flatten()
        accel_w_xx = accel_ang[0]
        accel_w_yy = accel_ang[1]
        accel_w_zz = accel_ang[2]
        
        #QUATERNION ANGULAR VELOCITY (INERTIAL)
   
        self.V_q = deriv_quat(W, q).flatten()
        dq0=self.V_q[0]
        dq1=self.V_q[1]
        dq2=self.V_q[2]
        dq3=self.V_q[3]

        
        # RESULTS ORDER:
        # 0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 q0, 7 q1, 8 q2, 9 q3, 10 w_xx, 11 w_yy, 12 w_zz
        out = np.array([vel_x, accel_x,
                         vel_y, accel_y,
                         vel_z, accel_z,
                         dq0, dq1, dq2, dq3,
                         accel_w_xx, accel_w_yy, accel_w_zz])
        return out

    def reset(self,det_state = None):
        
        """""
        inputs:
            det_state: 
                if == 0 randomized initial state
                else det_state is the actual initial state, depending on the euler flag
                if euler flag is on:
                    [x, dx, y, dy, z, dz, phi, theta, psi, w_xx, w_yy, w_zz]
                if euler flag is off:
                    [x, dx, y, dy, z, dz, q_0, q_1, q_2, q_3, w_xx, w_yy, w_zz]
        outputs:
            previous_state: system's initial state
        """""
        state = []
        action = []
        self.action_hist = []
        
        self.solved = 0
        self.done = False
        self.i = 0   
        self.prev_shaping = None
        self.previous_state = np.zeros(self.state_size)
        
        if det_state is not None:        
            self.previous_state = det_state
        else:
            self.ang = np.random.rand(3)-0.5
            Q_in = euler_quat(self.ang)
            self.previous_state[0:5:2] = (np.random.rand(3)-0.5)*BB_POS
            self.previous_state[1:6:2] = (np.random.rand(3)-0.5)*BB_POS/2
            self.previous_state[6:10] = Q_in.T
            self.previous_state[10:13] = (np.random.rand(3)-0.5)*1
        
        for i in range(self.T):
            self.action = self.zero_control
            self.action_hist.append(self.action)

            state_t, reward, done = self.step(self.action)
            state.append(state_t.flatten())
            action.append(self.zero_control)
        return np.array(state), np.array(action)
    


    def step(self, action):
        
        """""
        inputs:
            action: action to be applied on the system
        outputs:
            state: system's state in t+t_step actuated by the action
            done: False, else the system has breached any bounding box, exceeded maximum timesteps, or reached goal.
        """""
        
        if self.done:
            print('\n----WARNING----\n done flag is TRUE, reset the environment with environment.reset() before using environment.step()\n')
        self.i += 1
        self.action = np.clip(action,-1,1)
        self.action_hist.append(self.action)
        
        
        if self.direct_control_flag:
            u = self.action
            self.clipped_action = self.action
        else:
            f_in= action[0]*IC_THRUST+M*G
            m_action = (np.array([self.action[1:4]]).T)*IC_MOMENTUM
            u, _, f_new, m_new = self.f2w(f_in,m_action)
            #CLIPPED ACTION FOR LOGGING 
            self.clipped_action = np.array([(f_new-M*G)/IC_THRUST,
                                            m_new[0]/IC_MOMENTUM,
                                            m_new[1]/IC_MOMENTUM,
                                            m_new[2]/IC_MOMENTUM])
        
        
        self.y = (integrate.solve_ivp(self.drone_eq, (0, self.t_step), self.previous_state, args=(u, ))).y
        self.state = np.transpose(self.y[:, -1])
        self.quat_state = np.array([np.concatenate((self.state[0:10], self.V_q))])
        
        q = np.array([self.state[6:10]]).T
        q = q/np.linalg.norm(q)
        self.ang = quat_euler(q)
        self.previous_state = self.state
        self.done_condition()
        self.reward_function()
        return self.quat_state, self.reward, self.done

    def done_condition(self):
        
        """""
        Checks if bounding boxes done condition have been met
        """""
        
        cond_x = np.concatenate((self.state[0:6], self.ang, self.state[-3:]))
        for x, c in zip(np.abs(cond_x), self.bb_cond):
            if  x >= c:
                self.done = True

    def reward_function(self, debug=0):
        
        """""
        Reward Function: Working with PPO great results.
        Shaping with some ideas based on Continuous Lunar Lander v.2 gym environment:
            https://gym.openai.com/envs/LunarLanderContinuous-v2/
        
        """""
        
        
        self.reward = 0
        
        position = self.state[0:5:2]
        velocity = self.state[1:6:2]
        euler_angles = self.ang
        psi = self.ang[2]
        body_ang_vel = self.state[-3:]
        action = self.action
        action_hist = self.action_hist
        
        shaping = 100*(-norm(position/BB_POS)-norm(velocity/BB_VEL)-norm(psi/4)-0.3*norm(euler_angles[0:2]/BB_ANG))
        
        #CASCADING REWARDS
        r_state = np.concatenate((position,[psi]))        
        for TR_i,TR_Pi in zip(TR,TR_P): 
            if norm(r_state) < norm(np.ones(len(r_state))*TR_i):
                shaping += TR_Pi
                if norm(euler_angles) < norm(np.ones(3)*TR_i*2):
                    shaping += TR_Pi
                if norm(velocity) < norm(np.ones(3)*TR_i):
                    shaping += TR_Pi
                break
        
        if self.prev_shaping is not None:
            self.reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        
        #ABSOLUTE CONTROL PENALITY
                   
        abs_control = -np.sum(np.square(action - self.zero_control)) * P_C
        #AVERAGE CONTROL PENALITY        
        avg_control = -np.sum(np.square(action - np.mean(action_hist,0))) * P_C_D
        
        ## TOTAL REWARD SHAPING ##
        self.reward += + abs_control + avg_control
        
        #SOLUTION ACHIEVED?
        target_state = 12*(TR[0]**2)
        current_state = np.sum(np.square(np.concatenate((position, velocity, euler_angles, body_ang_vel))))      

        if current_state < target_state:
            self.reward = +500
            self.solved = 1
            # self.done = True 
            
        if self.i >= self.n and not self.done:
            self.reward = self.reward
            self.done = True
            self.solved = 0
            
        elif self.done:
            self.reward = -200
            self.solved = 0
         
            
class sensor():
    
    """Sensor class - simulates onboard sensors, given standard deviation and bias.
    Aimed to simulate kallman filters or to execute robust control, etc.
    Self explanatory, adds standard deviation noise and bias to quadrotor real state.
    
    """
    
    def __init__(self, env,
                 accel_std = 0.1, accel_bias_drift = 0.0005, 
                 gyro_std = 0.035, gyro_bias_drift = 0.00015, 
                 magnet_std = 15, magnet_bias_drift = 0.075, 
                 gps_std_p = 1.71, gps_std_v=0.5):
        
        self.std = [accel_std, gyro_std, magnet_std, gps_std_p, gps_std_v]
        self.b_d = [accel_bias_drift, gyro_bias_drift, magnet_bias_drift]
        self.quad = env
        self.error = True
        self.bias_reset()
   
    def bias_reset(self):        
        self.a_std = self.std[0]*self.error
        self.a_b_d = (np.random.random()-0.5)*2*self.b_d[0]*self.error        
        self.g_std = self.std[1]*self.error
        self.g_b_d = (np.random.random()-0.5)*2*self.b_d[1]*self.error        
        self.m_std = self.std[2]*self.error
        self.m_b_d = (np.random.random()-0.5)*2*self.b_d[2]*self.error        
        self.gps_std_p = self.std[3]*self.error
        self.gps_std_v = self.std[4]*self.error
    
        
    def accel(self):
    
        self.a_b_accel = self.a_b_accel + self.a_b_d*self.quad.t_step        
        read_error = np.random.normal(self.a_b_accel, self.a_std, 3)
        read_accel = np.dot(self.quad.mat_rot.T, self.quad.accel.flatten())
        return read_error+read_accel
    
    
    def gyro(self):
        
        self.g_b = self.g_b + self.g_b_d*self.quad.t_step
        
        read_error = np.random.normal(self.g_b, self.g_std, 3)
        read_gyro = self.quad.state[-3:].flatten()
        return read_error+read_gyro        
            
    def reset(self):
        self.a_b_grav = 0
        self.a_b_accel = 0
        self.m_b = 0
        self.g_b = 0
        self.acceleration_t0 = np.zeros(3)
        self.position_t0 = self.quad.state[0:5:2]
        self.velocity_t0 = self.quad.state[1:6:2]
        self.quaternion_t0 = self.quad.state[6:10]
        self.bias_reset()

    
    def gps(self):
        read_error_pos = np.random.normal(0, self.gps_std_p, 3)
        read_error_vel = np.random.normal(0, self.gps_std_v, 3)
        gps_pos = self.quad.state[0:5:2].flatten()
        gps_vel = self.quad.state[1:6:2].flatten()
        return read_error_pos+gps_pos, read_error_vel+gps_vel   
    
    def triad(self):
        gravity_vec = np.array([0, 0, -G])
        magnet_vec = np.array([-4047, 12911, -9899])*0.01 
        
        #Magnetic Vector of Santo André - Brasil in MiliGauss
        #https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
        
        
        self.a_b_grav = self.a_b_grav + self.a_b_d*self.quad.t_step
        self.m_b = self.m_b + self.m_b_d*self.quad.t_step
        
        #Gravity vector as read from body sensor
        gravity_body = np.dot(self.quad.mat_rot.T, gravity_vec) + np.random.normal(np.random.random(3)*self.a_b_grav, self.a_std, 3)
        #Magnetic Field vector as read from body sensor
        magnet_body = np.dot(self.quad.mat_rot.T, magnet_vec) + np.random.normal(np.random.random(3)*self.m_b, self.m_std, 3)      
      

        #Accel vector is more accurate
        #Body Coordinates
        t1b = gravity_body/np.linalg.norm(gravity_body)
        
        t2b = np.cross(gravity_body, magnet_body)
        t2b = t2b/np.linalg.norm(t2b)
        
        t3b = np.cross(t1b, t2b)
        t3b = t3b/np.linalg.norm(t3b)
        
        tb = np.vstack((t1b, t2b, t3b)).T
        
        
        #Inertial Coordinates
        t1i = gravity_vec/np.linalg.norm(gravity_vec)
        
        t2i = np.cross(gravity_vec, magnet_vec)        
        t2i = t2i/np.linalg.norm(t2i)
        
        t3i = np.cross(t1i, t2i)
        t3i = t3i/np.linalg.norm(t3i)
        
        ti = np.vstack((t1i, t2i, t3i)).T
        R = np.dot(tb, ti.T)
        q = Rotation.from_matrix(R.T).as_quat()
        q = np.concatenate(([q[3]], q[0:3]))
        return q, R.T
        
        
    def accel_int(self):
        accel_body = self.accel()      
        _, R = self.triad()             
        acceleration = np.dot(R, accel_body) 
       
        velocity = self.velocity_t0 + acceleration*self.quad.t_step
        position = self.position_t0 + velocity*self.quad.t_step
        
        self.acceleration_t0 = acceleration
        self.velocity_t0 = velocity
        self.position_t0 = position
        return acceleration, velocity, position
    
    def gyro_int(self):
        w = self.gyro()
        q = self.quaternion_t0
        V_q = deriv_quat(w, q).flatten()       
        for i in range(len(q)):
            q[i] = q[i] + V_q[i]*self.quad.t_step
        self.quaternion_t0 = q/np.linalg.norm(q)
        return q
        
        
class plotter(): 
        
    """""
    Render Class: Saves state and time until plot function is called.
                    Optionally: Plots a 3D graph of the position, with optional target position.
    
    init input:
        env: 
            class - quadrotor enviorment
        depth_plot:
            boolean - plot coordinates over time on 3D space
            
            
    add: saves a state and a time

    clear: clear memory

    plot: plot saved states     
    """""   
    
    def __init__(self, env, depth_plot=False):        
        plt.close('all')
        self.figure = plt.figure('States')
        self.depth_plot = depth_plot
        self.env = env
        self.states = []
        self.times = []
        self.print_list = range(10)
        self.plot_labels = ['x', 'y', 'z',
                            'phi', 'theta', 'psi', 
                            'u_1', 'u_2', 'u_3', 'u_4']
        
        self.line_styles = ['-', '-', '-',
                            '--', '--', '--', 
                            ':', ':', ':', ':']
        
            
    def add(self):
        state = np.concatenate((self.env.state[0:5:2].flatten(), self.env.ang.flatten(), self.env.clipped_action.flatten()))
        self.states.append(state)
        self.times.append(self.env.i*self.env.t_step)
        
    def clear(self,):
        self.states = []
        self.times = []
        
    def plot(self):
        plt.figure('States')
        self.states = np.array(self.states)
        self.times = np.array(self.times)
        for print_state, label, line_style in zip(self.print_list, self.plot_labels, self.line_styles):
            plt.plot(self.times, self.states[:,print_state], label = label, ls=line_style, lw=1)
        plt.legend()
        plt.grid(True)
        plt.show()
        if self.depth_plot:
            fig3d = plt.figure('3D map')
            ax = Axes3D(fig3d)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')       
            ax.set_zlabel('z (m)')    
            states = np.array(self.states)
            times = np.array(self.times)
            xs = self.states[:,0]
            ys = self.states[:,1]
            zs = self.states[:,2]
            t = self.times
            ax.scatter(xs,ys,zs,c=plt.cm.jet(t/max(t)))
            ax.plot3D(xs,ys,zs,linewidth=0.5)
            ax.set_xlim(-BB_POS, BB_POS)
            ax.set_ylim(-BB_POS, BB_POS)
            ax.set_zlim(-BB_POS, BB_POS)
            plt.grid(True)
            plt.show()
        self.clear()