from panda3d.core import loadPrcFile
loadPrcFile('./config/conf.prc')

import sys, os

#Panda 3D Imports
import panda3d
from panda3d.core import Filename
from direct.showbase.ShowBase import ShowBase

#Custom Functions
from environment.position import quad_position
from computer_vision.quadrotor_cv import computer_vision
from computer_vision.camera_calibration import calibration
from models.world_setup import world_setup, quad_setup
from models.camera_control import camera_control
from computer_vision.img_2_cv import opencv_camera

"""
INF209B − TÓPICOS ESPECIAIS EM PROCESSAMENTO DE SINAIS:

VISAO COMPUTACIONAL

PROJETO

RA: 21201920754
NOME: RAFAEL COSTA FERNANDES
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIÇÃO:
    Ambiente 3D de simulação do quadrirrotor. 
    Dividido em tarefas, Camera Calibration e Drone position
    Se não haver arquivo de calibração de camera no caminho ./config/camera_calibration.npz, realiza a calibração da câmera tirando 70 fotos aleatórias do padrão xadrez
    
    A bandeira IMG_POS_DETER adiciona a tarefa de determinação de posição por visão computacional, utilizando um marco artificial que já está no ambiente 3D, na origem. 
    A bandeira REAL_STATE_CONTROL determina se o algoritmo de controle será alimentado pelos valores REAIS do estado ou os valores estimados pelos sensores a bordo
    
    O algoritmo não precisa rodar em tempo real, a simulação está totalmente disvinculada da velocidade de renderização do ambiente 3D.
    Se a simulação 3D estiver muito lenta, recomendo mudar a resolução nativa no arquivo de configuração ./config/conf.prc
    Mudando a resolução PROVAVELMENTE será necessário recalibrar a câmera, mas sem problemas adicionais. 
    
    Recomendado rodar em um computador com placa de vídeo. 
    Em um i5-4460 com uma AMD R9-290:
        Taxa de FPS foi cerca de 95 FPS com a bandeira IMG_POS_DETER = False e REAL_STATE_CONTROL = True
        Taxa de FPS foi cerca de 35 FPS com a bandeira IMG_POS_DETER = True e REAL_STATE_CONTROL = False
    
    O algoritmo de detecção de pontos FAST ajudou na velocidade da detecção, mas a performance precisa ser melhorada 
    (principalmente para frames em que o tabuleiro de xadrez aparece parcialmente)
    
    
"""


# POSITION AND ATTITUDE ESTIMATION BY IMAGE, ELSE BY MEMS SENSORS
IMG_POS_DETER = True

# REAL STATE CONTROL ELSE BY ESTIMATION METHODS
REAL_CTRL = False

# HOVER FLIGHT ELSE RANDOM INITIAL STATE
HOVER = True

# NUMBER OF EPISODE TIMESTEPS 
EPISODE_STEPS = 3000

# TOTAL NUMBER OF EPISODES
ERROR_AQS_EPISODES = 40

# PATH TO ERROR DATALOG
ERROR_PATH = './data/hover/' if HOVER else './data/random_initial_state/'

mydir = os.path.abspath(sys.path[0])

mydir = Filename.fromOsSpecific(mydir).getFullpath()

class MyApp(ShowBase):
    def __init__(self):
        
        ShowBase.__init__(self)       
        render = self.render
        
        # CAMERA NEUTRAL POSITION
        self.cam_neutral_pos = panda3d.core.LPoint3f(5, 5, 7)

        self.cam_1 = opencv_camera(self, 'cam_1')
        self.cam_2 = opencv_camera(self, 'cam_2')
        
        # MODELS SETUP
        world_setup(self, render, mydir)
        quad_setup(self, render, mydir)
        
        # CALIBRATION ALGORITHM
        self.camera_cal = calibration(self, self.cam_2)    
        
        if self.camera_cal.calibrated:
            self.run_setup()
            
    def run_setup(self):
        # DRONE POSITION
        self.drone = quad_position(self, self.quad_model, self.prop_models, EPISODE_STEPS, REAL_CTRL, IMG_POS_DETER, ERROR_AQS_EPISODES, ERROR_PATH, HOVER)
        
        # COMPUTER VISION
        self.cv = computer_vision(self, self.quad_model, self.drone.env, self.drone.sensor, self.drone, self.cam_1, self.cam_2, self.camera_cal, mydir, IMG_POS_DETER)        
        
        # CAMERA CONTROL
        camera_control(self, self.render) 
app = MyApp()
app.run()
