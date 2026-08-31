# This file contains functions used for footstep planning of Kondo khr3hv using SIP model
# Author : Sunil Gora, Shakti S. Gupta and Ashish Dutta
import numpy as np
import os
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from copy import deepcopy


def selectRobot(num,vel,step_len,spno):
    dirname = os.path.dirname(__file__) #os.getcwd() 
    xml_dir=dirname #os.path.join(os.path.dirname(os.path.dirname(__file__)) + "/MuJoCo/Humanoid robot/") #os.getcwd()
    #Generate XML for scene
    xml_path='/scene_basic.xml' #load basic xml file of scene
    xml_path = os.path.join(dirname + xml_path)
    xml_tree = ET.parse(xml_path) 
    #Modify xml_tree for 3D terrain generation
    #Stairs
    trnnum=20
    trnheight=[i *0.00 for i in range(trnnum)] #height of each step
    xml_str = scenegen(xml_tree,trnnum=trnnum,trnlength=step_len,trnwidth=0.50,trnheight=trnheight)
    # print(xml_str)
    # print(asd)
    xml_tree = ET.ElementTree(ET.fromstring(xml_str))

    # make cases for different robots
    if num==1: #Kondo khr3hv
        xml_path= '/kondo/scene_Plane.xml' #Scene
        xml_path = os.path.join(xml_dir + xml_path)
        xml_tree = ET.parse(xml_path) 

        robotpath='/kondo/kondo_khr3hv.xml' #Robot
        robotpath = os.path.join(xml_dir + robotpath)
        #robotpath='example/model/humanoid/humanoid.xml' #Robot <!-- Include sites at hip and both foot -->
        xml_str=addrobot2scene(xml_tree,robotpath)

        # MuJoCo data structures
        model = mujoco.MjModel.from_xml_string(xml_str)  # MuJoCo model
        data = mujoco.MjData(model)  # MuJoCo data
        # cam = mujoco.MjvCamera()                        # Abstract camera
        # opt = mujoco.MjvOption()                        # visualization options
        
        model.opt.timestep=0.0001
        
        ub_jnts=np.arange(18,model.nv)
        left_legjnts=np.arange(6,12)
        right_legjnts=np.arange(12,18)
        foot_size = np.array([0.050, 0.040]) #50mm,40mm np.array([length, width])
        
        #init_controller(model, data)
        Kp=np.zeros(model.nu)
        Kv=np.zeros(model.nu)
        Ki=np.zeros(model.nu)
        Kp[0:12]=10 #7
        Kv[0:12]=.1  #0.5 #0.003
        #set lower gains for ankle joints -- not required for active walking with nonzero ankle height
        # Kp[4:6]=0.001
        # Kv[4:6]=0.00001
        # Kp[10:12]=0.001
        # Kv[10:12]=0.00001

        # Ki[0:12]=0.1
        Kp[12:]=1
        Kv[12:]=0.01 #0.05
        # Ki[12:]=0.01
        # Kp=10*Kp
        # Kv=10*Kv
        #FW
        # Kp[-1:]=10
        # Kv[-1:]=0.01

        # data to humanoid parameters
        humn=myRobot(ub_jnts,left_legjnts,right_legjnts,foot_size,vel)

        humn.mj2humn(model,data)

        # Initial joint angles and velocity
        q0=humn.data2q(data)
        q0[[2,9,10,15,16]]=[0.95*q0[2],0.5,-0.5,0.5,-0.5]

        # Initial COM position
        humn.r_com[0]=-0.0#1#-0.01
        humn.r_com[1]=(0+humn.o_left[1])/3
        humn.r_com[2]=0.9*humn.r_com[2]

        #Step length and height
        # step_len=0.01 # Max Steplength
        # step_time= step_len/(2*vel) #0.2
        zSw=0.03 #swing foot lift
        cam_dist=0.75 #camera distance

    elif num==2: #Unitree G1
        xml_path= 'sceneG1.xml' #Scene
        #xml_path= 'scene_3DT.xml' #Scene
        xml_path = os.path.join(dirname + "/" + xml_path)
        xml_tree = ET.parse(xml_path) 
        robotpath='/unitree_g1/g1.xml' #Robot <!-- Include sites at hip and both foot -->
        robotpath = os.path.join(xml_dir + robotpath)
        xml_str=addrobot2scene(xml_tree,robotpath)

        # MuJoCo data structures
        model = mujoco.MjModel.from_xml_string(xml_str)  # MuJoCo model
        data = mujoco.MjData(model)  # MuJoCo data
        # cam = mujoco.MjvCamera()                        # Abstract camera
        # opt = mujoco.MjvOption()                        # visualization options

        # model.opt.timestep=0.0001

        ub_jnts=np.arange(18,model.nv)
        left_legjnts=np.arange(6,12)
        right_legjnts=np.arange(12,18)
        foot_size = np.array([0.17, 0.055]) #50mm,40mm np.array([length, width])

        #init_controller(model, data)
        Kp=np.zeros(model.nu)
        Kv=np.zeros(model.nu)
        Ki=np.zeros(model.nu)
        Kp[0:12]=5000 #7
        Kv[0:12]=50  #0.003
        Kp[3]=10000
        Kv[3]=100
        Kp[9]=10000
        Kv[9]=100
        Kp[12:]=50
        Kv[12:]=1

        # data to humanoid parameters
        humn=myRobot(ub_jnts,left_legjnts,right_legjnts,foot_size,vel)

        humn.mj2humn(model,data)

        # Initial joint angles and velocity
        q0=humn.data2q(data)
        q0[[2,9,10,15,16]]=[0.95*q0[2],0.5,-0.5,0.5,-0.5]

        # Initial COM position
        humn.r_com[0]=-0.0#1#-0.01
        humn.r_com[1]=(0+humn.o_left[1])/3
        humn.r_com[2]=0.95*humn.r_com[2]

        humn.o_left[1]=0.5*humn.o_left[1]
        humn.o_right[1]=0.5*humn.o_right[1]


        #Step length, time and height
        # step_len=0.1 # Max Steplength
        # step_time= step_len/(2*vel) #0.5
        zSw=0.05 #swing foot lift
        cam_dist=2 #camera distance

    elif num==3: #Biped
        # xml_path= 'scene_bipedT.xml' #Scene
        # xml_path = os.path.join(dirname + "/" + xml_path)
        # xml_tree = ET.parse(xml_path) 
        robotpath='CARS/urdf/robot.xml' #Robot
        robotpath = os.path.join(dirname + "/" + robotpath)
        xml_str=addrobot2scene(xml_tree,robotpath)

        # MuJoCo data structures
        model = mujoco.MjModel.from_xml_string(xml_str)  # MuJoCo model
        data = mujoco.MjData(model)  # MuJoCo data
        # cam = mujoco.MjvCamera()                        # Abstract camera
        # opt = mujoco.MjvOption()                        # visualization options

        ub_jnts=np.empty(0)
        left_legjnts=np.arange(12,18)
        right_legjnts=np.arange(6,12)
        foot_size = np.array([0.264, 0.194]) #50mm,40mm np.array([length, width])

        #init_controller(model, data)
        Kp=np.zeros(model.nu)
        Kv=np.zeros(model.nu)
        Ki=np.zeros(model.nu)
        Kp[0:12]=5000 #7
        Kv[0:12]=50  #0.003
        Kp[3]=10000
        Kv[3]=100
        Kp[9]=10000
        Kv[9]=100
        Kp[12:]=50
        Kv[12:]=1

        # data to humanoid parameters
        humn=myRobot(ub_jnts,left_legjnts,right_legjnts,foot_size,vel)

        humn.mj2humn(model,data)


        # Initial joint angles and velocity
        q0=humn.data2q(data)
        #q0[7:]=0.1+q0[7:]
        q0[[2,9,11,15,17]]=[1*q0[2],0.5,-0.5,0.5,-0.5]

        # Initial COM position
        humn.r_com[0]=-0.0#1#-0.01
        humn.r_com[1]=(0+humn.o_left[1])/3
        humn.r_com[2]=1.1*humn.r_com[2] #0.9*humn.r_com[2]


        #Step length and height
        # step_len=0.1 # Max Steplength
        # step_time= step_len/(2*vel) #0.5
        zSw=0.05 #swing foot lift
        cam_dist=2.5 #camera distance


    #Walking pattern mode
    humn.SIPwalk=True #False #True #False
    humn.MPC_LIPM=False #True #True #True #True

    if humn.MPC_LIPM==True:
        spno=2 #1-SSP, 2-DSP
    else:
        spno=1 #1-SSP, 2-DSP

    # Set pos and orientation
    # print('o_left=',humn.o_left)
    humn.o_left[0]=0.0
    humn.o_left[2]=0#-model.geom_solimp[0][2]/2
    # print('o_right=',humn.o_right)
    humn.o_right[0]=0.0
    humn.o_right[2]=-0.0

    data.qvel[0]=vel#0.25#5#0.1 #0.1
    data.qvel[1]=0.0#5#0.1 #0.1

    humn.spno = spno  # SSP=1, DSP=2

    if humn.spno==2:
        data.qvel[0]=0#0.55#0.1 #0.1
        humn.r_com[1]=0.0#(0+humn.o_left[1])/3
        humn.r_com[2]=0.8/0.9*humn.r_com[2]


    #print(q0)
    # Find initial IK solution
    # q=numik(model,data,q0,1,humn.r_com,humn.o_left,humn.o_right,humn.ub_jnts,0)
    ti=0
    while ti<100:
        delt=1/100
        q,gradH0=humn.numik(model,data,q0,delt,humn.r_com,humn.o_left,humn.o_right,humn.ub_jnts,np.zeros([model.nv]),False,0,0)
        q0=q.copy()
        ti +=delt

    data=humn.q2data(data,q)
    humn.q0=q.copy()
    # print("Initial joint coord q0=",q0)
    #glfw.terminate()

    # Humanoid parameters
    humn.mj2humn(model,data)
    print('mass=',humn.m,' rCOM=',humn.r_com,' vCOM=',humn.v_com)
    print('o_left=',humn.o_left, 'o_right=',humn.o_right)
    # Check leg transition required
    humn.xlimft = 0.5 #10*step_len # Max Steplength
    humn.ylimft = abs(humn.o_left[1]-humn.o_right[1])   #max(0.1 * l, 2 * abs(qcm[1] - qcp[1]))
    # humn.spno = 1  # SSP=1, DSP=2
    humn.Stlr=np.array([1, 0])
    humn.zSw=zSw #swing foot lift
    humn.step_time= step_len/(2*vel) #step_time for MPC
    humn.step_len=step_len #steplength for MPC
    humn.sspbydsp=2#3 #2
    humn.Tsip = 0  # Cycle Time for footstep control
    humn.cam_dist=cam_dist


    # initialize the controller
    # humn.init_controller(Kp, Kv,Ki)
    humn.Kp=Kp
    humn.Kv=Kv
    humn.Ki=Ki
    if num==2: #G1 is position controlled
        humn.posCTRL=True #Control mode is Position or Torque

    return humn,model,data


#MuJoCo model to Humanoid Robot data
class myRobot:
    def __init__(self, ub_jnts,left_legjnts,right_legjnts,foot_size,vel):
        #Design
        self.ub_jnts=ub_jnts
        self.left_legjnts=left_legjnts
        self.right_legjnts=right_legjnts
        self.foot_size=foot_size
        self.cam_dist=1
        #Control
        self.vel=vel
        self.Tsip = 0  # Cycle Time for footstep control
        self.WD = 0 # Work Done
        self.posCTRL=False #Control mode is Position or Torque
        self.KINctrl=False #State estimation and traj correction using FK
        self.ZMPctrl=0#-1/(10**5)#.0001#01
        self.AMctrl=0.0
        self.k_ub=0.1000 #task priority weight for ub
        self.k_L=self.AMctrl*1/(1+self.k_ub*0*np.linalg.norm(np.zeros([6])))
        self.FWctrl=0
        self.ftctrl=0
        # Terrain parameters
        self.plno = 0  # 0-default 1-virtual terrain
        self.zpln = 0  # height of terrain plane

        #Identification
        self.xn = np.array([])
        self.dxn = np.array([])
        self.fn = np.array([])
        self.AM_CMspl=[]
        for i in range(3):
            self.AM_CMspl.append(CubicSpline([0,1], [0,0], bc_type='clamped')) #torque on COM due to change of AM

    def mj2humn(self,model,data):
        # Forward kinematics for position and velocity terms
        # Forward Position kinematics
        mujoco.mj_fwdPosition(model, data)
        # mujoco.mj_kinematics(model, data)
        mujoco.mj_comVel(model, data)
        mujoco.mj_subtreeVel(model, data)
        self.ti=data.time
        self.m =mujoco.mj_getTotalmass(model) #mass
        self.r_com=data.subtree_com[0].copy() #com position
        self.v_com=data.subtree_linvel[0].copy()  #com velocity
        #CRB inertia about COM in mujoco from data.crb # Extract inertia matrix (3x3 upper triangle)
        # print('crb Inertia matrix:',data.crb)
        # print('uper triangle of crb[1] inertia matrix of body-1: crb[1][:6]=Ixx,Iyy,Izz,Ixy,Ixz,Iyz:',data.crb[1][0:6]) 
        #Find inerrtia matrix about COM Ic=Ib #+ 1/m*skew(crb[1][6:9])*skew(crb[6:9])
        self.I=np.array([[data.crb[1][0],data.crb[1][3],data.crb[1][4]],
                         [data.crb[1][3],data.crb[1][1],data.crb[1][5]],
                         [data.crb[1][4],data.crb[1][5],data.crb[1][2]]]) 
        # print('Inertia matrix about COM Ic=',self.I)
        # Angular momentum matrix
        self.Iwb = np.zeros([3, model.nv])
        mujoco.mj_angmomMat(model, data, self.Iwb, 0)
        # print(asd)
        self.dq_com=data.cvel[1].copy() # com rot:lin velocity
        # Foot positions
        self.o_left=data.site('left_foot_site').xpos.copy()  # current Left foot position
        self.R_left = data.site('left_foot_site').xmat.copy() # current left foot orientation
        self.quat_left = euler2quat(mat2euler(self.R_left.reshape(3,3))) # left foot quaternion
        self.Jv_left = np.zeros((3, model.nv))  # Left foot center jacobian
        self.Jw_left = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, self.Jv_left, self.Jw_left, model.site('left_foot_site').id)
        self.v_left = self.Jv_left @ data.qvel  # current left foot linear velocity
        
        self.o_right=data.site('right_foot_site').xpos.copy()  # current Right foot position
        self.R_right = data.site('right_foot_site').xmat.copy() # current right foot orientation
        self.quat_right = euler2quat(mat2euler(self.R_right.reshape(3,3))) # right foot quaternion
        self.Jv_right = np.zeros((3, model.nv))  # right foot center jacobian
        self.Jw_right = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, self.Jv_right, self.Jw_right, model.site('right_foot_site').id)
        self.v_right = self.Jv_right @ data.qvel  # current right foot linear velocity

        # Foot edges positions
        # self.oL_edges=np.zeros([4,3]) #4 edges of left foot
        # self.oR_edges=np.zeros([4,3]) #4 edges of right foot
        # # Left foot edges by frame transform of left_foot_site
        # R_lf= data.site('left_foot_site').xmat.reshape(3,3)
        # self.oL_edges[0,:]=self.o_left + R_lf @ np.array([ self.foot_size[0],  self.foot_size[1],0]) #Front-Right
        # self.oL_edges[1,:]=self.o_left + R_lf @ np.array([ self.foot_size[0], -self.foot_size[1],0]) #Front-Left
        # self.oL_edges[2,:]=self.o_left + R_lf @ np.array([-self.foot_size[0], -self.foot_size[1],0]) #Back-Left
        # self.oL_edges[3,:]=self.o_left + R_lf @ np.array([-self.foot_size[0],  self.foot_size[1],0]) #Back-Right
        # # Right foot edges by frame transform of right_foot_site
        # R_rf= data.site('right_foot_site').xmat.reshape(3,3)
        # self.oR_edges[0,:]=self.o_right + R_rf @ np.array([ self.foot_size[0],  self.foot_size[1],0]) #Front-Right
        # self.oR_edges[1,:]=self.o_right + R_rf @ np.array([ self.foot_size[0], -self.foot_size[1],0]) #Front-Left
        # self.oR_edges[2,:]=self.o_right + R_rf @ np.array([-self.foot_size[0], -self.foot_size[1],0]) #Back-Left
        # self.oR_edges[3,:]=self.o_right + R_rf @ np.array([-self.foot_size[0],  self.foot_size[1],0]) #Back-Right


        #self.q
        self.data2q(data) #write self.q and self.dq
        self.q_err = np.zeros([model.nv])
        self.Eqdt=0*data.qvel #Integral error
        self.En_init=1/2*self.m*np.linalg.norm(self.v_com)**2 + self.m*9.81*self.r_com[2] #Initial energy
        self.gradH = np.ones([model.nv]) #Gradient for joint limits check

        #Traj save
        #Save desired and Simulation traj data
        if data.time==0:
            #Save desired and Simulation traj data
            self.tTraj=data.time
            #joint traj
            self.qTraj_des=np.array([self.q.copy()])
            self.dqTraj_des=self.dq.copy()
            self.qTraj_act=self.q.copy()
            self.dqTraj_act=self.dq.copy()
            #COM traj
            self.rcomTraj_act=np.array([self.r_com.copy()])
            self.rcomTraj_des=np.array([self.r_com.copy()])
            #COP traj
            self.rcopTraj_act=np.array([[np.nan,np.nan,0]])
            self.rcopTraj_des=np.array([[np.nan,np.nan,0]])
            #Foot traj
            self.oLTraj_act=np.array([self.o_left.copy()])
            self.oLTraj_des=np.array([self.o_left.copy()])
            self.quatLTraj_act=np.array([self.quat_left.copy()])
            self.oRTraj_act=np.array([self.o_right.copy()])
            self.oRTraj_des=np.array([self.o_right.copy()])
            self.quatRTraj_act=np.array([self.quat_right.copy()])
            #Foot edges traj
            # self.oL_edgesTraj_act=np.array([self.oL_edges.copy()])
            # self.oR_edgesTraj_act=np.array([self.oR_edges.copy()])
            #CAM traj
            self.CAMTraj=np.zeros(3)
            self.torqueTraj=np.zeros(model.nu)
            #Contact force traj
            self.fLTraj=np.zeros(6)
            self.fRTraj=np.zeros(6)


        self.model=model
        self.data=data    

    def updateTrajData(self,model,data):
        if data.time==0:
            #Save desired and Simulation traj data
            self.tTraj=data.time
            #joint traj
            self.qTraj_des=np.array([self.q.copy()])
            self.dqTraj_des=self.dq.copy()
            self.qTraj_act=self.q.copy()
            self.dqTraj_act=self.dq.copy()
            #COM traj
            self.rcomTraj_act=np.array([self.r_com.copy()])
            self.rcomTraj_des=np.array([self.r_com.copy()])
            #COP traj
            self.rcopTraj_act=np.array([[np.nan,np.nan,0]])
            self.rcopTraj_des=np.array([[np.nan,np.nan,0]])
            #Foot traj
            self.oLTraj_act=np.array([self.o_left.copy()])
            self.oLTraj_des=np.array([self.o_left.copy()])
            self.quatLTraj_act=np.array([self.quat_left.copy()])
            self.oRTraj_act=np.array([self.o_right.copy()])
            self.oRTraj_des=np.array([self.o_right.copy()])
            self.quatRTraj_act=np.array([self.quat_right.copy()])

            #Foot edges traj
            # self.oL_edgesTraj_act=np.array([self.oL_edges.copy()])
            # self.oR_edgesTraj_act=np.array([self.oR_edges.copy()])
            #CAM traj
            self.CAMTraj=np.zeros(3)
            self.torqueTraj=np.zeros(model.nu)
            #Contact force traj
            self.fLTraj=np.zeros(6)
            self.fRTraj=np.zeros(6)
        else:
            # Save desired and Simulation traj data
            self.tTraj=np.vstack((self.tTraj,data.time))
            #joint traj
            self.qTraj_des=np.vstack((self.qTraj_des,self.q.copy()))
            self.dqTraj_des=np.vstack((self.dqTraj_des,self.dq.copy()))
            self.qTraj_act=np.vstack((self.qTraj_act,self.q.copy()))    
            self.dqTraj_act=np.vstack((self.dqTraj_act,self.dq.copy()))
            #COM traj
            self.rcomTraj_act=np.vstack((self.rcomTraj_act,self.r_com.copy()))
            self.rcomTraj_des=np.vstack((self.rcomTraj_des,self.ocm_des.copy()))
            #COP traj
            self.rcopTraj_act=np.vstack((self.rcopTraj_act,self.r_cop.copy()))
            self.rcopTraj_des=np.vstack((self.rcopTraj_des,self.ocp_des.copy()))
            #Foot traj
            self.oLTraj_act=np.vstack((self.oLTraj_act,self.o_left.copy()))
            self.oLTraj_des=np.vstack((self.oLTraj_des,self.oL_des.copy()))
            self.quatLTraj_act=np.vstack((self.quatLTraj_act,self.quat_left.copy()))
            self.oRTraj_act=np.vstack((self.oRTraj_act,self.o_right.copy()))
            self.oRTraj_des=np.vstack((self.oRTraj_des,self.oR_des.copy()))
            self.quatRTraj_act=np.vstack((self.quatRTraj_act,self.quat_right.copy()))
            #Foot edges traj
            # self.oL_edgesTraj_act=np.vstack((self.oL_edgesTraj_act,self.oL_edges.copy()))
            # self.oR_edgesTraj_act=np.vstack((self.oR_edgesTraj_act,self.oR_edges.copy()))
            #CAM traj
            self.CAMTraj=np.vstack((self.CAMTraj,self.Iwb @ self.dq))
            #Joint torques
            self.torqueTraj=np.vstack((self.torqueTraj,data.actuator_force.copy()))
            #Contact force on left foot
            self.fLTraj=np.vstack((self.fLTraj,data.sensordata[0:6].copy()))
            #Contact force on right foot
            self.fRTraj=np.vstack((self.fRTraj,data.sensordata[6:12].copy()))
            #Work done


    # Copy data.qpos (with quaternion) to q (with euler angles)
    def data2q(self,data):
        self.q = 0 * data.qvel.copy()
        self.dq = 0 * data.qvel.copy()
        qqt = data.qpos[3:7].copy()
        qeulr = quat2euler(qqt)
        for i in np.arange(0, 3):
            self.q[i] = data.qpos[i].copy()
            self.dq[i] = data.qvel[i].copy()
        for i in np.arange(3, 6):
            self.q[i] = qeulr[i - 3].copy()
            self.dq[i] = data.qvel[i].copy()
        for i in np.arange(6, len(data.qvel)):
            self.q[i] = data.qpos[i + 1].copy()
            self.dq[i] = data.qvel[i].copy()
        return self.q

    # Copy q (with euler angles) to data.qpos (with quaternion)
    def q2data(self,data,q0):
        import copy
        qqt=euler2quat(q0[3:6])
        for i in np.arange(0,3):
            data.qpos[i]=copy.copy(q0[i])
        for i in np.arange(3,7):
            data.qpos[i]=copy.copy(qqt[i-3])
        for i in np.arange(7,len(data.qpos)):
            data.qpos[i]=copy.copy(q0[i-1])
        return data
    
    def plotData(self):
        #Plot des and act joint traj
        dataidx=np.arange(0,len(self.tTraj),int(max(len(self.tTraj)/50,1 / (25 * (self.tTraj[1]-self.tTraj[0])))))  # Plot at 20 Hz
        self.plotqTraj(self.tTraj[dataidx],self.qTraj_des[dataidx],linestyl='o',overlap=0)
        self.plotqTraj(self.tTraj,self.qTraj_act,linestyl='-',overlap=1)
        plt.savefig('qtraj.pdf', bbox_inches='tight', pad_inches=0.1)
        np.savetxt('3DSIPs_eqActUB.dat', np.hstack((self.tTraj.reshape(-1,1), np.linalg.norm(self.qTraj_act[:,self.ub_jnts], axis=1).reshape(-1,1))), header='Time(s) qUB_Norm', comments='')

        #Plot COM traj
        self.plotCOMTraj()
        plt.savefig('COMtraj.pdf', bbox_inches='tight', pad_inches=0.1)
        #Plot COM and COP xy traj
        self.plotxyTraj()
        plt.savefig('xytraj.pdf', bbox_inches='tight', pad_inches=0.1)
        #plot Hip and Foot x-z traj
        self.plotxzTraj()
        plt.savefig('xztraj.pdf', bbox_inches='tight', pad_inches=0.1)
        self.plotCAMTraj(self.tTraj,self.CAMTraj)
        plt.savefig('CAMtraj.pdf', bbox_inches='tight', pad_inches=0.1)
        # self.pltfc()
        # self.pltcom()
        # self.pltwd()
        plt.show()
    
    def plotqTraj(self,ttraj,qtraj,linestyl='-',overlap=0):
        #Plot joint angles in subplots
        # print(np.shape(ttraj),np.shape(qtraj))
        if overlap==0:
            fig1, self.qpltAaxis = plt.subplots(nrows=5, sharex=True,figsize=(7, 8))  # joint angles
            #Text for desired as dashed and Simulation as solid
            fig1.suptitle(r'$\circ$ Desired $-$ Simulation',y=0.98)
            self.qpltAaxis[len(self.qpltAaxis)-1].set_xlabel('Time (s)')
            for i in range(len(self.qpltAaxis)):
                self.qpltAaxis[i].set_ylabel('Angle (deg)')

        else:
            for i in range(len(self.qpltAaxis)):
                for line in self.qpltAaxis[i].lines:
                    line.set_label("_nolegend_")
                # self.qpltAaxis[i].legend_.remove()   # remove old legend

        ax1=self.qpltAaxis
        col = ['r', 'g', 'b', 'c', 'm', 'k']
        for i in self.left_legjnts:
            ax1[0].plot(ttraj, qtraj[:, i] * 180 / np.pi, linestyl, color=col[i - min(self.left_legjnts)],
                        label='$\u03B8_{' + str(i - min(self.left_legjnts) + 1) + '}$')
        for i in self.right_legjnts:
            ax1[1].plot(ttraj, qtraj[:, i] * 180 / np.pi, linestyl, color=col[i - min(self.right_legjnts)],
                        label='$\u03B8_{' + str(i - min(self.right_legjnts) + 1 + len(self.left_legjnts)) + '}$')

        for i in self.ub_jnts[0:3]:
            ax1[2].plot(ttraj, qtraj[:, i] * 180 / np.pi, linestyl, color=col[(i - min(self.ub_jnts))%6], label='$\u03B8_{' + str(
                i - min(self.ub_jnts) + 1 + len(self.left_legjnts) + len(self.right_legjnts)) + '}$')
        for i in self.ub_jnts[3:8]:
            ax1[3].plot(ttraj, qtraj[:, i] * 180 / np.pi, linestyl, color=col[i - min(self.ub_jnts) - 2], label='$\u03B8_{' + str(
                i - min(self.ub_jnts) + 1 + len(self.left_legjnts) + len(self.right_legjnts)) + '}$')
        for i in self.ub_jnts[8:12]:
            ax1[4].plot(ttraj, qtraj[:, i] * 180 / np.pi, linestyl, color=col[i - min(self.ub_jnts) - 6], label='$\u03B8_{' + str(
                i - min(self.ub_jnts) + 1 + len(self.left_legjnts) + len(self.right_legjnts)) + '}$')

        for i in range(len(ax1)):
            ax1[i].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.2), ncol=max(1,len(ax1[i].lines)))


    def plotHipTraj(self,ttraj,qtraj,linestyl='-',overlap=0):
        #Plot joint angles in subplots
        # print(np.shape(ttraj),np.shape(qtraj))
        if overlap==0:
            fig1, self.qpltAaxis = plt.subplots(nrows=2, sharex=True)  # Hip
            #Text for desired as dashed and Simulation as solid
            # fig1.suptitle(r'$\circ$ Desired $-$ Simulation',y=0.98)
            self.qpltAaxis[len(self.qpltAaxis)-1].set_xlabel('Time (s)')
            self.qpltAaxis[0].set_ylabel('Position (m)')
            self.qpltAaxis[1].set_ylabel('Angle (deg)')
            # self.qpltAaxis[1].set_ylim([-1,1])


        else:
            for i in range(len(self.qpltAaxis)):
                for line in self.qpltAaxis[i].lines:
                    line.set_label("_nolegend_")
                # self.qpltAaxis[i].legend_.remove()   # remove old legend

        ax1=self.qpltAaxis
        col = ['r', 'g', 'b']
        ax1[0].plot(ttraj, qtraj[:, 0], linestyl, color=col[0],label='x')
        ax1[0].plot(ttraj, qtraj[:, 1], linestyl, color=col[1],label='y')
        ax1[0].plot(ttraj, qtraj[:, 2], linestyl, color=col[2],label='z')
        ax1[1].plot(ttraj, qtraj[:, 3]* 180 / np.pi, linestyl, color=col[0],label='roll')
        ax1[1].plot(ttraj, qtraj[:, 4]* 180 / np.pi, linestyl, color=col[1],label='pitch')
        ax1[1].plot(ttraj, qtraj[:, 5]* 180 / np.pi, linestyl, color=col[2],label='yaw')

        for i in range(len(ax1)):
            ax1[i].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.2), ncol=max(1,len(ax1[i].lines)))

        #scale and save
        plt.tight_layout()
        plt.savefig('hip_traj.png', dpi=300, bbox_inches='tight')

    def plotCOMTraj(self):
        #Plot COM in subplots
        fig1, self.COMpltAaxis = plt.subplots(nrows=3, sharex=True)  # joint angles
        #Text for desired as dashed and Simulation as solid
        fig1.suptitle(r'$--$ Desired $-$ Simulation',y=0.98)
        self.COMpltAaxis[-1].set_xlabel('Time (s)')

        ax1=self.COMpltAaxis
        # col = ['r', 'g', 'b', 'c', 'm', 'k']
        ax_lab=['$X$ (m)','$Y$ (m)','$Z$ (m)']
        
        for i in range(3):
            #COM
            ax1[i].plot(self.tTraj, self.rcomTraj_des[:, i], 'b--',label='_hidden')
            ax1[i].plot(self.tTraj, self.rcomTraj_act[:, i], 'b-',label='COM')
            #COP
            ax1[i].plot(self.tTraj, self.rcopTraj_des[:, i], 'g--',label='_hidden',linewidth=1.0)
            ax1[i].plot(self.tTraj, self.rcopTraj_act[:, i], 'g-',label='COP',linewidth=1.0)
            ax1[i].set_ylabel(ax_lab[i])
            if i==0:
                ax1[i].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.3), ncol=max(1,len(ax1[i].lines)))
            #Tracking error in second yaxis in red color without label and color the twin axis to red            
            ax2 = ax1[i].twinx()
            ax2.plot(self.tTraj, self.rcomTraj_des[:, i] - self.rcomTraj_act[:, i], 'r-', label='_hidden', linewidth=0.5)
            ax2.set_ylabel('Error (m)', color='r')
            ax2.tick_params(axis='y', colors='r')
            ax2.spines['right'].set_color('r')
        #Save time and error data to dat file for tikz plot
        np.savetxt('3DSIPs_eCOM.dat', np.hstack((self.tTraj.reshape(-1,1), (self.rcomTraj_des - self.rcomTraj_act))), header='Time(s) COM_X_error(m) COM_Y_error(m) COM_Z_error(m)', comments='')


    def plotxyTraj(self):
        #Plot COM x-y traj
        fig1, self.xypltAaxis = plt.subplots(nrows=1, sharex=True,figsize=(4,3))  # joint angles
        #Text for desired as dashed and Simulation as solid
        fig1.suptitle(r'$--$ Desired $-$ Simulation',y=0.98)
        self.xypltAaxis.set_xlabel('X (m)')
        self.xypltAaxis.set_ylabel('Y (m)')
        #Plot foot edges trajectory
        for i in range(len(self.tTraj)):
            if self.oLTraj_act[i,2]<0: #Replace with foot contact state for accuracy
                leftfoot_edges = (quat2mat(self.quatLTraj_act[i,:]) @ np.array([
                [ self.foot_size[0]/2, self.foot_size[1]/2, 0],
                [ self.foot_size[0]/2, -self.foot_size[1]/2, 0],
                [-self.foot_size[0]/2, -self.foot_size[1]/2, 0],
                [-self.foot_size[0]/2, self.foot_size[1]/2, 0],
                [ self.foot_size[0]/2, self.foot_size[1]/2, 0],
                ]).T).T + self.oLTraj_act[i,:]
                if i==0:
                    self.xypltAaxis.plot(leftfoot_edges[:,0],leftfoot_edges[:,1],'k.-',label='Support boundary')
                else:
                    self.xypltAaxis.plot(leftfoot_edges[:,0],leftfoot_edges[:,1],'k.-',label='_hidden')
                
            if self.oRTraj_act[i,2]<0: 
                rightfoot_edges = (quat2mat(self.quatRTraj_act[i,:]) @ np.array([
                [ self.foot_size[0]/2, self.foot_size[1]/2, 0],
                [ self.foot_size[0]/2, -self.foot_size[1]/2, 0],
                [-self.foot_size[0]/2, -self.foot_size[1]/2, 0],
                [-self.foot_size[0]/2, self.foot_size[1]/2, 0],
                [ self.foot_size[0]/2, self.foot_size[1]/2, 0],
                ]).T).T + self.oRTraj_act[i,:]
                self.xypltAaxis.plot(rightfoot_edges[:,0],rightfoot_edges[:,1],'k.-',label='_hidden')

        ax1=self.xypltAaxis
        #COM desired
        ax1.plot(self.rcomTraj_des[:, 0], self.rcomTraj_des[:, 1], 'b--',label='_hidden')
        #COM Simulation
        ax1.plot(self.rcomTraj_act[:, 0], self.rcomTraj_act[:, 1], 'b-',label='COM')
        #COP desired
        ax1.plot(self.rcopTraj_des[:, 0], self.rcopTraj_des[:, 1], 'g--',label='_hidden',linewidth=1.0)
        #COP Simulation
        ax1.plot(self.rcopTraj_act[:, 0], self.rcopTraj_act[:, 1], 'g-',label='COP',linewidth=1.0)
        ax1.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.1), ncol=max(1,len(ax1.lines)))

    def plotxzTraj(self):
        #Plot COM x-z traj
        #broken y axis
        fig1, self.xzpltAaxis = plt.subplots(nrows=2, sharex=True)  # cartesian plot with broken y-axis
        #Text for desired as dashed and Simulation as solid
        fig1.suptitle(r'$--$ Desired $-$ Simulation',y=0.98)
        fig1.supylabel('Z (m)', fontsize=plt.rcParams['axes.labelsize'])
        ax=self.xzpltAaxis
        # ax1.set_ylabel('Z (m)')
        ax[1].set_xlabel('X (m)')
        ax[0].set_ylim(self.r_com[2]-0.1, self.r_com[2]+0.1)  # upper y-axis limits
        ax[1].set_ylim(-0.02, 0.05)

        # hide the spines between ax and ax2
        ax[0].spines.bottom.set_visible(False)
        ax[1].spines.top.set_visible(False)
        ax[0].xaxis.tick_top()
        ax[0].tick_params(labeltop=False)  # don't put tick labels at the top
        ax[1].xaxis.tick_bottom()
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax[0].plot(0, 0, transform=ax[0].transAxes, **kwargs)
        ax[1].plot(0, 1, transform=ax[1].transAxes, **kwargs)

        #Plot hip traj in x-z
        #Hip desired
        ax[0].plot(self.qTraj_des[:, 0], self.qTraj_des[:, 2], 'r--',label='_hidden')
        #Hip Simulation
        ax[0].plot(self.qTraj_act[:, 0], self.qTraj_act[:, 2], 'r-',label='Hip')
        ax[0].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.1), ncol=max(1,len(ax[0].lines)))
        #Left foot desired
        ax[1].plot(self.oLTraj_des[:, 0], self.oLTraj_des[:, 2], 'g--',label='_hidden')
        #Left foot Simulation
        ax[1].plot(self.oLTraj_act[:, 0], self.oLTraj_act[:, 2], 'g-',label='Left Foot')
        #Right foot desired
        ax[1].plot(self.oRTraj_des[:, 0], self.oRTraj_des[:, 2], 'b--',label='_hidden')
        #Right foot Simulation
        ax[1].plot(self.oRTraj_act[:, 0], self.oRTraj_act[:, 2], 'b-',label='Right Foot')
        ax[1].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.1), ncol=max(1,len(ax[1].lines)))


    def plotCAMTraj(self,ttraj,CAMtraj,linestyl='-',overlap=0):
        #Plot AM in subplots with rate of change of AM in twin axis  
        dt=ttraj[1]-ttraj[0] 
        dCAMtraj=np.zeros_like(CAMtraj)
        for i in range(3):
            for j in range(1,len(ttraj)):
                dCAMtraj[j,i]=(CAMtraj[j,i]-CAMtraj[j-1,i])/dt
        if overlap==0:
            fig1, self.CAMpltAaxis = plt.subplots(nrows=3, sharex=True)  # joint angles
            self.CAMpltAaxis[-1].set_xlabel('Time (s)')

        ax1=self.CAMpltAaxis
        ax_lab=['$h_x$ (kg-m²/s)','$h_y$ (kg-m²/s)','$h_z$ (kg-m²/s)']
        col = ['k', 'k', 'k', 'c', 'm', 'k']
        for i in range(len(ax1)):
            ax1[i].plot(ttraj, CAMtraj[:, i] , col[i],linestyle=linestyl)#,                        label='$h_{' + str(i) + '}$')
            ax1[i].set_ylabel(ax_lab[i])
            # ax2 = ax1[i].twinx()
            # ax2.plot(ttraj, dCAMtraj[:, i], 'r', linestyle='-')  # ,                        label='$\dot{h}_{' + str(i) + '}$')
            # # ax2.plot(ttraj, np.cumsum(CAMtraj[:, i], axis=0), 'k', linestyle='--')  # ,                        label='$\dot{h}_{' + str(i) + '}$')
            # ax2.set_ylabel('$\dot{' + ax_lab[i].split('$')[1].split('$')[0] + '}$ (N-m)', color='r')

            # ax1[i].grid(visible=None, which='major', axis='both')
            # ax1[i].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.2), ncol=max(1,len(ax1[i].lines)))



    
    # Numerical inverse kinematics of the robot
    def numik(self,model,data,q0, delt, ocm,oleft,oright,ubjnts,gradH0,WLN,k_ub,zeroAM):
        data=self.q2data(data,q0)
        mujoco.mj_fwdPosition(model, data)
        ocmi=data.subtree_com[0].copy() #current COM position

        olefti=data.site('left_foot_site').xpos.copy() #current Left foot position
        orighti=data.site('right_foot_site').xpos.copy() #current right foot position

        quat_hip=data.qpos[3:7].copy()
        quat_left=np.zeros([4])
        quat_right=np.zeros([4])

        quat_conj=np.zeros([4])
        err_quat=np.zeros([4])
        err_ori_hip = np.zeros([3])
        err_ori_left=np.zeros([3])
        err_ori_right=np.zeros([3])
        # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
        mujoco.mju_negQuat(quat_conj, quat_hip)
        mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
        mujoco.mju_quat2Vel(err_ori_hip, err_quat, 1.0)

        if model.nu-len(ubjnts)>=6:
            mujoco.mju_mat2Quat(quat_left, data.site('left_foot_site').xmat)
            mujoco.mju_negQuat(quat_conj, quat_left)
            mujoco.mju_mulQuat(err_quat, np.array([1,0,0,0]), quat_conj)
            mujoco.mju_quat2Vel(err_ori_left, err_quat, 1.0)

            # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
            mujoco.mju_mat2Quat(quat_right, data.site('right_foot_site').xmat)
            mujoco.mju_negQuat(quat_conj, quat_right)
            mujoco.mju_mulQuat(err_quat, np.array([1,0,0,0]), quat_conj)
            mujoco.mju_quat2Vel(err_ori_right, err_quat, 1.0)

        # Error to minimize for numerical inverse kinematics
        delE=np.linalg.norm(oleft-olefti)+np.linalg.norm(err_ori_left)+np.linalg.norm(oright-orighti)+np.linalg.norm(err_ori_right)+np.linalg.norm(ocm-ocmi)
        k=1 # Increase k to increase accuracy of q
        while delE>1e-8:
            Jcm = np.zeros((3, model.nv)) # COM position jacobian
            mujoco.mj_jacSubtreeCom(model, data, Jcm,0)
            Jwb = np.zeros((3, model.nv))  # Base orientation jacobian
            Jwb[0:3,3:6]=np.eye(3)
            Jvleft = np.zeros((3, model.nv)) # Left foot center jacobian
            Jwleft = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jvleft, Jwleft, model.site('left_foot_site').id)
            #mujoco.mju_mat2Quat(quat_left, data.site(model.site('left_foot_site').id).xmat)
            #mujoco.mju_negQuat(quat_left, quat_left)
            #mujoco.mju_quat2Vel(err_ori_left, quat_left, 1.0)
            Jvright = np.zeros((3, model.nv)) # right foot center jacobian
            Jwright = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jvright, Jwright, model.site('right_foot_site').id)
            #mujoco.mju_mat2Quat(quat_right, data.site(model.site('right_foot_site').id).xmat)
            #mujoco.mju_negQuat(quat_right, quat_right)
            #mujoco.mju_quat2Vel(err_ori_right, quat_right, 1.0)
            # lock upperbody
            if ubjnts.size:
                #ubjnts=np.arange(18,model.nv) #kondo khr3hv
                #ubjnts=np.append([6,7,8],np.arange(21,model.nv)) #MuJoCo humanoid model
                Jub = np.zeros((len(ubjnts), model.nv))  # Base orientation jacobian
                Jub[:,ubjnts]=np.eye(len(ubjnts))
            # Ang momentum
            Iwb=np.zeros([3,model.nv])
            mujoco.mj_angmomMat(model, data, Iwb, 0)

            Avec=np.zeros([18+len(ubjnts)+3+3,model.nv])
            bvec=np.zeros([18+len(ubjnts)+3+3])
            #COM traj
            Avec[0:3,0:model.nv]=Jcm
            bvec[0:3] = ocm-ocmi
            #Hip orient
            Avec[3:6,0:model.nv]=Jwb
            bvec[3:6]=err_ori_hip #np.zeros([3])
            #Left ankle lin vel
            Avec[6:9,0:model.nv]=Jvleft
            bvec[6:9] = oleft-olefti
            #Left ankle ang vel
            Avec[9:12,0:model.nv]=Jwleft
            bvec[9:12] = err_ori_left #np.zeros([3])
            #Right ankle lin vel
            Avec[12:15,0:model.nv]=Jvright
            bvec[12:15] = oright-orighti
            #Right ankle ang vel
            Avec[15:18,0:model.nv]=Jwright
            bvec[15:18] = err_ori_right #np.zeros([3])
            #Upper body joints
            invWroot=np.eye(model.nv) #WLN
            if ubjnts.size:
                #Weighted least norm solution
                qmax=np.pi/4*np.ones([model.nv])
                qmin=-np.pi/4*np.ones([model.nv])
                qmin[21]=-0.1
                qmax[21]=np.pi/4
                qmin[25]=-np.pi/4
                qmax[25]=0.1
                qref=0*0.5*(qmin+qmax)
                gradH=np.zeros([model.nv])
                # W = np.eye(model.nv) + np.diag(gradH)
                for i in np.arange(18,model.nv):
                    gradH[i]=((qmax[i]-qmin[i])**2)*(2*q0[i]-qmax[i]-qmin[i])/(4*(qmax[i]-q0[i])**2 *(q0[i]-qmin[i])**2)
                    if abs(gradH[i])-abs(gradH0[i])<0:
                        invWroot[i,i]=1
                    else:
                        invWroot[i,i]=np.sqrt(1/(1+abs(gradH[i])))
                gradH0=gradH.copy()
                # print('max gradH:', np.max(abs(gradH)))#,gradH)
                # print('W^(-1/2):', invWroot)
                
                #Lock upper body joints
                Avec[18:18+len(ubjnts),0:model.nv]=Jub
                bvec[18:18+len(ubjnts)] = ( qref[ubjnts] - q0[ubjnts] )/1 #np.zeros([len(ubjnts)])#
                #Zero angular momentum using upper body joints
                Avec[18+len(ubjnts):18+len(ubjnts)+3,0:model.nv]=Iwb
                bvec[18+len(ubjnts):18+len(ubjnts)+3] = np.zeros([3])
                # #Sym. motion of upper body joints
                # Avec[18+len(ubjnts)+3:18+len(ubjnts)+6,ubjnts]=Iwb[:,ubjnts]
                # bvec[18+len(ubjnts)+3:18+len(ubjnts)+6] = np.zeros([3])
            else:
                Avec[18:18+3,0:model.nv]=Iwb
                bvec[18:18+3] = np.zeros([3])


            #J=np.append(np.append(np.append(np.append(Jcm,Jwb,axis=0), np.append(Jvleft,Jwleft,axis=0),axis=0), np.append(Jvright,Jwright,axis=0), axis=0),Jub,axis=0)
            #delx=np.append(np.append(np.append( np.append(ocm-ocmi,np.zeros([3]),axis=0), np.append(oleft-olefti,np.zeros([3]),axis=0),axis=0), np.append(oright-orighti,np.zeros([3]),axis=0), axis=0),np.zeros([model.nv-18]),axis=0)
            if (model.nv-len(ubjnts) )<18: #Planer biped
                eqnJ1=np.append(np.array([0,2]),np.array([6,8,10, 12,14,16])) # remove foot orientation
            else: #Spatial biped
                eqnJ1 = np.append(np.array([0, 1, 2]), np.arange(6, 18))

            J1 = Avec[eqnJ1, :].copy()
            delx1 = bvec[eqnJ1].copy()/delt
            dqN1 = np.matmul(np.linalg.pinv(J1), delx1)
            InJ1=np.eye(model.nv)-np.matmul(np.linalg.pinv(J1),J1)
            if zeroAM==True and WLN==False: #Zero ang momentum
                # K_ub=Jub
                # for i in range(len(ubjnts)):
                #     K_ub[i,ubjnts[i]]=0.0001+(np.exp(k_ub*abs(qref[ubjnts[i]] - q0[ubjnts[i]]))-1)/np.exp(k_ub*abs(qref[ubjnts[i]] - q0[ubjnts[i]])) #Fails because it causes large motion and vibrations in joint which is free to move
                # k_L=1-np.linalg.norm(K_ub)
                # k_ub=np.linalg.norm(Iwb)
                k_ref=k_ub**2/4 #10*model.opt.timestep
                k_L =np.exp(-k_ub*np.linalg.norm((qref[ubjnts] - q0[ubjnts])))

                #k_ub = 0.0#0.1
                # print(k_ub)
                # k_ub=0.1*np.linalg.norm(Iwb)
                # k_ub=1*np.linalg.norm((qref[ubjnts] - q0[ubjnts]))
                # K_ub=np.eye(len(ubjnts)) #+ np.diag(abs(gradH[18:]))
                # k_L =(1 / (1 + k_ub*np.linalg.norm((qref[ubjnts] - q0[ubjnts]))))
                # k_L =(1 - k_ub* np.linalg.norm((qref[ubjnts] - q0[ubjnts]))**2)
                # k_L =(1 / (1 + k_ub* np.linalg.norm((qref[ubjnts] - q0[ubjnts]))))/np.linalg.norm(Iwb)
                # k_ub = k_ub * np.linalg.norm(Iwb) 
                # k_L = 1/(np.linalg.norm(Iwb)*np.linalg.norm(K_ub)) #(np.linalg.norm(invWroot)**2) #(1 / (1 + k_ub * np.linalg.norm((qref[ubjnts] - q0[ubjnts]))))
                # k_L=1/(1+k_ub*np.linalg.norm(err_ori_hip))                
                # Avec[3:6,0:model.nv]=Jwb*k_ub*(1-k_L)
                # bvec[3:6]=err_ori_hip*model.opt.timestep #np.zeros([3])
                print('k_L:', k_L, 'k_ub:', k_ub, 'k_ref:', k_ref, len(ubjnts))
                # w_ub=k_ub**2*np.linalg.norm((qref[ubjnts] - q0[ubjnts]))
                # Lock upper body joints
                Avec[18:18 + len(ubjnts), 0:model.nv] = Jub* (1 - k_L) * k_ub
                bvec[18:18 + len(ubjnts)] =   (qref[ubjnts] - q0[ubjnts])*(1-k_L) * k_ref #* model.opt.timestep #*(w_ub+k_ub)/k_ub #* np.linalg.norm(Iwb) #delt #k_ub * (1-k_L) # np.zeros([len(ubjnts)])
                # Zero angular momentum using upper body joints
                Avec[18 + len(ubjnts):18 + len(ubjnts) + 3, 0:model.nv] = 0.01*Iwb * k_L
                bvec[18 + len(ubjnts):18 + len(ubjnts) + 3] = np.zeros([3])

                eqnJ2 = np.append(np.append(np.array([3, 4, 5]), np.arange(18, 18 + len(ubjnts))), np.arange(18 + len(ubjnts), 18 + len(ubjnts) + 3))  # Abt all axes
                # eqnJ2 = np.append(np.array([3, 4, 5]), np.arange(18 + len(ubjnts) +2, 18 + len(ubjnts) + 3)) #Abt Z-axis
                # eqnJ2=np.append(np.array([3,4,5]),np.arange(18+len(ubjnts),18+len(ubjnts)+3)) #Abt all axes
                #eqnJ2=np.arange(18+len(ubjnts),18+len(ubjnts)+2)
                #eqnJ2=np.array([18,19,21,22,23,25,26,27,model.nv,model.nv+1,model.nv+2])
                # eqnJ2 = np.append(np.arange(18, 18 + len(ubjnts)), np.arange(18 + len(ubjnts), 18 + len(ubjnts) + 3))  # Abt all axes
                
            elif zeroAM==True and WLN==True:
                # Lock upper body joints
                Avec[18:18 + len(ubjnts), 0:model.nv] = Jub
                bvec[18:18 + len(ubjnts)] = (np.zeros([len(ubjnts)]) - q0[ubjnts]) * delt *delt # np.zeros([len(ubjnts)])
                # Zero angular momentum using upper body joints
                Avec[18 + len(ubjnts):18 + len(ubjnts) + 3, 0:model.nv] = Iwb
                bvec[18 + len(ubjnts):18 + len(ubjnts) + 3] = np.zeros([3])

                eqnJ2=np.append(np.array([3,4,5]),np.arange(18+len(ubjnts),18+len(ubjnts)+3)) #Abt all axes

            else:
                eqnJ2=np.append(np.array([3,4,5]),np.arange(18,18+len(ubjnts)))

            if WLN==False:
                invWroot=np.eye(model.nv)
            else:
                print(invWroot)
                
            J2 = Avec[eqnJ2, :] @ invWroot #.copy()
            delx2 = bvec[eqnJ2].copy()/delt
            Jt2=np.matmul(J2,InJ1)
            dqN2=dqN1+ invWroot @ np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2,dqN1))
            InJ2=InJ1-np.matmul(np.linalg.pinv(Jt2),Jt2)
            dq=dqN2.copy() #+np.matmul(InJ2,qref-qi)
            # print(asd)


            if delt<1:
                # Integrate joint velocities to obtain joint positions.
                # q0=q0+dq*delt
                data=self.q2data(data,q0)
                q = data.qpos.copy()  # Note the copy here is important.
                mujoco.mj_integratePos(model, q, dq, delt)
                # np.clip(q, *model.jnt_range.T, out=q)

                # q0=data2q(data)
                q0 = 0 * data.qvel.copy()
                qqt = q[3:7].copy()
                qeulr = quat2euler(qqt)
                for i in np.arange(0, 3):
                    q0[i] = q[i].copy()
                for i in np.arange(3, 6):
                    q0[i] = qeulr[i - 3].copy()
                for i in np.arange(6, len(data.qvel)):
                    q0[i] = q[i + 1].copy()

                return q0,gradH0

            while delE<=(np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)) :
                # Integrate joint velocities to obtain joint positions.
                # qi=q0+dq*delt/k
                data=q2data(data,q0)
                q = data.qpos.copy()  # Note the copy here is important.
                mujoco.mj_integratePos(model, q, dq, delt/k)
                # np.clip(q, *model.jnt_range.T, out=q)

                # q0=data2q(data)
                qi = 0 * data.qvel.copy()
                qqt = q[3:7].copy()
                qeulr = quat2euler(qqt)
                for i in np.arange(0, 3):
                    qi[i] = q[i].copy()
                for i in np.arange(3, 6):
                    qi[i] = qeulr[i - 3].copy()
                for i in np.arange(6, len(data.qvel)):
                    qi[i] = q[i + 1].copy()

                data=q2data(data,qi)
                mujoco.mj_fwdPosition(model, data)
                ocmi = data.subtree_com[0]
                olefti = data.site('left_foot_site').xpos.copy()  # current Left foot position
                orighti = data.site('right_foot_site').xpos.copy()  # current right foot position
                # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
                # mujoco.mju_negQuat(quat_conj, quat_hip)
                # mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
                # mujoco.mju_quat2Vel(err_ori_hip, err_quat, 1.0)

                if model.nu - len(ubjnts) >= 60:
                    mujoco.mju_mat2Quat(quat_left, data.site('left_foot_site').xmat)
                    mujoco.mju_negQuat(quat_conj, quat_left)
                    mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
                    mujoco.mju_quat2Vel(err_ori_left, err_quat, 1.0)

                    # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
                    mujoco.mju_mat2Quat(quat_right, data.site('right_foot_site').xmat)
                    mujoco.mju_negQuat(quat_conj, quat_right)
                    mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
                    mujoco.mju_quat2Vel(err_ori_right, err_quat, 1.0)

                # print(np.linalg.norm(np.matmul(J,dq)-delx))
                k=2*k
                if k>2:
                    print('Error (x_des-x_cur) is diverging')


            if delE>(np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)) :
                delE = np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)
                q0 = qi.copy()
                k=1
            """
            else:
                print('Error (x_des-x_cur) is diverging')
                k=2*k
                data = q2data(data, q0)
                mujoco.mj_fwdPosition(model, data)
                ocmi = data.subtree_com[0]
                olefti = data.site('left_foot_site').xpos.copy()  # current Left foot position
                orighti = data.site('right_foot_site').xpos.copy()  # current right foot position
            """

        return q0, gradH0
    
    def DepthvsForce(self,model,data,plotdata):
        # setting font sizeto 30
        # plt.rcParams['text.usetex'] = True
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 24})
        #Plot
        #   fig=plt.figure()#figsize=(8, 6))
        # Find the height at which the vertical force becomes less than the weight, i.e. contact is initiated
        weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
        mujoco.mj_inverse(model, data)
        if data.ncon: #Contact already exists
            dz=0.000001
        else: #No contact exists
            dz=-0.000001
        while True:
            data.qpos[2] += dz
            mujoco.mj_inverse(model, data)
            # print(data.qpos[2],data.qfrc_inverse[2],weight)
            if (dz>0)*(data.ncon==0) or (dz<0)*(data.ncon>0):
                z_0=data.qpos[2]
                break
        #Plot height vs Vertical Force
        height_arr = np.linspace(z_0-0.025, z_0, 101)
        vertical_forces = []
        contact_forces = []
        for z in height_arr:
            data.qpos[2] = z
            mujoco.mj_inverse(model, data)
            #if z%0.0005==0: print(z,data.efc_KBIP, data.efc_diagApprox)
            vertical_forces.append(data.qfrc_inverse[2])

            # contact force
            fc = np.zeros([6])
            for i in np.arange(0, data.ncon):
                #conid = data.contact[i].geom1
                fci = np.zeros([6])
                try:
                    mujoco.mj_contactForce(model, data, i, fci)
                    fc = fc + fci
                except:
                    print('no contact')
            contact_forces.append(fc)
            # print('fc =', fc[0])
            # Reproduce MuJoCo Forces
            # fmj,fsd,defmj,margmj =mjforce(model,data)
            # print(data.qfrc_inverse[2],fmj,data.ncon)
            # if data.ncon>0:
            #     plt.plot(abs(z-z_0)*1, fc[0], 'g.', markersize=2)
            #     # plt.plot(abs(z-z_0)*1, fmj[-1], 'g.', markersize=2)
            #print(asd)

        height_offsets=height_arr-z_0
        vertical_forces=np.array(vertical_forces)
        contact_forces=np.array(contact_forces)[:,0]

        # Find the height-offset at which the vertical force is smallest.
        idx = np.argmin(np.abs(vertical_forces))
        best_offset = height_offsets[idx]
        # Plot the relationship.
        if plotdata==1:
            print('weight=', weight)
            fig, ax = plt.subplots()
            fig.subplots_adjust(right=0.75)

            #   twinax = ax.twinx()

            p1, =ax.plot(abs(height_offsets) * 1, contact_forces, 'r-', linewidth=3, label='force')
            #plt.plot(abs(height_offsets) * 1, vertical_forces, 'r-', linewidth=3)
            # Red vertical line at offset corresponding to smallest vertical force.
            ax.axvline(x=abs(best_offset) * 1, color='black', linestyle='-', linewidth=1)
            # Green horizontal line at the humanoid's weight.
            weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
            ax.axhline(y=weight, color='black', linestyle='-', linewidth=1)
            ax.set(xlabel='Deformation (m)')
            ax.set(ylabel='Contact force (N)')
            ax.set_xlim([0, max(abs(height_offsets) * 1)])
            #   ax.yaxis.label.set_color(p1.get_color())
            # ax.grid(visible=None, which='major', axis='both')
            #   plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = 2)
            #   plt.ylabel('Vertical force on base (N)')
            #   plt.twinx()
            stiff_y=np.gradient(vertical_forces,height_offsets*1)
            #   p2, =twinax.plot(abs(height_offsets) * 1, stiff_y, 'b--', linewidth=2, label='Normal stiffness')
            #   twinax.set(ylabel='Stiffness (N/m')
            #   twinax.yaxis.label.set_color(p2.get_color())
            #   twinax.grid(visible=None, which='major', axis='both')
            #   plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = 2)
            #   plt.legend(handles=[p1, p2],loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = 2)
            # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
            #   plt.minorticks_on()
            # plt.title(f'Min. vertical force at deformation {str(best_offset * 1000)[1:5]} mm.')
            #   plt.show(block=False)
            plt.pause(1)
            #   plt.show()

        return best_offset
    
    def modifyTerrain(self,model,data,nocp,zeta1):
        # Modify parameters of terrain
        # model.geom_solimp[0][0]=0.0
        # model.geom_solimp[0][2]=0.01
        #model.geom_solimp[1]=[0.0, 0.95, 0.0002, 0.5, 2]
        for nocp1 in np.arange(nocp,0,-1/10): #[nocp]:#[0.3]:#for defT, [1.8]:for hardT #
            trn1=trnparam(nocp1,zeta1,self.zpln) #hard terrain parameters for left foot terrain
            trn1.mjparam(model)

            #Change terrain solref of Humanoid xml model
            i=0
            while model.geom_bodyid[i] == 0:
                model.geom_solimp[i] = trn1.solimp[i]
                model.geom_solref[i]=trn1.solref[i]
                i=i+1
            # print(DepthvsForce(model,data,0))
            if abs(self.DepthvsForce(model,data,0))<(model.geom_solimp[0][2]/2):
                print('nocp=',nocp1)
                print('stiffness=', model.geom_solref[0][0], 'damping=', model.geom_solref[0][1], ', ...wait for 2 sec')
                print('Des vs Act Deformation is:',(model.geom_solimp[0][2]/2),self.DepthvsForce(model,data,0))
                break
        return model
    
    def mjfc(self,model,data):
        # Normal Contact Force
        self.fcl = np.zeros([3])
        self.fcl[0]=data.xfrc_applied[model.site_bodyid[data.site("left_foot_site").id]][2]
        self.fcr = np.zeros([3])
        self.fcr[0] = data.xfrc_applied[model.site_bodyid[data.site("right_foot_site").id]][2]
        self.fcn = self.fcl[0] + self.fcr[0]
        self.rfl = np.zeros([3])
        self.rfr = np.zeros([3])
        self.rf = np.zeros([3])
        for i in np.arange(0, data.ncon):
            # conid = min(data.contact[i].geom1,data.contact[i].geom2)
            fci = np.zeros([6])
            try:
                mj.mj_contactForce(model, data, i, fci)
                # print(fci[0])
                self.fcn = self.fcn + abs(fci[0])
                self.rf = self.rf + np.array(data.contact[i].pos) * abs(fci[0])  # pos*normal force
                #fc0 = np.matmul(np.array(data.contact[i].frame).reshape((3, 3)), -fci[0:3])  # force vector in world frame

                if model.geom_bodyid[data.contact[i].geom2] == model.site_bodyid[data.site("left_foot_site").id] or model.geom_bodyid[data.contact[i].geom1] == model.site_bodyid[data.site("left_foot_site").id]:  # Left foot body
                    self.fcl = self.fcl + fci[0:3]
                    self.rfl = self.rfl + np.array(data.contact[i].pos) * abs(fci[0])
                elif model.geom_bodyid[data.contact[i].geom2] == model.site_bodyid[data.site("right_foot_site").id] or model.geom_bodyid[data.contact[i].geom1] == model.site_bodyid[data.site("right_foot_site").id]:  # Right foot body
                    self.fcr = self.fcr + fci[0:3]
            except:
                print('no contact')

        #COP position
        if abs(self.fcn) > 0:
            self.r_cop = self.rf / abs(self.fcn)
        else:  # No contact
            self.r_cop = np.array([np.nan, np.nan, np.nan])













def COT(m,WD,dist):
    COT = WD / (m*9.81*dist)
    return COT



# inv dynamics in MuJoCo
def tauinvd(model,data,ddqdes):
    data.qacc=ddqdes.copy()
    #mujoco.mj_fwdActuation(model,data)
    mujoco.mj_inverse(model,data)
    #mujoco.mj_fwdActuation(model,data)
    tau=data.qfrc_inverse[6:].copy()
    return tau

# Find terrain parameters (solref) given solimp=[d0,dwidth,width,midpt,power] and zeta
class trnparam:
    def __init__(self,nocp,zeta,zpln):
        i=0
        self.zeta=zeta
        self.zpln=zpln
        self.nocp=nocp
        # self.A=[]
        self.r=[]
        self.rdot=[]
        self.aref=[]
        self.f=[]
        self.efc_f=[]
        self.q=[]
        self.dq=[]
        self.ddq=[]

    def mjparam(self, model):
        # print(model.body_mass,model.body_invweight0)
        self.m=mujoco.mj_getTotalmass(model)
        self.solimp=[]
        self.solref=[]
        self.pos=[]
        self.size=[]
        self.xmean=[]
        i=0
        while model.geom_bodyid[i]==0:
            # self.m=model.body_mass[model.site_bodyid[model.site("left_foot_site").id]] #Of robot contact body not plane
            # self.Ainv=model.body_invweight0[model.site_bodyid[model.site("left_foot_site").id]][0] #Of robot contact body not plane
            # print('m=', self.m, 'Ainv=', self.Ainv)
            self.pos.append(model.geom_pos[i].copy())
            self.size.append(model.geom_size[i].copy())
            solimp=model.geom_solimp[i].copy()
            solref=model.geom_solref[i].copy()
            self.solimp.append(solimp)
            d0=solimp[0]
            dwidth=solimp[1]
            width=solimp[2]
            midpt=solimp[3]
            power=solimp[4]
            #trn.solref = (-Stiffness, -damping)
            #self.solimp = [d0, dwidth, width, midpt, power]
            dmean=(d0+dwidth)/2
            deln=width*self.nocp
            xmean = deln / 2
            if solref[0]<0:
                # kn=9.81/deln
                dampratio=self.zeta
                stiffness=(9.81*(1-dmean)*dwidth*dwidth)/(xmean*dmean*dmean) #/self.zeta**2
                timeconst=1/(dampratio*np.sqrt(stiffness))
                #dampratio = 1 / (timeconst * np.sqrt(stiffness))
                # k=stiffness*d(r)/dwidth
                #xmax=(1-dwidth)*9.81/stiffness
                # wn=np.sqrt(stiffness)
                #timeconst = 1 / (zeta * wn) #0.02 default
                # damping=2*wn*self.zeta
            else:
                timeconst=solref[0] #np.sqrt(1/(9.81*(1-dwidth)/width)) # 1*solref[0]
                dampratio = solref[1]
                # kn=9.81/deln
                stiffness=1/((timeconst**2)*(dampratio**2))#9.81*(1-dwidth)/deln
                # k=stiffness*d(r)/dwidth
                #xmax=(1-dwidth)*9.81/stiffness
                #wn=np.sqrt(nocp*stiffness)
                #timeconst = 1 / (zeta * wn) #0.02 default
                #damping=2*zeta*wn/nocp #
            damping=2/timeconst #2/(dwidth*timeconst)
            i=i+1

            #stiffness=stiffness/nocp
            #damping=damping/nocp

            self.solref.append([-stiffness,-damping])
            self.xmean.append(xmean)

    def cntplane(self,cntpt,spno):
        i=0
        for pos in self.pos:
            size=self.size[i].copy()
            if cntpt[0]>(pos[0]-size[0]) and cntpt[0]<(pos[0]+size[0]):
                if cntpt[1] > (pos[1] - size[1]) and cntpt[1] < (pos[1] + size[1]):
                    self.cntgeomid=i
                    self.cntpos=self.pos[i].copy()
                    self.cntsize=self.size[i].copy()
                    if spno==1:
                        self.cntsolref = self.solref[i].copy()
                        self.cntsolimp = self.solimp[i].copy()
                        self.cntnocp = self.nocp
                        self.cntpos[2] +=  self.cntsize[2]
                    else:
                        self.cntsolref = [0.02, 1] #self.solref[i].copy()
                        self.cntsolimp = [0.9,0.95,0.001,0.5,2] #self.solimp[i].copy()
                        self.cntnocp = 2*self.nocp
                        self.cntpos[2] = cntpt[2] + 0.00036 #self.cntsolimp[2]/2.5
                        # self.cntnocp=2*self.cntnocp
                    # else:
                    #     self.qcp[2]=self.cntpos[2] - self.cntsolimp[2]
                    break
            i=i+1

    def paramidentify(self):
        def fun(x):
            d0 = x[0]
            dwidth = x[1]
            width = x[2]
            midpt = x[3]
            p = x[4]
            zeta=x[5]
            dampratio=zeta
            dmean=(d0+dwidth)/2
            deln = width * 1/self.cntnocp
            xmean = deln / 2
            stiffness = (9.81*(1-dmean)*dwidth*dwidth)/(xmean*dmean*dmean) #9.81 * (1 - d_width) / width
            #wn = np.sqrt(stiffness)
            timeconst = 1/(dampratio*np.sqrt(stiffness)) #1 / (zeta * wn) #0.02 default
            damping = 2/timeconst #2 * zeta * wn  #
            # kn=x[5]
            # bn=x[6]
            # if delr > 0:
            #     delr = 0
            #     rdot = 0
            fval=0
            i=0
            for delr in self.r:
                rdot=self.rdot[i]
                x = abs(delr) / width
                if x <= midpt:
                    a = (1 / midpt) ** (p - 1)
                    y = a * (x ** p)
                else:
                    b = 1 / (1 - midpt) ** (p - 1)
                    y = 1 - b * max(0, 1 - x) ** p
                y = min(1, y)
                # print('d(r) =',1/(1+data.efc_R[0]/data.efc_diagApprox[0]))
                d = d0 + y * (dwidth - d0)
                # d=max(d,d_0)
                # d=min(d,d_width)
                # k = -model.geom_solref[0][0] * d / (d_width ** 2)
                # b = -model.geom_solref[0][1] / d_width
                # print(k/d,b)
                # zeta = (b * d_width) / (2 * np.sqrt(k * (d_width ** 2) / d))  # b/(2*np.sqrt(k)) #
                k= d*stiffness/(dwidth**2)
                b= damping/dwidth
                aref = (-b * rdot - k * delr)
                # A=self.A[i]
                # R=(1-d)/d*A
                # f=1/(A+R)*(aref-self.a0[i])
                fval +=abs(self.aref[i] - aref)
                i=i+1
            return fval
        x0=np.append([0.9,0.95,0.001,0.5,2],1) #np.append(0.5*self.cntsolimp,-self.cntsolref[1]/(2*np.sqrt(-self.cntsolref[0])))
        # print('f(x0)=',fun(x0))
        bnds = ((0, 0.99), (0.1, 0.99), (0.0001, 0.02), (0.0, 0.99), (0, 5), (1e-5,1000))
        # Gradient-based method
        # x=minimize(fun,x0, bounds=bnds).x
        # Search method
        # Solve minimization using differential evolution
        x = differential_evolution(fun, bounds=bnds)['x']

        print('efc_aref-aref=',fun(x))
        dmean=(x[0]+x[1])/2
        deln = x[2] * 1/self.cntnocp
        xmean=deln/2
        stiffness = (9.81*(1-dmean)*x[1]*x[1])/(xmean*dmean*dmean)#9.81 * (1 - x[1]) / x[2]
        #wn = np.sqrt(stiffness)
        timeconst =1/(x[5]*np.sqrt(stiffness)) #1 / (zeta * wn) #0.02 default
        damping = 2/timeconst #2 * x[5] * wn  #
        self.cntsolimp=x[0:5].copy()
        self.cntsolref=np.append(-stiffness,-damping) #x[5:7].copy()


# Reproduce normal contact force of MuJoCo
def mjforce(model,data):
    # Reproduce Mujoco model
    m=mujoco.mj_getTotalmass(model)
    # fmj=np.zeros([model.nbody])
    fmj=np.zeros([data.ncon])
    fsd=np.zeros([model.nbody])
    a0 = np.zeros([data.ncon])
    aref=np.zeros([data.ncon])
    alcp=np.zeros([data.ncon])
    A=np.zeros([data.ncon])
    R=np.zeros([data.ncon])
    D=np.zeros([data.ncon])
    def_mj=np.zeros([data.ncon])
    marg_mj=np.zeros([data.ncon])
    #efc_force=np.zeros([data.nefc])
    a0vec = np.zeros([data.nefc])
    ddq0 = data.qacc_smooth.copy()  ##Unconstrained acceleration in joint space
    mujoco.mj_mulJacVec(model, data, a0vec, ddq0)  # Unconstrained acceleration in contact space #1/m*(-m*9.81) #J*data.qacc_smooth --Unconstrained acceleration

    for i in np.arange(0,data.ncon):
        efcid=data.contact[i].efc_address
        geomid=min(data.contact[i].geom) #Terrain id
        bodyid=model.geom_bodyid[max(data.contact[i].geom)] #Parent body of contact geom2
        delr=data.efc_pos[efcid]-data.efc_margin[efcid] #deformation
        rdot=data.efc_vel[efcid] #deformation rate
        # print('def,rate of def',data.efc_pos[efcid],rdot)

        d_0=model.geom_solimp[geomid][0]
        d_width = model.geom_solimp[geomid][1]
        width=model.geom_solimp[geomid][2]
        midpt = model.geom_solimp[geomid][3]
        p = model.geom_solimp[geomid][4]
        if delr>0:
            delr=0
            rdot=0
        # stifness and damping
        if model.geom_solref[geomid][0]<0:
            stiffness=-model.geom_solref[geomid][0] #*data.ncon
            damping=-model.geom_solref[geomid][1] #*data.ncon
            # k=-model.geom_solref[0][0]*d/(d_width**2)
            # b=-model.geom_solref[0][1]/d_width
        else:
            stiffness=1/((model.geom_solref[geomid][0]**2)*(model.geom_solref[geomid][1]**2))
            damping=2/model.geom_solref[geomid][0]
            # k=d/((d_width**2)*(model.geom_solref[0][0]**2)*(model.geom_solref[0][1]**2))
            # b=2/(d_width*model.geom_solref[0][0])

        x=abs(delr)/width
        y=0
        if x>=1:
            y=1
        elif x<=midpt:
            a=(1/midpt)**(p-1)
            y=a*(x**p)
        elif x>midpt and x<1:
            b=1/(1-midpt)**(p-1)
            y=1-b*(1-x)**p
        #y=min(1,y)
        #print('d(r) =',1/(1+data.efc_R[0]/data.efc_diagApprox[0]))
        d=d_0+y*(d_width-d_0)

        k=stiffness*d/(d_width**2)
        b=damping/d_width
        #print(data.efc_KBIP)
        zeta=(b*d_width)/(2*np.sqrt(k*(d_width**2)/d)) #b/(2*np.sqrt(k)) #

        aref[i] +=(-b*rdot-k*delr) #data.efc_aref[efcid] -- Reference acceleration
        a0[i]=-9.81 #a0vec[efcid]
        A[i]=data.efc_diagApprox[efcid] #1/(m/data.ncon)#data.efc_diagApprox[efcid] # #data.efc_diagApprox[efcid]
        R[i]=(1-d)/d*A[i]
        D[i] =1/(R[i])
        #print(D,data.efc_D)
        def_mj[i]=data.efc_pos[efcid]
        marg_mj[i]=data.efc_margin[efcid]
        # print('my_KB: ',k/d,b,' data.efc_KB:',data.efc_KBIP[efcid][0],data.efc_KBIP[efcid][1])
        # print('myaref',aref[i],'data.efc_aref',data.efc_aref[efcid])
        # print('myD:',D[i],'data.efc_D:',data.efc_D[efcid])
        # print('myA:',A[i],'data.efc_A:',data.efc_diagApprox[efcid])
        # print('myR:',R[i],'data.efc_R:',data.efc_R[efcid])

        # if abs(delr)<midpt*width:
        #     print(delr,width)
        #     C1=d_0**2*stiffness/(d_width**2)
        #     C2=a0[i]*(d_width-d_0)/(midpt**(p-1)*width**p)
        #     C3=(2*d_0*(d_width-d_0)*stiffness)/(d_width**2*midpt**(p-1)*width**p)
        #     C4=((d_width-d_0)**2*stiffness)/(d_width**2*(midpt)**(2*p-2)*width**(2*p))
        #     C5=d_0*damping/(d_width)
        #     C6=(damping*(d_width-d_0))/(d_width*midpt**(p-1)*width**p)  
        #     # Contact force
        #     fmj[i]=1/A[i]*(-C1*delr-C2*delr**(p)-C3*delr**(1+p)-C4*delr**(1+2*p)-C5*rdot-C6*delr**p*rdot-d_0*a0[i])
        # elif abs(delr)>=midpt*width and abs(delr)<width:
        #     print(delr,width)
        #     C1=stiffness
        #     C2=a0[i]*(d_width-d_0)/((1-midpt)**(p-1)*width**p)
        #     C3=(2*(d_width-d_0)*stiffness)/(d_width*(1-midpt)**(p-1)*width**p)
        #     C4=((d_width-d_0)**2*stiffness)/(d_width**2*(1-midpt)**(2*p-2)*width**(2*p))
        #     C5=damping
        #     C6=(damping*(d_width-d_0))/(d_width*(1-midpt)**(p-1)*width**p)  
        #     # Contact force
        #     drwidth=width+delr
        #     fmj[i]=1/A[i]*(-C1*delr+C2*(drwidth)**p+C3*(drwidth)**(p)*delr-C4*(drwidth)**(2*p)*delr-C5*rdot+C6*(drwidth)**(p)*rdot-d_width*a0[i])
        # else:
        #     fmj[i]=1/A[i]*(-stiffness*delr-damping*rdot-d_width*a0[i]) #data.efc_force[efcid] -- Contact force


        # Spring force
        kn=1235#-model.geom_solref[0][0]/(d_width**2)#m*9.81/width/data.ncon
        cn=9.9#b#2*zeta*np.sqrt(kn*m)
        fsd[bodyid]=-kn*delr-cn*rdot


    #a0=a0vec[efcid] #-9.81
    #jar=(a0-aref) #data.efc_b[efcid] start deviation from ref accln
    #print((1 - d) * (-9.81) + d * aref)
    # Convex optimization Newton method
    # Minimize KE of contact to obtain yslack
    #Min(x+9.81).'M*(x+9.81) + s(J*x-aref)
    # def fct(ddqi): #Minimize-- constraint + Gauss
    #     Ma=np.zeros([model.nv])
    #     mujoco.mj_mulM(model,data,Ma,ddqi)
    #     #grad = Ma - data.qfrc_smooth
    #     #print(data.qacc_smooth)
    #     #print(Ma-data.qfrc_smooth-(ddqi-data.qacc_smooth))
    #     cost=1/2*np.matmul(np.transpose(Ma-data.qfrc_smooth),(ddqi-data.qacc_smooth))
    #     mujoco.mj_mulJacVec(model,data,a0vec,ddqi)
    #     s=0
    #     """
    #     # From Only Normal contacts
    #     a0 = np.zeros([data.ncon])
    #     for i in np.arange(0,data.ncon):
    #         bodyid = model.geom_bodyid[data.contact[i].geom2]
    #         efcid = data.contact[i].efc_address
    #         a0[i] =a0vec[efcid].copy()
    #         jar=(a0[i]-aref[i])
    #         if jar<0:
    #             s +=1/2*D[i]*jar*jar # For all constraints
    #     """
    #     # From MuJoCo's constraints
    #     a0 = np.zeros([data.nefc])
    #     for i in np.arange(0,data.ncon):
    #         efcid = data.contact[i].efc_address
    #         a0[i] =ddqi[2]#a0vec[efcid].copy()
    #         jar=(a0[i]-aref[i])
    #         if jar<0:
    #             s +=1/2*D[i]*jar*jar # For all constraints
    #
    #
    #     cost +=s
    #     return cost
    # ddqsol= minimize(fct, 0*ddq0)['x'] #data.qacc_smooth
    # # Error in reproduced accln
    # print('Sol for ddq is',ddqsol)
    # print('Error in ddqsol is',ddqsol-data.qacc)
    # mujoco.mj_mulJacVec(model, data, a0vec, ddqsol)
    """def sa(x):
        jar=(x-aref)
        return jar*m*jar
    vdot= fct(alcp) + sa(alcp)
    dsa=(sa(vdot)-sa(alcp))/(vdot-alcp)
    """
    #alcp=-9.81#data.qacc_smooth[2].copy()
    #print(yslack-aref)

    #jar=alcp-aref # efc_b= J*q_accsmooth-aref, # min deviation from ref accln
    #a1=a0+A*f
    for i in np.arange(0,data.ncon):
        #bodyid = model.geom_bodyid[data.contact[i].geom2]
        efcid = data.contact[i].efc_address
        # mujoco.mj_mulJacVec(model, data, a0vec, ddqsol)
        # alcp[i]=ddqsol[2] #a0vec[efcid] #data.qacc_warmstart[2] #-9.81
        # print('a0 =',alcp[i], (m * (-9.81) +  D[i]*aref[i])/(m +  D[i]))
        #jar=alcp[i]-aref[i]
        # if jar<0:
        # print(a0,data.efc_diagApprox)
        # print('r:',delr,'width:',width,'a0:',a0[i])
        fmj[i] = 1/(A[i]+R[i])*(aref[i]-a0[i]) #*data.efc_b[efcid] #-D[i]*(jar)  #fmj=-1*D[i]*(jar)=-m*d*(jar<0)*(jar)
        # print('fmjforce =',fmj[i],-m*D[i]/(m+D[i])*(a0vec[efcid]-aref[i])) #for single contact pt
        #fmjsd=-1*D*(-aref) #fmj=-m*d*(jar)
        #fmjlcp=-1*D*(alcp) #fmj=-m*d*(jar)
        # print(fmj)

    return fmj,fsd,def_mj,marg_mj


# Contact parameters identification given the position or force profile
def sysident(model,ttraj,rspl,rdottraj,fspl,nocp,simfreq):
    # MuJoCo data structures
    #model = mujoco.MjModel.from_xml_path(xml_path)  # MuJoCo model

    # function for error
    def fn_error(solimpzeta):
        data = mujoco.MjData(model)  # MuJoCo data
        solimp=solimpzeta[0:5]
        zeta=solimpzeta[5]
        zpln=0
        #nocp = 1  # No of contact points
        trn=trnparam( nocp, zeta, zpln)  # hard terrain parameters for left foot terrain
        trn.mjparam(model)
        # Change terrain solref of Kondo xml model
        model.geom_solref[0] = trn.solref[0]
        model.geom_solimp[0] = trn.solimp[0]
        #print('solref is',model.geom_solref[0])
        #print('solimp is',model.geom_solimp[0])
        ti = 0
        fn_err=0
        while ti < ttraj[-1]:
            while (data.time-ti)<(1/simfreq):
                mujoco.mj_step(model,data)
            ti=data.time

            # contact force
            fc = np.zeros([6])
            for i in np.arange(0, data.ncon):
                conid = data.contact[i].geom1
                fci = np.zeros([6])
                try:
                    mujoco.mj_contactForce(model, data, conid, fci)
                    fc = fc + fci
                except:
                    print('no contact')
            rcdes=np.zeros([3])
            for i in np.arange(0,3):
                rcdes[i]=rspl[i](ti)
            fcdes=np.zeros([6])
            for i in np.arange(0,3):
                fcdes[i]=fspl[i](ti)
            fn_err +=np.linalg.norm(data.qpos[0:3]-rcdes[0:3]) #np.linalg.norm(fc[0:3]-fcdes[0:3])  #
            """plt.figure(1)
            plt.plot(data.time, data.qpos[2], '.g')
            plt.figure(2)
            plt.plot(data.time, fc[0], '.g')
            """
        print('Error in sol =',fn_err)
        return fn_err

    # Actual solution
    init_sol=np.ones([6])
    init_sol[0:5]=1*model.geom_solimp[0]
    init_sol[5]=-model.geom_solref[0][1]/(2*np.sqrt(nocp*-model.geom_solref[0][0]))
    print('Act sol for ballsim is ',init_sol)
    fn_error(init_sol)
    # Sol. obtained from minimization of force error
    par_sol=np.array([9.20408857e-05, 9.00000000e-01, 8.00000000e-04, 5.00000000e-01, 2.00000000e+00, 0.32126470265858953])
    print('Sol 1 for ballsim is ',par_sol)
    fn_error(par_sol)
    # Sol. obtained from minimization of position error
    par_sol=np.array([4.00580157e-06, 6.98917856e-01, 7.38566972e-04, 4.25074547e-01, 1.98996017e+00, 10.0])
    print('Sol 2 for ballsim is ',par_sol)
    fn_error(par_sol)

    # Another Sol. obtained from minimization of force error with closer bounds
    par_sol = np.array([0.369757209, 0.835229815, 0.000776542288, 0.941126337, 1.72142685e+00, 0.3212647026585895])
    print(par_sol)
    #fn_error(par_sol)
    # Sol. obtained from minimization of force error for box
    par_sol = np.array([4.00580157e-06, 9.00000000e-01, 2.00000000e-04, 5.00000000e-01, 2.00000000e+00,  0.07139215614635253])
    print('Sol 1 for boxsim is ', par_sol)
    fn_error(par_sol)
    # Solve minimization
    bnds = ((0, 0.95), (0.5, 1), (0.0001, 0.001), (0.0, 1), (1, 2), (0.01,10))
    #par_sol=minimize(fn_error,par_sol, bounds=bnds).x
    # Solve minimization using differential evolution
    #par_sol= differential_evolution(fn_error, bounds=bnds)['x'] #data.qacc_smooth
    #print(par_sol)
    #fn_error(par_sol)

    return par_sol[0:5],par_sol[-1]

# Add robot xml to scene xml
def addrobot2scene(xml_tree,robotpath):
    #xml_path = r"C:\Users\SG\OneDrive - IIT Kanpur\Documents\MATLAB Drive\Python\mujoco\kondo\scene_defT.xml" #xml file (assumes this is in the same folder as this file)

    # get the full path
    #dirname = os.getcwd() #os.path.dirname(__file__)
    #abspath = os.path.join(dirname + "/" + xml_path)
    #xml_path = abspath

    # xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    # Change mass,pos, orientation and length of pendulum
    bodyeul=np.zeros([1,3])
    #model.geom_size[2,1]=l/2 # length of cylindrical rod
    for tag1 in root.findall("include"):
        tag1.attrib['file']=robotpath #' '.join(map(str, np.array([0,0,zpln]))) #change contact plane pos

    # xmltree.write('robotwithscene.xml')
    xml_str = ET.tostring(root)
    # ET.dump(root)
    # xml_path = 'robotwithscene.xml'
    return xml_str

def scenegen(xml_tree,trnnum,trnlength,trnwidth,trnheight):
    #xml_path = r"C:\Users\SG\OneDrive - IIT Kanpur\Documents\MATLAB Drive\Python\mujoco\kondo\scene_defT.xml" #xml file (assumes this is in the same folder as this file)

    # get the full path
    #dirname = os.getcwd() #os.path.dirname(__file__)
    #abspath = os.path.join(dirname + "/" + xml_path)
    #xml_path = abspath

    # xmltree = ET.parse(xml_path)
    root = xml_tree.getroot()

    if trnnum==1: #Add one plane geom
        for tag_wb in root.findall("worldbody"):
            geom = ET.SubElement(tag_wb, "geom")
            geom.set("name", "terrain1")
            geom.set("type", "plane")
            geom.set("size", f"{trnlength/2} {trnwidth} {0.001+trnheight}")
            geom.set("pos", f"0 0 0.0+{trnheight}")
            geom.set("rgba", "0.2 0.9 0.2 1")
    else: #Add boxes as terrain
        for tag_wb in root.findall("worldbody"):
            for i in range(trnnum):
                if i==0:
                    geom = ET.SubElement(tag_wb, "geom")
                    geom.set("name", f"terrain{i+1}")
                    geom.set("type", "box")
                    geom.set("size", f"{0.5+trnlength/4} {trnwidth} {0.05+trnheight[i]/2}")
                    geom.set("pos", f"{-1+0.5+trnlength/4} 0 {-0.05+trnheight[i]/2}")
                    geom.set("rgba", f"{0.25+0.5*i/trnnum*(trnheight[i]>0)} {0.25+0.5*i/trnnum*(trnheight[i]>0)} {1.0-0.5*i/trnnum*(trnheight[i]>0)} 1")
                else:
                    geom = ET.SubElement(tag_wb, "geom")
                    geom.set("name", f"terrain{i+1}")
                    geom.set("type", "box")
                    geom.set("size", f"{trnlength/2} {trnwidth} {0.05+trnheight[i]/2}")
                    geom.set("pos", f"{i*trnlength} 0 {-0.05+trnheight[i]/2}")
                    geom.set("rgba", f"{0.25+0.5*i/trnnum*(trnheight[i]>0)} {0.25+0.5*i/trnnum*(trnheight[i]>0)} {1.0-0.5*i/trnnum*(trnheight[i]>0)} 1")



                
    # xmltree.write('robotwithscene.xml')
    xml_str = ET.tostring(root)
    # ET.dump(root)
    # xml_path = 'robotwithscene.xml'
    return xml_str

class mydataparam:
    def __init__(self, d0, dwidth, width, midpt, power,nocp,zeta,zpln):
        self.solimp = [d0, dwidth, width, midpt, power]
        deln=width*nocp
        kn=9.81/deln
        stiffness=9.81*(1-dwidth)/deln
        wn=np.sqrt(nocp*stiffness)
        #timeconst = 1 / (zeta * wn) #0.02 default
        damping=2*zeta*wn/nocp #
        #damping=2/(dwidth*0.02)#timeconst
        self.solref=[-stiffness,-damping]
        self.zpln=zpln


### OpenAI codes for Euler-quat conversion
# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0
def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler

def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))




# DAIR LAB's cube
# import os.path as op

# ROOT_DIR = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
# def root_path(*path: str) -> str:
#     return op.join(ROOT_DIR, *path)

# DATA_DIR = root_path(r"C:\Users\SG\OneDrive - IIT Kanpur\Documents\MATLAB Drive\Python\MuJoCo\example\ContactNets Cube")
# OUT_DIR = root_path('ContactNets Cube\out')
# LIB_DIR = root_path('ContactNets Cube\lib')
# RESULTS_DIR = root_path('ContactNets Cube\results')

# def out_path(*path: str) -> str:
#     return op.join(OUT_DIR, *path)

# def data_path(*path: str) -> str:
#     return op.join(DATA_DIR, *path)

# def results_path(*path: str) -> str:
#     return op.join(RESULTS_DIR, *path)


# def lib_path(*path: str) -> str:
#     return op.join(LIB_DIR, *path)

# PROCESSING_DIR = root_path('contactnets', 'utils', 'processing')
# def processing_path(*path: str):
#     return op.join(PROCESSING_DIR, *path)

# import distutils.dir_util
# """
# # Copy the tosses data and processing parameters into the working directory
# distutils.dir_util.copy_tree(data_path('DAIRLab contact-nets main data-tosses_processed'),
#                              out_path('data', 'all'))
# distutils.dir_util.copy_tree(data_path('params_processed'),
#                              out_path('params'))
# """
