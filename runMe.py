import mujoco as mj
import numpy as np
import os,time,ctypes
# from myfun import selectRobot,q2data,data2q,trnparam,mydataplots,DepthvsForce
# from lib_ZMPctrl import selectRobot,q2data,data2q,trnparam,mydataplots,DepthvsForce
from myRobotEnv import selectRobot,trnparam
from myRobotControl import myController
from myRobotPlots import myDataplots
import pickle
from scipy.interpolate import CubicSpline
import scipy.io
import matplotlib.pyplot as plt

simend = 5 # simulation time
simfreq = 100 # 100 fps

# vel=0.1
# spno=1 #1-SSP, 2-DSP
# step_len=0.5 # step length for MPC


#Select Robot - # 1-Kondo, 2-G1, 3-Biped
# humn,model,data = selectRobot(num=1,vel=0.1,step_len=0.05,spno=1) #1-Kondo
humn,model,data = selectRobot(num=2,vel=0.1,step_len=0.1,spno=1) #2-G1
# humn,model,data = selectRobot(num=3,vel=0.2,step_len=0.4,spno=1) #3-Biped

# model.opt.timestep=0.00001
humn.simend=simend
humn.simfreq=simfreq

#Control
humn.KINctrl=True #False #True #True #False #False #True #False #True #True #Kin traj correction
humn.AMctrl=1.0 #Angular momentum control
humn.ZMPctrl=0 #-1/(10**3)#.0001#01 #ZMP control

# Change the controller
# mj.set_mjcb_control(controller)
# humn.COMctrl=1#.01#.5
humn.k_ub=0.1 #0.05 #1 #0.1000 #/model.opt.timestep #task priority weight for ub
# humn.k_L=humn.AMctrl*1/(1+humn.k_ub*0*np.linalg.norm(data.qpos))
humn.FWctrl=0
humn.ftctrl=0

# Terrain parameters
# Terrain parameters
zeta1=1.5 # damping ratio=15 for SIP model on deformable terrain 
nocp=4 #number of contact points of foot
humn.trn=trnparam(nocp,1*zeta1,humn.zpln) #terrain parameters for equivalent foot terrain # high damping to prevent slip in y-direction on deformable terrain
humn.trn.mjparam(model)

print('stiffness and damping =',humn.trn.solref)

# print(asdf)
#Modify terrain parameters in model
model = humn.modifyTerrain(model,data,nocp,zeta1)
humn.mj2humn(model,data)

st_time=time.time()

# humn.SIPwalk=False
# humn.MPC_LIPM=False
#Modify class for offline planning and control
humn=myController(humn)

# Generate Gait / Cartesian traj to joint space traj
# humn.genWalkingGait(model,data,1*humn.step_time,humn.step_len,0)
ttraj=np.arange(0,simend,1/1000)
#No arm swing
# humn.k_ub=0.05#05#1
qtraj=humn.cart2joint(model,data,ttraj,0,1)
# humn.plotHipTraj(ttraj,qtraj,linestyl='-',overlap=0)
humn.plotqTraj(ttraj,qtraj,linestyl='-',overlap=0)
#save time vs norm of ub_jnts
print(qtraj[:,humn.ub_jnts].shape)
# np.savetxt('3DSIPs_eUB.dat', np.hstack((ttraj.reshape(-1,1), np.linalg.norm(qtraj[:,humn.ub_jnts], axis=1).reshape(-1,1))), header='Time(s) qUB_Norm', comments='')

humn.plotCAMTraj(ttraj,humn.CAMTraj,linestyl='-',overlap=0)
np.savetxt('CAMTraj.dat', np.hstack((ttraj.reshape(-1,1), humn.CAMTraj)), header='Time(s) CAM_X(m) CAM_Y(m) CAM_Z(m)', comments='')
#Proposed method
# humn.k_ub=0.05 #00*model.opt.timestep #Set k_ub
# qtraj=humn.cart2joint(model,data,ttraj,0,1)
# humn.plotqTraj(ttraj,qtraj,linestyl='--',overlap=1)
# humn.plotCAMTraj(ttraj,humn.CAMTraj,linestyl='--',overlap=1)
# # only CAM control
# humn.k_ub=0
# qtraj=humn.cart2joint(model,data,ttraj,0,1)
# humn.plotqTraj(ttraj,qtraj,linestyl='-.',overlap=1)
# humn.plotCAMTraj(ttraj,humn.CAMTraj,linestyl='-.',overlap=1)
# #WLN method
# qtraj=humn.cart2joint(model,data,ttraj,1,1)
# humn.plotqTraj(ttraj,qtraj,linestyl='-.',overlap=1)
# humn.k_ub=0.1 #Reset k_ub
plt.show()
# plt.pause(0.1)
# humn.k_ub=0.1 #Reset k_ub

# Or open qtraj
# with open('qtraj.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     ttraj,qtraj = pickle.load(f)


#Run SIP traj again with this CAM trajectory
#AM_spl
# humn.AM_CMspl=[]
# for i in range(3):
#     humn.AM_CMspl.append(CubicSpline(ttraj,humn.CAMTraj[:,i]))
# # SIP to Kondo traj
# humn.sip2humn(1/100,humn.simend,humn.trn,humn.model,humn.data,0)

print('Time taken in SIP traj=',time.time()-st_time,'sec for sim time of ',simend,'sec, ...wait for 1 sec')
time.sleep(1)

# Generate spline from qtraj
humn.qspl=[]
for i in np.arange(0,model.nv):
    humn.qspl.append(CubicSpline(ttraj,qtraj[:,i])) #,bc_type='clamped'
#     #dqi.append(CubicSpline.derivative(qi[i]))

#Nonlinear optimization for gait generation
# humn.optGait(ttraj)

# Initialize
data=humn.q2data(data,humn.q0)

# contact force
# humn.mjfc(model,data)


# Simulate
DesData,ActData=humn.sim(model,data,humn.trn,simfreq,simend,saveVid=True)

#Save data for MATLAB plot
# DesData = np.array(DesData, dtype=object)
# ActData = np.array(ActData, dtype=object)

# Save the list to a .mat file
# scipy.io.savemat('myPyData.mat', {'DesData': DesData,'ActData':ActData})

#RMS error of COM self.rcomTraj_des[:, i]-self.rcomTraj_act[:, i]
print('RMS error of COM trajectory in x-direction =',np.sqrt(np.mean((humn.rcomTraj_des[:,0]-humn.rcomTraj_act[:,0])**2))*1000,'mm')
print('RMS error of COM trajectory in y-direction =',np.sqrt(np.mean((humn.rcomTraj_des[:,1]-humn.rcomTraj_act[:,1])**2))*1000,'mm')
print('RMS error of COM trajectory in z-direction =',np.sqrt(np.mean((humn.rcomTraj_des[:,2]-humn.rcomTraj_act[:,2])**2))*1000,'mm')
# Plot data
humn.plotData()
# myDataplots(DesData,ActData,humn)
# humn.plotData(model,data)