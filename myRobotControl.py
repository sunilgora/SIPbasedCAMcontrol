# This file contains functions used for footstep planning of Kondo khr3hv using SIP model
# Author : Sunil Gora, Shakti S. Gupta and Ashish Dutta
import numpy as np
import mujoco
import os,time
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize,LinearConstraint,differential_evolution
from scipy.linalg import solve_discrete_are
from moviepy import ImageSequenceClip
from myRobotEnv import myRobot,trnparam
from myRobotGait import myPlanner
from myRobotSIP import humn2SIP
from scipy.optimize import least_squares

class myController(myPlanner):
    def __init__(self,humn):
        #Run parent class init
        super().__init__(humn)
        #Import humn class variables
        # self.__dict__ = humn.__dict__.copy()
        

    # Instantaneous Joint angles from cartesian coord. with respect to stance leg foot
    def humnIKstep(self, model, data, q0, delt, docp, ocm, oleft, oright, Stlr, spno, ubjnts, zeroAM):
        data = self.q2data(data, q0)
        mujoco.mj_fwdPosition(model, data)

        #Base
        Jwb = np.zeros((3, model.nv))  # Base orientation jacobian
        Jwb[0:3, 3:6] = np.eye(3)

        #COM
        ocmi = data.subtree_com[0].copy()  # current COM position
        Jcm = np.zeros((3, model.nv))  # COM position jacobian
        mujoco.mj_jacSubtreeCom(model, data, Jcm, 0)

        # Ang momentum
        Iwb = np.zeros([3, model.nv])
        mujoco.mj_angmomMat(model, data, Iwb, 0)

        #Left foot
        olefti = data.site('left_foot_site').xpos.copy()  # current left foot position
        Rlefti = data.site('left_foot_site').xmat.copy() # current left foot orientation
        Jvleft = np.zeros((3, model.nv))  # Left foot center jacobian
        Jwleft = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, Jvleft, Jwleft, model.site('left_foot_site').id)

        #Right foot
        orighti = data.site('right_foot_site').xpos.copy()  # current right foot position
        Rrighti = data.site('right_foot_site').xmat.copy() # current right foot orientation
        Jvright = np.zeros((3, model.nv))  # right foot center jacobian
        Jwright = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, Jvright, Jwright, model.site('right_foot_site').id)

        quat_hip = data.qpos[3:7].copy()
        quat_left = np.zeros([4])
        quat_right = np.zeros([4])

        quat_conj = np.zeros([4])
        err_quat = np.zeros([4])
        err_ori_hip = np.zeros([3])
        err_ori_left = np.zeros([3])
        err_ori_right = np.zeros([3])
        # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
        mujoco.mju_negQuat(quat_conj, quat_hip)
        mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
        mujoco.mju_quat2Vel(err_ori_hip, err_quat, 1.0)

        if model.nu - len(ubjnts) >= 6:
            mujoco.mju_mat2Quat(quat_left, data.site('left_foot_site').xmat)
            mujoco.mju_negQuat(quat_conj, quat_left)
            mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
            mujoco.mju_quat2Vel(err_ori_left, err_quat, 1.0)

            # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
            mujoco.mju_mat2Quat(quat_right, data.site('right_foot_site').xmat)
            mujoco.mju_negQuat(quat_conj, quat_right)
            mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
            mujoco.mju_quat2Vel(err_ori_right, err_quat, 1.0)

        # mujoco.mju_mat2Quat(quat_left, data.site(model.site('left_foot_site').id).xmat)
        # mujoco.mju_negQuat(quat_left, quat_left)
        # mujoco.mju_quat2Vel(err_ori_left, quat_left, 1.0)

        # mujoco.mju_mat2Quat(quat_right, data.site(model.site('right_foot_site').id).xmat)
        # mujoco.mju_negQuat(quat_right, quat_right)
        # mujoco.mju_quat2Vel(err_ori_right, quat_right, 1.0)
        # lock upperbody
        if ubjnts.size:
            # ubjnts=np.arange(18,model.nv) #kondo khr3hv
            # ubjnts=np.append([6,7,8],np.arange(21,model.nv)) #MuJoCo humanoid model
            Jub = np.zeros((len(ubjnts), model.nv))  # Base orientation jacobian
            Jub[:, ubjnts] = np.eye(len(ubjnts))

        #Stance leg
        if Stlr[0]==1:
            Jvst=Jvleft.copy()
            Jwst=Jwleft.copy()
            dovst = oleft - olefti
        else:
            Jvst=Jvright.copy()
            Jwst=Jwright.copy()
            dovst = oright - orighti
        Avec = np.zeros([18 + len(ubjnts) + 3 + 3, model.nv])
        bvec = np.zeros([18 + len(ubjnts) + 3 + 3])
        # COM traj
        Avec[0:3, 0:model.nv] = Jcm - Jvst
        bvec[0:3] = (ocm - ocmi - dovst) + self.ZMPctrl*(docp- dovst)
        # Hip orient
        Avec[3:6, 0:model.nv] = Jwb
        bvec[3:6] = err_ori_hip  # np.zeros([3])
        # Left ankle lin vel
        Avec[6:9, 0:model.nv] = Jvleft - Jvst #+ (Stlr[0]==0)*(spno==1)*Jvst
        bvec[6:9] = oleft - olefti - dovst #+ (Stlr[0]==0)*(spno==1)*dovst
        # Left ankle ang vel
        Avec[9:12, 0:model.nv] = Jwleft
        bvec[9:12] = err_ori_left  # np.zeros([3])
        # Right ankle lin vel
        Avec[12:15, 0:model.nv] = Jvright - Jvst #+ (Stlr[1]==0)*(spno==1)*Jvst
        bvec[12:15] = oright - orighti - dovst #+ (Stlr[1]==0)*(spno==1)*dovst
        # Right ankle ang vel
        Avec[15:18, 0:model.nv] = Jwright
        bvec[15:18] = err_ori_right  # np.zeros([3])
        # Upper body joints
        if ubjnts.size:
            k_ub=self.AMctrl*self.k_ub + (1-self.AMctrl)*1 #0.1 #min(self.foot_size)/10 #1/100
            k_ref=self.k_ub**2/4 #10*delt #10*model.opt.timestep
            self.k_L =self.AMctrl*np.exp(-k_ub*np.linalg.norm((self.q0[ubjnts] - q0[ubjnts])))

            # self.k_L=(self.AMctrl*1/(1+k_ub*np.linalg.norm(( self.q0[ubjnts] - q0[ubjnts] )))) #[abs(docp[0]-dovst[0])/(self.foot_size[0]/10+abs(docp[0]-dovst[0])), abs(docp[1]-dovst[1])/(self.foot_size[1]/10+abs(docp[1]-dovst[1])), 1]
            # self.k_L=(self.AMctrl*1/(1+k_ub*np.linalg.norm(( self.q0[ubjnts] - q0[ubjnts] )))) #[abs(docp[0]-dovst[0])/(self.foot_size[0]/10+abs(docp[0]-dovst[0])), abs(docp[1]-dovst[1])/(self.foot_size[1]/10+abs(docp[1]-dovst[1])), 1]
            # print(self.k_L,k_ub)
            # k_ub=k_ub*np.linalg.norm(Iwb)
            # k_ub=(1-self.AMctrl) #min(self.foot_size)/10 #1/100
            # K_ub = 1*(1-K_AM)*(abs(np.zeros([len(ubjnts)]) - q0[ubjnts]))/np.linalg.norm(( np.zeros([len(ubjnts)]) - q0[ubjnts] ))
            # K_AM=(1-k_ub)*(1-min(1,np.linalg.norm(K_ub))) #max(1,self.AMctrl/(k_ub+np.linalg.norm(( np.zeros([len(ubjnts)]) - q0[ubjnts] )))) #[abs(docp[0]-dovst[0])/(self.foot_size[0]/10+abs(docp[0]-dovst[0])), abs(docp[1]-dovst[1])/(self.foot_size[1]/10+abs(docp[1]-dovst[1])), 1]
            # print(K_ub)
            # print(k_L)
            # Lock upper body joints
            Avec[18:18 + len(ubjnts), 0:model.nv] = Jub*k_ub*(1-self.k_L) #(k_ub/(k_ub+np.linalg.norm(docp-dovst)))
            # Avec[18:18 + len(ubjnts), ubjnts] = np.diag(K_ub)
            bvec[18:18 + len(ubjnts)] = ( self.q0[ubjnts] - q0[ubjnts] )*(1-self.k_L)*k_ref #*delt #*(1-self.k_L)* np.linalg.norm(Iwb) #delt #k_ub *(1-self.k_L)#max velocity is (q0-q)/delt
            # Zero angular momentum using upper body joints
            Avec[18 + len(ubjnts):18 + len(ubjnts) + 3, 0:model.nv] = 0.01*Iwb*self.k_L #/np.linalg.norm(Iwb) #(np.linalg.norm(docp-dovst)/(k_ub+np.linalg.norm(docp-dovst)))
            # Avec[18 + len(ubjnts), 0:model.nv] = 1*Iwb[0,:]*K_AM[0]
            # Avec[18 + len(ubjnts)+1, 0:model.nv] = 1*Iwb[1,:]*K_AM[1]
            # Avec[18 + len(ubjnts) + 2, 0:model.nv] = 1 * Iwb[2, :] * K_AM[2]
            bvec[18 + len(ubjnts):18 + len(ubjnts) + 3] = np.zeros([3])
            # #Sym. motion of upper body joints
            # Avec[18+len(ubjnts)+3:18+len(ubjnts)+6,ubjnts]=Iwb[:,ubjnts]
            # bvec[18+len(ubjnts)+3:18+len(ubjnts)+6] = np.zeros([3])
        else:
            Avec[18:18 + 3, 0:model.nv] = Iwb
            bvec[18:18 + 3] = np.zeros([3])

        # J=np.append(np.append(np.append(np.append(Jcm,Jwb,axis=0), np.append(Jvleft,Jwleft,axis=0),axis=0), np.append(Jvright,Jwright,axis=0), axis=0),Jub,axis=0)
        # delx=np.append(np.append(np.append( np.append(ocm-ocmi,np.zeros([3]),axis=0), np.append(oleft-olefti,np.zeros([3]),axis=0),axis=0), np.append(oright-orighti,np.zeros([3]),axis=0), axis=0),np.zeros([model.nv-18]),axis=0)
        if (model.nv - len(ubjnts)) < 18:  # Planer biped
            eqnJ1 = np.append(np.array([0, 2]), np.array([6, 8, 10, 12, 14, 16]))  # remove foot orientation
        else:  # Spatial biped
            eqnJ1 = np.append(np.arange(0, 3), np.arange(6, 18))

        J1 = Avec[eqnJ1, :].copy()
        delx1 = bvec[eqnJ1].copy() #/ delt
        dqN1 = np.matmul(np.linalg.pinv(J1), delx1)
        InJ1 = np.eye(model.nv) - np.matmul(np.linalg.pinv(J1), J1)
        if zeroAM == True:  # Zero ang momentum
            eqnJ2 = np.append(np.append(np.array([3, 4, 5]), np.arange(18, 18 + len(ubjnts))), np.arange(18 + len(ubjnts), 18 + len(ubjnts) + 3))
            # eqnJ2 = np.append(np.array([3, 4, 5]), np.arange(18 + len(ubjnts), 18 + len(ubjnts) + 2)) #Abt XY-axis
            # eqnJ2 = np.append(np.array([3, 4, 5]), np.arange(18 + len(ubjnts) +2, 18 + len(ubjnts) + 3)) #Abt Z-axis
            # eqnJ2 = np.append(np.array([3, 4, 5]),  np.arange(18 + len(ubjnts), 18 + len(ubjnts) + 3))  # Abt all axes
            # eqnJ2=np.arange(18+len(ubjnts),18+len(ubjnts)+2)
            # eqnJ2=np.array([18,19,21,22,23,25,26,27,model.nv,model.nv+1,model.nv+2])
            # eqnJ2 = np.append(np.arange(18, 18 + len(ubjnts)), np.arange(18 + len(ubjnts), 18 + len(ubjnts) + 3))  # Abt all axes

        else:
            eqnJ2 = np.append(np.array([3, 4, 5]), np.arange(18, 18 + len(ubjnts)))

        J2 = Avec[eqnJ2, :].copy()
        # Weighted least square to avoid joint limits
        # invsqrtW = np.ones(model.nv)
        # for i in self.ub_jnts:
        #     delHbydelTh = (np.pi/4+np.pi/4)**2*(2*q0[i]-0)/(4*(np.pi/4-q0[i])**4)
        #     print(i,delHbydelTh)
        #     invsqrtW[i] = 1 / (np.sqrt(1 + abs(delHbydelTh)))
        # print(invsqrtW)
        # J2 = J2 @ np.diag(invsqrtW)
        delx2 = bvec[eqnJ2].copy() #/ delt
        Jt2 = np.matmul(J2, InJ1)
        dqN2 = dqN1 + np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, dqN1))
        # dqN2 = dqN1 + np.diag(invsqrtW) @ np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, dqN1)) #for weighted least square
        # InJ2 = InJ1 - np.matmul(np.linalg.pinv(Jt2), Jt2)
        dq = dqN2.copy()  # +np.matmul(InJ2,qref-qi)

        # Integrate joint velocities to obtain joint positions.
        # qdes = q0 + dq * delt
        q = data.qpos.copy()  # Note the copy here is important.
        mujoco.mj_integratePos(model, q, dq, 1) #delt)
        # np.clip(q, *model.jnt_range.T, out=q)

        #qdes=data2q(data)
        qdes = 0 * data.qvel.copy()
        qqt = q[3:7].copy()
        qeulr = quat2euler(qqt)
        for i in np.arange(0, 3):
            qdes[i] = q[i].copy()
        for i in np.arange(3, 6):
            qdes[i] = qeulr[i - 3].copy()
        for i in np.arange(6, len(data.qvel)):
            qdes[i] = q[i + 1].copy()

        return qdes

    #Generate online gait for leg trajectories with current position and velocities
    def genWalkingGait(self,model,data,tcyc,step_len,step_width):
        # tcyc = (humn.step_len/humn.vel/2)
        self.tCMtraj = np.array([])
        self.oCMtraj = np.empty((0, 3))
        self.oCPtraj = np.empty((0, 3))
        self.tLtraj = np.array([])
        self.oLtraj = np.empty((0, 3))
        self.tRtraj = np.array([])
        self.oRtraj = np.empty((0, 3))

        #Current position and velocities: humn.r_com, humn.o_left, humn.o_right, humn.v_com
        # Update gait from simplified model of one cycle
        oLeft = self.o_left.copy()
        vLeft = self.v_left.copy()
        oRight = self.o_right.copy()
        vRight = self.v_right.copy()
        Stlr = self.Stlr.copy() 
        spno = 1  # single support phase
        vxSwing = step_len /tcyc
        # print(r_ftplac)
        # print('Foot Velocities:',vLeft,vRight) #Not followed as desired velocities, hence
        # if data.time>0:
        #     oLeft =  np.array([self.oLx(data.time), self.oLy(data.time), self.oLz(data.time)])  # Desired Left Pos
        #     vLeft = np.array([self.oLx(data.time,1), self.oLy(data.time,1), self.oLz(data.time,1)])  # Desired Left Vel
        #     oRight = np.array([self.oRx(data.time), self.oRy(data.time), self.oRz(data.time)])  # Desired Right Pos
        #     vRight = np.array([self.oRx(data.time,1), self.oRy(data.time,1), self.oRz(data.time,1)])  # Desired Right Vel



        tcm = [data.time, data.time + tcyc/2,  data.time + tcyc]  # time for current gait cycle
        # print("ocm:", ocm,'rcm:',self.r_com,'step_len:',step_len)
        if Stlr[0]==1:
            oct = [self.o_left, self.o_left, self.o_left]
            r_ftplac = [self.o_right[0]+step_len, self.o_right[1]+step_width, self.o_right[2]]
        else:
            oct = [self.o_right, self.o_right, self.o_right]
            r_ftplac = [self.o_left[0]+step_len, self.o_left[1]+step_width, self.o_left[2]]

        # print('Foot Placement:',r_ftplac)
        # time.sleep(10)
        r_cmfin = 0.5 * np.array([oct[0][0]+r_ftplac[0], oct[0][1]+r_ftplac[1], 2*self.r_com[2]]) # Final COM position
        ocm = [self.r_com, 0.5*(self.r_com + r_cmfin),  r_cmfin]  # COM positions

        self.tCMtraj = np.append(self.tCMtraj, tcm, axis=0)
        self.oCMtraj = np.vstack([self.oCMtraj, ocm])
        self.oCPtraj = np.vstack([self.oCPtraj, oct])
        if spno == 1:  # SSP
            if Stlr[0] == 1:  # left foot is stance
                zSw = min(self.zSw,abs(step_len)) * (vxSwing - self.v_right[0]) / vxSwing  # adjust step height based on current speed
                print('zSw:',zSw)
                # if abs(oRight[2] - ftplac[ncyc][3][2])>self.zSw/2: #Increase Step height on stairs if needed
                #     self.zSw=2*abs(oRight[2] - ftplac[ncyc][3][2])

                self.tLtraj = np.append(self.tLtraj, tcm, axis=0)
                self.oLtraj = np.vstack([self.oLtraj, oct])

                self.tRtraj = np.append(self.tRtraj, np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), axis=0)
                self.oRtraj = np.append(
                    np.append(self.oRtraj, np.array([oRight, 0.5 * (oRight + r_ftplac) + [0, 0, zSw]]), axis=0),
                    np.array([r_ftplac]), axis=0)
            else:
                zSw = min(self.zSw,abs(step_len)) * (vxSwing - self.v_left[0]) / vxSwing  # adjust step height based on current speed

                # if abs(oLeft[2] - ftplac[ncyc][3][2])>self.zSw/2: #increase step height if needed
                #     self.zSw=2*abs(oLeft[2] - ftplac[ncyc][3][2])
                self.tRtraj = np.append(self.tRtraj, tcm, axis=0)
                self.oRtraj = np.vstack([self.oRtraj, oct])
                self.tLtraj = np.append(self.tLtraj, np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), axis=0)
                self.oLtraj = np.append(
                    np.append(self.oLtraj, np.array([oLeft, 0.5 * (oLeft + r_ftplac) + [0, 0, zSw]]), axis=0),
                    np.array([r_ftplac]), axis=0)
        else:  # DSP
            Stlr = np.array([1, 1]) - Stlr
            self.tLtraj = np.append(self.tLtraj, tcm, axis=0)
            self.tRtraj = np.append(self.tRtraj, tcm, axis=0)
            ncyc=0
            if Stlr[0] == 1:  # left foot is stance
                ftplac = [[], oct, [], r_ftplac]
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[0], 0.5 * (oLeft[0] + ftplac[ncyc][3][0]), ftplac[ncyc][3][0]]),
                                    bc_type=((1, vLeft[0]), (1, 0.0)))
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[1], 0.5 * (oLeft[1] + ftplac[ncyc][3][1]), ftplac[ncyc][3][1]]),
                                    bc_type=((1, vLeft[1]), (1, 0.0)))
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[2], 0.5 * (oLeft[2] + ftplac[ncyc][3][2]), ftplac[ncyc][3][2]]),
                                    bc_type=((1, vLeft[2]), (1, 0.0)))
                self.oLtraj = np.append(self.oLtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[0], 0.5 * (oRight[0] + ftplac[ncyc][1][0]), ftplac[ncyc][1][0]]), bc_type=((1, vRight[0]), (1, 0.0)))
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[1], 0.5 * (oRight[1] + ftplac[ncyc][1][1]), ftplac[ncyc][1][1]]), bc_type=((1, vRight[1]), (1, 0.0)))
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[2], 0.5 * (oRight[2] + ftplac[ncyc][1][2]), ftplac[ncyc][1][2]]), bc_type=((1, vRight[2]), (1, 0.0)))
                self.oRtraj = np.append(self.oRtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)
            else:
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[0], 0.5 * (oRight[0] + ftplac[ncyc][3][0]), ftplac[ncyc][3][0]]), bc_type=((1, vRight[0]), (1, 0.0)))
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[1], 0.5 * (oRight[1] + ftplac[ncyc][3][1]), ftplac[ncyc][3][1]]), bc_type=((1, vRight[1]), (1, 0.0)))
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[2], 0.5 * (oRight[2] + ftplac[ncyc][3][2]), ftplac[ncyc][3][2]]), bc_type=((1, vRight[2]), (1, 0.0)))
                self.oRtraj = np.append(self.oRtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[0], 0.5 * (oLeft[0] + ftplac[ncyc][1][0]), ftplac[ncyc][1][0]]),
                                    bc_type=((1, vLeft[0]), (1, 0.0)))
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[1], 0.5 * (oLeft[1] + ftplac[ncyc][1][1]), ftplac[ncyc][1][1]]),
                                    bc_type=((1, vLeft[1]), (1, 0.0)))
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[2], 0.5 * (oLeft[2] + ftplac[ncyc][1][2]), ftplac[ncyc][1][2]]),
                                    bc_type=((1, vLeft[2]), (1, 0.0)))
                self.oLtraj = np.append(self.oLtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)
        # print(self.oCMtraj)
        self.oCMx = CubicSpline(self.tCMtraj, self.oCMtraj[:, 0])
        self.oCMy = CubicSpline(self.tCMtraj, self.oCMtraj[:, 1])
        self.oCMz = CubicSpline(self.tCMtraj, self.oCMtraj[:, 2])
        # oCMi=np.array([oCMx,oCMy,oCMz])
        self.oLx = CubicSpline(self.tLtraj, self.oLtraj[:, 0], bc_type=((1, vLeft[0]), (1, 0.0)))
        self.oLy = CubicSpline(self.tLtraj, self.oLtraj[:, 1], bc_type=((1, vLeft[1]), (1, 0.0)))
        self.oLz = CubicSpline(self.tLtraj, self.oLtraj[:, 2], bc_type=((1, vLeft[2]), (1, 0.0)))
        # oLi=np.array([oLx,oLy,oLz])
        self.oRx = CubicSpline(self.tRtraj, self.oRtraj[:, 0], bc_type=((1, vRight[0]), (1, 0.0)))
        self.oRy = CubicSpline(self.tRtraj, self.oRtraj[:, 1], bc_type=((1, vRight[1]), (1, 0.0)))
        self.oRz = CubicSpline(self.tRtraj, self.oRtraj[:, 2], bc_type=((1, vRight[2]), (1, 0.0)))    
        self.oCPx = CubicSpline(self.tCMtraj, self.oCPtraj[:, 0])
        self.oCPy = CubicSpline(self.tCMtraj, self.oCPtraj[:, 1])
        self.oCPz = CubicSpline(self.tCMtraj, self.oCPtraj[:, 2])
        # print(tcyc,self.oLtraj,self.oRtraj)
        # time.sleep(10)



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
                mujoco.mj_contactForce(model, data, i, fci)
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

    #def mjaref(self,cnt):


    def init_controller(self,Kp,Kv,Ki):
        self.Kp=Kp
        self.Kv=Kv
        self.Ki=Ki

    def controller(self,model,data):
        self.tau = 0 * data.ctrl
        # Desired traj
        # thdes=np.zeros([model.nu])
        # dthdes = np.zeros([model.nu])
        self.qdes = np.zeros([model.nv])
        self.dqdes = np.zeros([model.nv])
        self.ddqdes = np.zeros([model.nv])

        # Current joint angles
        q = self.data2q(data)

        # current COM position
        self.r_com = data.subtree_com[0].copy()
        # current COP position
        self.ocp_des = np.array( [self.oCPx(data.time), self.oCPy(data.time), self.oCPz(data.time)])  # Desired COP Pos

        #Desired cartesian trajt
        self.ocm_des = np.array( [self.oCMx(data.time), self.oCMy(data.time), self.oCMz(data.time)])  # Desired COM Pos
        self.oL_des = np.array([self.oLx(data.time), self.oLy(data.time), self.oLz(data.time)])  # Desired Left Pos
        self.oR_des = np.array([self.oRx(data.time), self.oRy(data.time), self.oRz(data.time)])  # Desired Right Pos

        # Normal contact force
        self.mjfc(model, data)

        # COP pos
        if abs(self.fcn) > 0:
            self.r_cop = self.rf / abs(self.fcn)
        else:  # No contact
            self.r_cop = self.ocp_des  # np.array([np.nan, np.nan, np.nan])

        if self.ocp_des[2] > self.r_com[2]:
            if abs(self.fcn) == 0:
                self.r_cop[2] = 0
            self.ocp_des[0] = self.ocm_des[0] + (self.ocm_des[0] - self.ocp_des[0]) * abs(
                self.r_cop[2] - self.ocm_des[2]) / (self.ocp_des[2] - self.ocm_des[2])
            self.ocp_des[1] = self.ocm_des[1] + (self.ocm_des[1] - self.ocp_des[1]) * abs(
                self.r_cop[2] - self.ocm_des[2]) / (self.ocp_des[2] - self.ocm_des[2])
            self.ocp_des[2] = self.r_cop[2]
        if abs(self.fcn) == 0:
            self.r_cop = self.ocp_des  # np.array([np.nan, np.nan, np.nan])
        # Angular momentum
        self.Iwb = np.zeros([3, model.nv])
        Lw = self.Iwb @ (0 * data.qvel)
        mujoco.mj_angmomMat(model, data, self.Iwb, 0)
            
        if self.KINctrl==True:
            if abs(self.fcl[0]) > 0 and abs(self.fcr[0]) == 0:
                Stlr=[1, 0]
                spno=1
            elif abs(self.fcr[0]) > 0 and abs(self.fcr[0]) == 0:
                Stlr=[0, 1]
                spno=1
            else:
                spno=2
                if self.o_right[0] > self.o_left[0]:
                    Stlr = [1, 0]
                else:
                    Stlr = [0, 1]
            spno=2 #No effect of support phase
            # print(self.ocm_des[0]-self.r_com[0])

            self.qdes = self.humnIKstep(model,data,q,model.opt.timestep,self.ocp_des-self.r_cop,self.ocm_des,self.oL_des,self.oR_des,Stlr,spno,self.ub_jnts,self.AMctrl)
        else:
            # Follow the pre-defined joint trajectory

            for i in np.arange(0, model.nv):
                self.qdes[i] = self.qspl[i](1 * data.time)  # + self.q_err[i]
                self.dqdes[i] = self.qspl[i](1 * data.time, 1)
                self.ddqdes[i] = self.qspl[i](1 * data.time, 2)
            #     # qdes[i+6]=thdes[i].copy()
                # data.qacc[i]=ddqdes[i].copy()
        #Desired joint velocity
        self.dqdes = 1*(self.qdes - q) / model.opt.timestep
        # PID Controller for torque control mode
        self.tau_PID = 0*data.ctrl
        self.tau_PID = self.PIDcontrol(model, data)
        if self.posCTRL==True:
            self.tau=self.qdes[6:model.nv].copy() #Position control mode
        else:
            # Gear Ratio for tau_PID to data.ctrl
            for i in range(model.nu):
                self.tau[i] = self.tau_PID[i] / model.actuator_gear[i][0]
        # print(max(abs(self.tau)),max(abs(self.dqdes)))
        # print(min(abs(self.tau)))
        # return self.tau_PD


        # # PD control
        # self.tau += self.PDcontrol(model, data)
        #
        # self.q_err = np.zeros([model.nv])
        #
        # # Ankle Torque Control
        # # tau_ankle = 0
        #
        # # COM control # Choi et al. without ZMP control
        # self.tau += self.COMcontrol(model,data)
        #
        # # ZMP control # Choi et al.
        # self.tau += self.ZMPcontrol(model,data)
        #
        # # Angular momentum control
        # self.tau += self.AMcontrol(model,data)
        #
        # # FW control # FW inverted pendulum to control COM
        # self.tau +=self.FWcontrol(model,data)

        # Controller data.ctrl
        #data.ctrl = tau_PD + COMctrl * tau_COM + AMctrl * tau_AM + FWctrl * tau_FW

        # Inverse dynamics for model-based control
        # tauid = tauinvd(model, data, self.ddqdes)
        # print(max(data.actuator_force))
        # self.tau +=tauid

        #Torque limits
        # np.clip(self.tau, -1.37, 1.37, out=self.tau)

        return self.tau

    def PIDcontrol(self, model, data):
        # Current joint angles
        q = self.data2q(data)
        # Desired joint angles

        # for i in np.arange(0, model.nv):
        #     self.qdes[i] = self.qspl[i](1 * data.time) #+ self.q_err[i]
        #     self.dqdes[i] = self.qspl[i](1 * data.time, 1)
        #     self.ddqdes[i] = self.qspl[i](1 * data.time, 2)
        #     # qdes[i+6]=thdes[i].copy()
        #     # data.qacc[i]=ddqdes[i].copy()
        # PID Controller
        self.tau_PID = 0 * data.ctrl

        for i in range(model.nu):
            self.tau_PID[i] = (self.Kp[i]) * (self.qdes[i + 6] - q[i + 6]) + self.Kv[i] * (
                    self.dqdes[i + 6] - data.qvel[i + 6]) + self.Ki[i] * (self.Eqdt[i+6])
            #Integral error
            self.Eqdt[i+6]=self.Eqdt[i+6]+(self.qdes[i + 6] - q[i + 6])*model.opt.timestep

        return self.tau_PID


    def COMcontrol(self,model, data):
        self.ocm_des = np.array(
            [self.oCMx(data.time), self.oCMy(data.time), self.oCMz(data.time)])  # Desired COM Pos
        self.r_com = data.subtree_com[0].copy()  # current COM position
        # dr_com = data.subtree_linvel[0].copy()
        if self.COMctrl == 0:
            self.tau_COM = 0 * data.ctrl
        else:
            jnts = range(model.nv)
            #tau_COM, q_err, dq_err = self.COMcontrol(model, data, ocm_des, r_com, self.Kp, self.Kv, jnts)

            Jcm = np.zeros((3, model.nv))  # COM position jacobian
            Jct = np.zeros((3, model.nv))  # Stance foot center position jacobian
            mujoco.mj_jacSubtreeCom(model, data, Jcm, 0)

            if abs(self.fcl[0])>0 and abs(self.fcr[0])==0:
                mujoco.mj_jacSite(model,data, Jct, None, data.site("left_foot_site").id)
            elif abs(self.fcr[0])>0 and abs(self.fcr[0])==0:
                mujoco.mj_jacSite(model,data, Jct, None, data.site("right_foot_site").id)
            else:
                if self.o_right[0]>self.o_left[0]:
                    mujoco.mj_jacSite(model, data, Jct, None, data.site("left_foot_site").id)
                else:
                    mujoco.mj_jacSite(model, data, Jct, None, data.site("right_foot_site").id)

            J1 = np.zeros([model.nv - len(jnts), model.nv])
            J1[:, [x for x in range(model.nv) if x not in jnts]] = np.eye(model.nv - len(jnts))
            InJ1 = np.eye(model.nv) - np.matmul(np.linalg.pinv(J1), J1)
            J2 = Jcm - Jct
            delx2 = 1 * self.COMctrl * (self.ocm_des - self.r_com) / model.opt.timestep
            Jt2 = np.matmul(J2, InJ1)
            # dq_err = 1 * np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, data.qvel))
            dq_err=1*np.matmul(np.linalg.pinv(Jcm - Jct), delx2)
            self.q_err = self.q_err + dq_err * model.opt.timestep

            self.tau_COM = 0 * data.ctrl
            for i in np.arange(0, model.nu):
                self.tau_COM[i] = self.Kp[i] * (dq_err[i + 6] * model.opt.timestep) + self.Kv[i] * (dq_err[i + 6])  # (dqdes[i+6]-dq[i+6])

            # Modify desired joint traj
            self.dqdes = self.dqdes + dq_err
            self.qdes = self.qdes + dq_err * model.opt.timestep


        return self.tau_COM

    def ZMPcontrol(self,model, data):
        self.ocp_des = np.array(
            [self.oCPx(data.time), self.oCPy(data.time), self.oCPz(data.time)])  # Desired COP Pos
        # Normal contact force
        self.mjfc(model, data)
        # COP pos
        if abs(self.fcn) > 0:
            self.r_cop = self.rf / abs(self.fcn)
        else: #No contact
            self.r_cop = self.ocp_des #np.array([np.nan, np.nan, np.nan])

        if self.ocp_des[2]>self.r_com[2]:
            if abs(self.fcn) == 0:
                self.r_cop[2]=0
            self.ocp_des[0]=self.ocm_des[0] + (self.ocm_des[0]-self.ocp_des[0])*abs(self.r_cop[2]-self.ocm_des[2])/(self.ocp_des[2]-self.ocm_des[2])
            self.ocp_des[1] = self.ocm_des[1] +  (self.ocm_des[1] - self.ocp_des[1])*abs(self.r_cop[2] - self.ocm_des[2])/(self.ocp_des[2] - self.ocm_des[2])
            self.ocp_des[2] = self.r_cop[2]
        if abs(self.fcn) == 0:
            self.r_cop = self.ocp_des #np.array([np.nan, np.nan, np.nan])

        # self.r_com = data.subtree_com[0].copy()  # current COM position
        # dr_com = data.subtree_linvel[0].copy()
        if self.ZMPctrl == 0:
            self.tau_ZMP = 0 * data.ctrl
        else:
            jnts = range(model.nv)
            #tau_COM, q_err, dq_err = self.COMcontrol(model, data, ocm_des, r_com, self.Kp, self.Kv, jnts)

            Jcm = np.zeros((3, model.nv))  # COM position jacobian
            Jct = np.zeros((3, model.nv))  # Stance foot center position jacobian
            mujoco.mj_jacSubtreeCom(model, data, Jcm, 0)

            if abs(self.fcl[0])>0 and abs(self.fcr[0])==0:
                mujoco.mj_jacSite(model,data, Jct, None, data.site("left_foot_site").id)
            elif abs(self.fcr[0])>0 and abs(self.fcr[0])==0:
                mujoco.mj_jacSite(model,data, Jct, None, data.site("right_foot_site").id)
            else:
                if self.o_right[0] > self.o_left[0]:
                    mujoco.mj_jacSite(model, data, Jct, None, data.site("left_foot_site").id)
                else:
                    mujoco.mj_jacSite(model, data, Jct, None, data.site("right_foot_site").id)

            J1 = np.zeros([model.nv - len(jnts), model.nv])
            J1[:, [x for x in range(model.nv) if x not in jnts]] = np.eye(model.nv - len(jnts))
            InJ1 = np.eye(model.nv) - np.matmul(np.linalg.pinv(J1), J1)
            J2 = Jcm - Jct
            delx2 =  1 * self.ZMPctrl * (self.ocp_des - self.r_cop) / model.opt.timestep
            # print(delx2,self.ocp_des,self.r_cop)
            Jt2 = np.matmul(J2, InJ1)
            # dq_err = 1 * np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, data.qvel))
            dq_err=1*np.matmul(np.linalg.pinv(Jcm - Jct), delx2)
            self.q_err = self.q_err + dq_err * model.opt.timestep

            self.tau_ZMP = 0 * data.ctrl
            for i in np.arange(0, model.nu):
                self.tau_ZMP[i] = self.Kp[i] * (dq_err[i + 6]* model.opt.timestep) + self.Kv[i] * (dq_err[i + 6])  # (dqdes[i+6]-dq[i+6])

            # Modify desired joint traj
            self.dqdes = self.dqdes + dq_err
            self.qdes = self.qdes + dq_err * model.opt.timestep


        return self.tau_ZMP

    def AMcontrol(self,model, data):
        # Zero angular momentum is feasible with no contact force. In dynamics, The Zero angular momentum contradict with the contact force.
        # Angular momentum
        self.Iwb = np.zeros([3, model.nv])
        Lw = self.Iwb @ (0*data.qvel)
        mujoco.mj_angmomMat(model, data, self.Iwb, 0)

        self.tau_AM = 0 * data.ctrl
        if self.AMctrl == 1:
            jnts = self.ub_jnts
            #tau_AM, q_err, dq_err = AMcontrol(model, data, Iwb, dqdes, Lw, self.Kp, self.Kv, jnts)

            J1 = np.zeros([model.nv - len(jnts), model.nv])
            J1[:, [x for x in range(model.nv) if x not in jnts]] = np.eye(model.nv - len(jnts))
            InJ1 = np.eye(model.nv) - np.matmul(np.linalg.pinv(J1), J1)
            J2 = self.Iwb
            delx2 = Lw  # np.zeros([3])
            Jt2 = np.matmul(J2, InJ1)
            dq_err = 1 * np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, self.dqdes))
            self.q_err += dq_err * model.opt.timestep

            #self.tau_AM = 0 * data.ctrl
            for i in np.arange(0, model.nu):
                self.tau_AM[i] = self.Kp[i] * (dq_err[i + 6]* model.opt.timestep) + self.Kv[i] * ((dq_err[i + 6]))  # (dqdes[i+6]-dq[i+6])

            # Modify desired joint traj
            self.dqdes = self.dqdes + dq_err
            self.qdes = self.qdes + dq_err* model.opt.timestep

        return self.tau_AM

    def FWcontrol(self,model,data):
        if self.FWctrl == 0:
            self.tau_FW = 0 * data.ctrl
        else:
            self.tau_FW = 0 * data.ctrl
            jnts = 0
            # th_des=-np.arctan2(ocm_des[2],ocm_des[0])
            # th_FW=-np.arctan2(r_com[2],r_com[0])
            # tau_FW[0]=(data.ncon>0)*(-Kp[0]*(th_des-th_FW)-Kv[0]*(0))
            comX_err = (self.ocm_des[0] - self.r_com[0])
            self.tau_FW[jnts] = (data.ncon > 0) * (-self.Kp[0] * (comX_err) - self.Kv[0] * (0))
            self.tau[jnts] = 0

        return self.tau_FW

    def sdmodel(self,model,data):
        data.xfrc_applied = 0 * data.xfrc_applied # f/tau applied on body
        lfoot_bodyid = 10000 # model.site_bodyid[data.site("left_foot_site").id]
        rfoot_bodyid = 10000 # model.site_bodyid[data.site("right_foot_site").id]
        kn=self.m*9.81/0.003/4
        cn=10*0.3*np.sqrt(kn)
        ct=10000
        mu=1
        for i in range(0,model.ngeom): #check for all geom contact
            f_geom = np.zeros(3)
            r_body2geom = np.zeros(3)
            if model.geom_bodyid[i]==lfoot_bodyid and model.geom_contype[i]==1:
                # print(i)
                rct_vec = data.geom_xpos[i]-[0,0,model.geom_size[i][0]]
                Jvct = np.zeros((3, model.nv))  #  jacobian
                Jwct = np.zeros((3, model.nv))  #  jacobian
                mujoco.mj_jacGeom(model, data, Jvct, Jwct, i)
                drct_vec=np.matmul(Jvct,data.qvel)
                # print(drct_vec)

                if rct_vec[2] < 0:
                    f_geom[2] = kn*abs(rct_vec[2])+cn*(drct_vec[2]<0)*abs(drct_vec[2])
                    f_geom[0:2] = -ct*(drct_vec[0:2])
                    if np.linalg.norm(f_geom[0:2])>mu*abs(f_geom[2]):
                        f_geom[0:2]=mu*f_geom[0:2]/np.linalg.norm(f_geom[0:2])

                    if data.time>0.05:
                        self.xn=np.append(self.xn,abs(rct_vec[2]))
                        self.dxn=np.append(self.dxn,(drct_vec[2]<0)*abs(drct_vec[2]))
                        self.fn=np.append(self.fn,f_geom[2])

                r_body2geom = rct_vec - data.xpos[lfoot_bodyid]
                data.xfrc_applied[lfoot_bodyid] += np.append(f_geom, np.cross(r_body2geom, f_geom))
                # plt.figure(45)
                # plt.plot(data.time, f_geom[2], '.r')
                # plt.pause(0.00001)

            elif model.geom_bodyid[i]==rfoot_bodyid and model.geom_contype[i]==1:
                rct_vec = data.geom_xpos[i] - [0, 0, model.geom_size[i][2]] #box #[0, 0, model.geom_size[i][0]] ball
                Jvct = np.zeros((3, model.nv))  # jacobian
                Jwct = np.zeros((3, model.nv))  # jacobian
                mujoco.mj_jacGeom(model, data, Jvct, Jwct, i)
                drct_vec=np.matmul(Jvct,data.qvel)

                if rct_vec[2] < 0:
                    # print(rct_vec[2],data.geom_xpos[i],model.geom_size[i][2],data.geom_xpos[i] - [0, 0, model.geom_size[i][0]])
                    f_geom[2] = kn * abs(rct_vec[2]) + cn * (drct_vec[2] < 0) * abs(drct_vec[2])
                    f_geom[0:2] = -ct * (drct_vec[0:2])
                    if np.linalg.norm(f_geom[0:2]) > mu * abs(f_geom[2]):
                        f_geom[0:2] = mu * f_geom[0:2] / np.linalg.norm(f_geom[0:2])

                    if data.time>0.05:
                        self.xn=np.append(self.xn,abs(rct_vec[2]))
                        self.dxn=np.append(self.dxn,(drct_vec[2]<0)*abs(drct_vec[2]))
                        self.fn=np.append(self.fn,f_geom[2])


                r_body2geom = rct_vec - data.xpos[rfoot_bodyid]
                data.xfrc_applied[rfoot_bodyid] += np.append(f_geom, np.cross(r_body2geom, f_geom))
                # plt.figure(45)
                # plt.plot(data.time, f_geom[2], '.b')
                # plt.pause(0.00001)

        #Estimate parameters from SD model
        if len(self.fn)>0:
            print(len(self.fn))
            #print('xn,dxn,fn:',self.xn,self.dxn,self.fn)
            #print(np.linalg.pinv(np.vstack((self.xn, self.dxn)).T),self.fn)
            kc=np.matmul(np.linalg.pinv(np.vstack((self.xn, self.dxn)).T),self.fn)
            print('kn_est,kn',kc[0],kn,'cn_est,cn:',kc[1],cn)
            time.sleep(10)

        # print(data.xfrc_applied)
        return data.xfrc_applied



    def sim(self,model,data,trn,simfreq,simend,saveVid=False):

        # Humanoid parameters
        self.mj2humn(model, data)

        # Parameters of SIP
        sip = humn2SIP(self,trn, model, data)
        sip.trn.cntplane(sip.qcp,sip.spno)

        # Mocap body we will control with our mouse. for COP
        mocap_id_COP = model.body("COP").mocapid[0]
        mocap_id_COM_des = model.body("COM_des").mocapid[0]

        ActData = []
        DesData = []
        
        if saveVid == True:
            # Create a renderer
            renderer = mujoco.Renderer(model, width=1280, height=720)

            frames = []
            #duration = 5  # seconds
            framerate = int(simfreq/4) #60 #fps # saved Vid is 0.25 x real-time

        with mujoco.viewer.launch_passive(
                model=model, data=data, show_left_ui=True, show_right_ui=False
        ) as viewer:

            # Initialize the camera view to that of the free camera.
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Visualization.
            # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE #Site frame
            # viewer.opt.flags[2] = 1  # Joints
            # viewer.opt.flags[4] = 1  # Actuators
            # viewer.opt.flags[14] = 1 #Contact Points
            viewer.opt.flags[16] = 1 #Contact Forces
            viewer.opt.flags[18] = 1 #Transparent
            # viewer.opt.flags[20] = 1 #COM

            #Change viewer angle
            # viewer.cam.azimuth = 0
            # viewer.cam.elevation = 0# -20

            if saveVid == True:
                # Make new camera, set distance.
                camera = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, camera)
                camera.distance = self.cam_dist
                # Change view angle in degrees if required
                camera.azimuth = 30 #-30 #np.pi/3
                camera.elevation = 0#30 #np.pi/6

                # Enable contact force visualisation.
                scene_option = mujoco.MjvOption()
                scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            # Update scene and render
            # viewer.sync()

            # print("Press any key to proceed.")
            # key = keyboard.read_key()
            print("Simulation starting.....")
            time.sleep(2)

            while viewer.is_running() and data.time < simend:
                time_prev = data.time

                clock_start = time.time()
                while (data.time - time_prev < 1.0 / simfreq) and data.time < simend:
                    # Current joint angles
                    #q = data2q(data)
                    #dq = data.qvel.copy()

                    if self.ftctrl==1 and data.time>=self.Tsip:

                        # Humanoid parameters
                        self.mj2humn(model, data)

                        sip.qcm=self.r_com.copy()
                        sip.dqcm=self.v_com.copy()
                        #sip.dth=self.dq[3:6].copy()
                        if self.spno==1:
                            if self.Stlr[0]==1:
                                sip.qcp=self.o_left
                            else:
                                sip.qcp=self.o_right

                        # Generate SIP walking pattern *** For simTime *** for one cycle i.e. SSP+DSP
                        sipdata,ftplac = sip.sipstep(data.time,simend,simfreq,1)
                        self.Tsip=sipdata[-1][0]

                        # SIP to Kondo traj
                        # self.sip2humn(sipdata, ftplac)

                        # Generate Gait / Cartesian traj to joint space traj
                        ttraj = np.arange(data.time, self.Tsip, min(1 / 1000,(self.Tsip-data.time)/2))
                        #qtraj = self.cart2joint(model, data, ttraj, 0)
                        # Or open qtraj
                        # with open('qtraj.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
                        #     ttraj, qtraj = pickle.load(f)


                        # Generate spline from qtraj
                        # self.qspl = []
                        # for i in np.arange(0, model.nv):
                        #     self.qspl.append(CubicSpline(ttraj, qtraj[:, i]))
                        #     # dqi.append(CubicSpline.derivative(qi[i]))

                        # Change SSP/DSP
                        if self.spno==2:
                            self.Stlr=np.array([1, 1])-self.Stlr
                            data0 = deepcopy(data)
                        self.spno = 3 - self.spno

                    # Spring damper contact model
                    # data.xfrc_applied=self.sdmodel(model,data).copy()

                    #Forward dyn
                    data.ctrl=self.controller(model,data).copy()
                    # data.ctrl=0*data.ctrl #Zero Torque
                    mujoco.mj_step(model, data)  # Forward dynamics

                    # print('solver_fwdinv[2] flag',data.solver_fwdinv)

                    # Inv dyn
                    # mujoco.mj_inverse(model, data)  # Inverse  dynamics
                    # print(data.qfrc_inverse[self.left_legjnts],data.qfrc_inverse[self.right_legjnts])
                    # for i in np.arange(0, model.nv):
                    #     self.qdes[i] = self.qspl[i](1 * data.time)
                    #     self.dqdes[i] = self.qspl[i](1 * data.time, 1)
                    #     self.ddqdes[i] = self.qspl[i](1 * data.time, 2)
                    #     # qdes[i+6]=thdes[i].copy()
                    #     # data.qacc[i]=ddqdes[i].copy()
                    # data = self.q2data(data, self.qdes)  # Joint traj for inv dynamics
                    # data.qvel = self.dqdes
                    # data.qacc = self.ddqdes
                    # data.time = data.time + 1 / simfreq

                    # Terrain identification - solref,solimp
                    if self.ftctrl==1 and self.spno==1: #data.time>0.9*self.Tsip:
                        # MuJoCo to Humanoid Parameters
                        self.mj2humn(model, data)

                        # Normal contact force
                        # self.mjfc(model, data)

                        if self.spno==1:
                            if self.Stlr[0]==1:
                                cntpt=self.o_left
                            else:
                                cntpt=self.o_right
                        else:
                            if self.Stlr[0]==1:
                                cntpt=self.o_right
                            else:
                                cntpt=self.o_left

                        # Find the contact geom id
                        trn.cntplane(cntpt,1)
                        # a0vec = np.zeros([data.nefc])
                        # mujoco.mj_mulJacVec(model, data, a0vec,data.qacc_smooth)  # Unconstrained acceleration in contact space #1/m*(-m*9.81) #J*data.qacc_smooth --Unconstrained acceleration
                        # fmj,fsd,defmj,margmj=mjforce(model,data)
                        # print('efc_force,fmj= ',data.efc_force,fmj)
                        try:
                            if data.time>=data0.time:
                                # print(trn.efc_f)
                                trn.q.append(data.qpos)
                                trn.dq.append(data.qvel)
                                trn.ddq.append(data.qacc)
                                trn.efc_f.append(data.efc_force)  # contact force
                        except:
                            pass

                        for item in data.contact:
                            if item.geom1 == trn.cntgeomid or item.geom2 == trn.cntgeomid:
                                efc_id = item.efc_address
                                trn.r.append(data.efc_pos[efc_id] - data.efc_margin[efc_id])  # deformation
                                trn.rdot.append(data.efc_vel[efc_id])  # deformation rate
                                trn.aref.append(data.efc_aref[efc_id])
                                # trn.A.append(data.efc_diagApprox[efc_id])
                                trn.f.append(data.efc_force[efc_id])  # contact force


                        if (self.spno==1)*(len(trn.f)>0) and data.time>=self.Tsip: # len(trn.r)==100: # Find solref and solimp for cntpt
                            # Estimate parameters of Spring-Damper model from Mj model
                            if len(trn.f) > 0:
                                print(len(trn.f))
                                # print('xn,dxn,fn:',self.xn,self.dxn,self.fn)
                                # print(np.linalg.pinv(np.vstack((self.xn, self.dxn)).T),self.fn)
                                kc = np.matmul(np.linalg.pinv(np.vstack( (-np.array(trn.r), -1 * (np.array(trn.rdot) < 0) * np.array(trn.rdot))).T), np.array(trn.f))
                                if np.any(kc<0):
                                    def fun(x):
                                        fval=np.linalg.norm(trn.f - np.matmul((np.vstack((-np.array(trn.r), -1 * (np.array(trn.rdot) < 0) * np.array(trn.rdot))).T), np.array(x)))
                                        return fval
                                    lcon=LinearConstraint(np.eye(2),[0,0])
                                    kc=minimize(fun,[1,1],constraints=lcon).x

                                #print(kc)
                                print('SD model in Mj model, kn_est:', kc[0], 'cn_est:', kc[1])
                                # time.sleep(10)
                            # Estimate parameters of Mj model
                            #trn.paramidentify()
                            trn=self.paramidentify(model,data0,trn,data.time)
                            print('Actual solref=', np.round(model.geom_solref[trn.cntgeomid], 2))
                            print('Calc solref=', np.round(trn.cntsolref, 2))
                            print('Actual solimp=', model.geom_solimp[trn.cntgeomid])
                            print('Calc solimp=', trn.cntsolimp)
                            #Update sip.trn
                            sip.trn.solref[trn.cntgeomid]=trn.cntsolref.copy()
                            sip.trn.solimp[trn.cntgeomid]=trn.cntsolimp.copy()
                            #Erase saved data
                            trn.r=[]
                            trn.rdot=[]
                            trn.aref=[]
                            trn.f=[]

                    # Work Done

                    for i in range(model.nu):
                        #self.WD += abs(data.ctrl[i] * model.actuator_gear[i][0] * data.qvel[i+6] * model.opt.timestep)
                        self.WD += abs(data.actuator_force[i] * data.qvel[i+6] * model.opt.timestep)


                if (data.time >= simend):
                    break

                # MuJoCo to Humanoid Parameters
                self.mj2humn(model, data)
                # print(data.ctrl[self.left_legjnts - 6], data.ctrl[self.right_legjnts - 6])
                # plt.figure(10)
                # plt.plot(data.time, self.v_com[0], '.r')
                # plt.plot(data.time, self.v_com[1], '.g')
                # plt.plot(data.time, self.v_com[2], '.b')
                # plt.pause(0.001)

                # Normal contact force
                # print('self.fmj:')
                self.mjfc(model, data)
                # print('ncon,efc_force',data.ncon,data.efc_force)
                # fmj,fsd,fdef,dddef=mjforce(model,data)
                # print('mj_force',fmj)
                # time.sleep(10)
                # print('nefc,efc_pos,efc_force',data.nefc,data.efc_pos,data.efc_force)
                # print('efc_aref,force',data.efc_aref,data.efc_force)
                #COP COM mocap
                if abs(self.fcn) > 0:
                    r_cop = self.rf / abs(self.fcn)
                    # Set the target position of the end-effector site.
                    data.mocap_pos[mocap_id_COP, 0:3] = r_cop
                else:
                    r_cop = np.array([np.nan, np.nan, np.nan])
                    # Set the target position of the end-effector site.
                    data.mocap_pos[mocap_id_COP, 0:3] = np.array([1000, 1000, 1000])

                sipCnt=np.array([self.oCPx(data.time),self.oCPy(data.time),self.oCPz(data.time)])
                sipCOM=np.array([self.oCMx(data.time),self.oCMy(data.time),self.oCMz(data.time)])
                data.mocap_pos[mocap_id_COM_des, 0:3] = sipCOM
                #Visualize SIP Model
                # iterator for decorative geometry objects
                idx_geom = 0
                for i in range(100):
                    # mj Geometry from vyankatesh's code
                    sipPt=sipCOM+i/100*(sipCnt-sipCOM)
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[idx_geom],
                                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                        size=[0.005, 0, 0],
                                        pos=sipPt,
                                        mat=np.eye(3).flatten(),
                                        rgba=np.array([1, 0, 0, 0.3]))
                    idx_geom += 1
                    viewer.user_scn.ngeom = idx_geom
                    # Reset if the number of geometries hit the limit
                    if idx_geom > (viewer.user_scn.maxgeom - 50):
                        # Reset
                        idx_geom = 1

                # Reproduce MuJoCo Forces
                # fmj,fsd =mjforce(model,data)

                # Save data for plots [t,q,dq,r_com,dr_com,oL,oR,r_cop,fcl,fcr,tau,I*dq,WD]
                self.updateTrajData(model, data)
                DesData.append([data.time, self.qdes.copy(), self.dqdes.copy(), np.array([self.oCMx(data.time),self.oCMy(data.time),self.oCMz(data.time)]), np.empty(0), np.array([self.oLx(data.time),self.oLy(data.time),self.oLz(data.time)]),
                               np.array([self.oRx(data.time),self.oRy(data.time),self.oRz(data.time)]), self.ocp_des.copy()])
                ActData.append([data.time, self.q.copy(), self.dq.copy(), self.r_com.copy(), self.v_com.copy(), self.o_left.copy(),
                               self.o_right.copy(), r_cop.copy(), self.fcl.copy(), self.fcr.copy(), self.tau_PID.copy(),
                               self.Iwb @ data.qvel, self.WD.copy(), self.WD/(self.m*9.81), self.k_L])

                # Update scene and render
                viewer.sync()
                if saveVid == True:
                    # Set the lookat point to the humanoid's center of mass.
                    camera.lookat = self.r_com
                    renderer.update_scene(data, camera, scene_option)
                    # initialize the geom, here is a ball, if you want the label only, just change it to the mujoco.mjtGeom.mjGEOM_LABEL
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.0001, 0.0001, 0.0001]),  # label_size
                        pos=self.r_com + 0.5*self.r_com[2]*np.array([2.0, 0.0, 1.0]),  # label position
                        mat=np.eye(3).flatten(),  # label orientation
                        rgba=np.array([1, 0, 0, 1])  # red for the sphere
                    )
                    # add label
                    geom.label = "0.25 x real-time"
                    # add geom into scene
                    renderer.scene.ngeom += 1
                    pixels = renderer.render()
                    frames.append(pixels)

                time_until_next_step = 1 / simfreq - (time.time() - clock_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                print(data.time)
                # time.sleep(500)

        if saveVid==True:
            # Convert frames to a MoviePy clip and save
            clip = ImageSequenceClip(frames, fps=framerate)
            # saving video clip as gif or mp4
            # clip.write_gif("simulation_video.gif")
            clip.write_videofile("simulation_video.mp4")

        #End of simulation
        return DesData, ActData
    

    def paramidentify(self,model0,data0,trn,endsim):
        # Estimate parameters of Mj model
        def fun(x):
            d0 = x[0]
            dwidth = x[1]
            width = x[2]
            midpt = x[3]
            p = x[4]
            zeta=x[5]
            dampratio=zeta
            dmean=(d0+dwidth)/2
            deln = width * 1/trn.cntnocp
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
            model1=deepcopy(model0)
            model1.geom_solimp[trn.cntgeomid] = np.array([d0,dwidth, width, midpt, p])
            model1.geom_solref[trn.cntgeomid] = np.array([-stiffness, -damping])
            data1=deepcopy(data0)
            f=[]
            fval=0
            i=0
            while data1.time<endsim and i<len(trn.efc_f):
                #Inverse dyn
                # print(i)
                data1.qpos=trn.q[i]
                data1.qvel = trn.dq[i]
                data1.qacc = trn.ddq[i]
                #data.ctrl=self.controller(model,data).copy()
                # data.ctrl=0*data.ctrl #Zero Torque
                mujoco.mj_inverse(model1, data1)  # inverse dynamics
                #print(data.time)
                f.append(data1.efc_force)
                if len(trn.efc_f[i])==len(f[i]):
                    #print(trn.efc_f[i],len(trn.efc_f[i]),f[i],len(f[i]))
                    fval +=np.linalg.norm(np.array(trn.efc_f[i])-np.array(f[i]))
                i=i+1
            #print('len(trn.f),len(f):',len(trn.efc_f),len(f))
            #if len(f)==len(trn.f):
            #fval=np.linalg.norm(np.concatenate(trn.efc_f)-np.concatenate(f))
            #else:
            #    fval=np.linalg.norm(np.array(trn.efc_f))
            print('fval:',fval,'x:',x)
            return fval
        
        st_time=time.time()
        x0=np.append([0.9,0.95,0.001,0.5,2],1.0)
        bnds = ((0, 0.99), (0.1, 0.99), (0.0001, 0.02), (0.0, 0.99), (0, 5), (1e-5,1000))
        # Gradient-based method
        xsol=minimize(fun,x0,bounds=bnds)
        print('Time taken to minimize:',time.time()-st_time)
        x=xsol.x

        print('efc_force-f=',fun(x))
        dmean=(x[0]+x[1])/2
        deln = x[2] * 1/trn.cntnocp
        xmean=deln/2
        stiffness = (9.81*(1-dmean)*x[1]*x[1])/(xmean*dmean*dmean)#9.81 * (1 - x[1]) / x[2]
        #wn = np.sqrt(stiffness)
        timeconst =1/(x[5]*np.sqrt(stiffness)) #1 / (zeta * wn) #0.02 default
        damping = 2/timeconst #2 * x[5] * wn  #
        trn.cntsolimp=x[0:5].copy()
        trn.cntsolref=np.append(-stiffness,-damping) #x[5:7].copy()

        return trn
    
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
                print('Des vs Act Deformation is:',(model.geom_solimp[0][2]/2),self.DepthvsForce(model,data,1))
                break
        return model

        
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

