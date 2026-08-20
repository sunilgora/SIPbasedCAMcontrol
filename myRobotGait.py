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
from myRobotEnv import myRobot
from myRobotSIP import humn2SIP

class myPlanner(myRobot):
    def __init__(self,humn):
        #Import humn class variables
        self.__dict__ = humn.__dict__.copy()

        # Run walking pattern generator
        if self.SIPwalk==True:
            # SIP to Kondo traj
            self.sip2humn(1/100,self.simend,self.trn,self.model,self.data,0)

        else:
            if self.MPC_LIPM==True:
                #LIPM MPC to Kondo traj
                self.mpc2humn(1/1000,self.simend,self.trn,50,self.step_time,self.step_len,1)
            else:
                #LIPM to Kondo traj
                self.lipm2humn(1/1000,self.simend,50,0)

        print('!!!!!!! Walking pattern is generated/saved')


    # Cartesian trajectory of Humanoid Robot from SIP traj.
    def sip2humn(self,simfreq,simend,trn,model,data,vis):
        # SIP motion
        # Kondo to SIP model

        # Parameters of SIP
        sip=humn2SIP(self,trn,model, data)

        # Generate SIP walking pattern *** for one cycle i.e. SSP+DSP
        sipdata,ftplac = sip.siptraj(simend,1/simfreq,vis) #COM model
        # sipdata,ftplac = sip.ASIPtraj(simend,1/simfreq,vis) #COM+FW model
        
        # Or load sipdata,ftplac:
        # with open('siptraj.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        #     sipdata, ftplac = pickle.load(f)

        i = 0
        ti = sipdata[i][0]
        dt=sipdata[i+1][0]-sipdata[i][0]
        tcyc = []
        ncyc = 0
        oLeft=self.o_left.copy()
        oRight=self.o_right.copy()
        oCM=self.r_com.copy()
        Stlr=self.Stlr.copy()
        spno=self.spno
        self.tCMtraj = np.array([self.ti])
        self.oCMtraj = np.array([oCM])
        self.tLtraj = np.array([self.ti])
        self.oLtraj = np.array([oLeft])
        self.tRtraj = np.array([self.ti])
        self.oRtraj = np.array([oRight])
        self.FSPtraj = np.empty([3,1])
        if Stlr[0] == 1:  # left foot is stance
            self.oCPtraj = np.array([self.o_left]) #COP
        else:
            self.oCPtraj = np.array([self.o_right])

        for item in ftplac:
            tf = item[0]
            tcyc.append(tf)
            tcm = np.empty((0))
            ocm = np.empty((0, 3))
            oct = np.empty((0, 3))
            while ti < tf:
                ti = sipdata[i][0]
                tcm = np.append(tcm, np.array([ti]), axis=0)
                ocm = np.append(ocm, np.array([sipdata[i][1][0:3]]), axis=0)
                if spno == 2: #DSP
                    oct = np.append(oct,np.array([sipdata[i][1][0:3]]) + (np.array([sipdata[i][1][0:3]]) - np.array([sipdata[i][3]])) * abs(0 - sipdata[i][1][2]) / (sipdata[i][3][2] - sipdata[i][1][2]), axis=0) #ocm + (ocm - oct) * abs(0 - ocm[2]) / (oct[2] - ocm[2])
                else:
                    oct = np.append(oct, np.array([sipdata[i][3]]), axis=0)

                i = i + 1
            self.updateGait(tcm, ocm, oct, oLeft, oRight, ncyc, Stlr, spno, ftplac)
            ncyc = ncyc + 1
            if spno==2:
                Stlr = np.array([1, 1]) - Stlr
            spno = 3 - spno
            oLeft = self.oLtraj[-1]
            oRight = self.oRtraj[-1]

        self.oCMx = CubicSpline(self.tCMtraj, self.oCMtraj[:, 0])
        self.oCMy = CubicSpline(self.tCMtraj, self.oCMtraj[:, 1])
        self.oCMz = CubicSpline(self.tCMtraj, self.oCMtraj[:, 2])
        # oCMi=np.array([oCMx,oCMy,oCMz])
        self.oLx = CubicSpline(self.tLtraj, self.oLtraj[:, 0], bc_type='clamped')
        self.oLy = CubicSpline(self.tLtraj, self.oLtraj[:, 1], bc_type='clamped')
        self.oLz = CubicSpline(self.tLtraj, self.oLtraj[:, 2], bc_type='clamped')
        # oLi=np.array([oLx,oLy,oLz])
        self.oRx = CubicSpline(self.tRtraj, self.oRtraj[:, 0], bc_type='clamped')
        self.oRy = CubicSpline(self.tRtraj, self.oRtraj[:, 1], bc_type='clamped')
        self.oRz = CubicSpline(self.tRtraj, self.oRtraj[:, 2], bc_type='clamped')
        # oRi=np.array([oRx,oRy,oRz])
        self.oCPx = CubicSpline(self.tCMtraj, self.oCPtraj[:, 0])
        self.oCPy = CubicSpline(self.tCMtraj, self.oCPtraj[:, 1])
        self.oCPz = CubicSpline(self.tCMtraj, self.oCPtraj[:, 2])
        # return oCMx, oCMy, oCMz, oLx, oLy, oLz, oRx, oRy, oRz


    def lipm2humn(self,dt,Tf,sspbydsp,vis):
        i = 0
        ti = dt#sipdata[i][0]
        t0=0
        # dt=sipdata[i+1][0]-sipdata[i][0]
        tcyc = []
        ncyc = 0
        oLeft=self.o_left.copy()
        oRight=self.o_right.copy()
        oCM=self.r_com.copy()
        Stlr=self.Stlr.copy()
        spno=self.spno
        self.tCMtraj = np.array([self.ti])
        self.oCMtraj = np.array([oCM])
        self.tLtraj = np.array([self.ti])
        self.oLtraj = np.array([oLeft])
        self.tRtraj = np.array([self.ti])
        self.oRtraj = np.array([oRight])
        if Stlr[0] == 1:  # left foot is stance
            self.oCPtraj = np.array([self.o_left]) #COP
            rct = self.o_left
        else:
            self.oCPtraj = np.array([self.o_right])
            rct = self.o_right

        # LIPM motion
        r1=rct.copy()
        rcm0=self.r_com.copy()
        drcm0=self.v_com.copy()
        tcm = np.empty((0))
        ocm = np.empty((0, 3))
        oct = np.empty((0, 3))
        ftplac=[]
        for ti in np.arange(dt,2*Tf,dt):
            if spno==1: #SSP
                Ts = np.sqrt(abs(rcm0[2] - rct[2]) / 9.81)
                # x(t)=1/2*(x(0)+Ts xdot(0)) * np.exp(t/Ts) + 1/2*(x(0)-Ts xdot(0)) * np.exp(-t/Ts)
                rcm=rct+1/2*(rcm0-rct + Ts * drcm0) * np.exp((ti-t0)/Ts) + 1/2*(rcm0-rct - Ts * drcm0) * np.exp(-(ti-t0)/Ts)
                drcm=1/2*(rcm0-rct + Ts * drcm0) * (1/Ts) * np.exp((ti-t0)/Ts) - 1/2*(rcm0-rct - Ts * drcm0) *(1/Ts)* np.exp(-(ti-t0)/Ts)
            else: #DSP
                # x(t) = x(0)*cos(t/Td) + xdot(0)* Td * sin(t/Td)
                Td=np.sqrt(abs(rcm0[2] - rct[2]) / 9.81)
                rcm = rct + (rcm0-rct) * np.cos((ti-t0)/Td) + drcm0 * Td * np.sin((ti-t0) / Td)
                drcm= -(rcm0-rct) *(1/Td)* np.sin((ti-t0)/Td) + drcm0 * 1 * np.cos((ti-t0) / Td)
            rcm[2]=rcm0[2]
            drcm[2]=0
            tcm = np.append(tcm, np.array([ti]), axis=0)
            ocm = np.append(ocm, np.array([rcm]), axis=0)
            oct = np.append(oct, np.array([rct]), axis=0)
            if spno==1:
                rcp=drcm*Ts
                r2=rcm+(1/sspbydsp)*np.linalg.norm(rcm-rct)*drcm/np.linalg.norm(drcm)
                r2[2]=rcm[2]/sspbydsp
                r3=r1+2*(r2-r1)
                r3[2]=r1[2]

            # print(rcm,r1,r2,r3)
            if (spno==1)*(rcm[0]>rct[0])*( (abs(r3[0]-r1[0]) > self.xlimft) + (abs(r3[1]-r1[1]) > self.ylimft) ) or (spno==2)*(rcm[0]>rct[0])*(abs(rcm[0]-rct[0])>abs(rcm0[0]-rct[0]))*(abs(rcm[1]-rct[1])>abs(rcm0[1]-rct[1])):
                print(ti,oLeft,oRight)
                # Foot placement data
                ftplac.append([ti, r1, r2, r3])

                self.updateGait(tcm, ocm, oct, oLeft, oRight, ncyc, Stlr, spno, ftplac)  
                if spno==2:
                    Stlr = np.array([1, 1]) - Stlr
                t0 = ti
                tcyc.append(t0)
                ncyc = ncyc + 1
                spno = 3 - spno
                if spno==2:
                    rct=r2.copy()
                else:
                    rct = r3.copy()
                    r1 = r3.copy()
                rcm0 = rcm.copy()
                drcm0 = drcm.copy() #(ocm[-1,:] - ocm[-2]) / dt

                # plt.figure(18)
                # # print(len(tcm[0:-1]),(np.diff([sublist[1] for sublist in ocm])))
                # plt.plot(tcm, (np.array([sublist[0] for sublist in ocm])))
                # print(oct[-1])
                # plt.plot(tcm[-1],oct[-1][0],'*r')
                # plt.plot(tcm, (np.array([sublist[1] for sublist in ocm])))
                # plt.plot(tcm[-1],oct[-1][1],'*b')
                # plt.figure(19)
                # # print(len(tcm[0:-1]),(np.diff([sublist[1] for sublist in ocm])))
                # plt.plot(tcm[0:-1],(np.diff([sublist[0] for sublist in ocm])/dt))
                # plt.plot(tcm[0:-1],(np.diff([sublist[1] for sublist in ocm])/dt))


                oLeft = self.oLtraj[-1]
                oRight = self.oRtraj[-1]
                tcm = np.empty((0))
                ocm = np.empty((0, 3))
                oct = np.empty((0, 3))
                if ti >= Tf:
                    break
            if vis==1 and (ti%0.01)<dt:
                # print(ti) # ,drcm,rcm,r2)
                plt.figure(10)
                # plt.plot(ctime+data.time,qcp[2],'.r')
                plt.plot(ti,drcm[0],'*r',ti,drcm[1],'*g',ti,drcm[2],'*b')
                plt.xlabel('Time')
                plt.ylabel('dq/dt')
                plt.pause(0.001)
        # plt.show()
        # print(self.tCMtraj)
        self.oCMx = CubicSpline(self.tCMtraj, self.oCMtraj[:, 0])
        self.oCMy = CubicSpline(self.tCMtraj, self.oCMtraj[:, 1])
        self.oCMz = CubicSpline(self.tCMtraj, self.oCMtraj[:, 2])
        # oCMi=np.array([oCMx,oCMy,oCMz])
        self.oLx = CubicSpline(self.tLtraj, self.oLtraj[:, 0], bc_type='clamped')
        self.oLy = CubicSpline(self.tLtraj, self.oLtraj[:, 1], bc_type='clamped')
        self.oLz = CubicSpline(self.tLtraj, self.oLtraj[:, 2], bc_type='clamped')
        # oLi=np.array([oLx,oLy,oLz])
        self.oRx = CubicSpline(self.tRtraj, self.oRtraj[:, 0], bc_type='clamped')
        self.oRy = CubicSpline(self.tRtraj, self.oRtraj[:, 1], bc_type='clamped')
        self.oRz = CubicSpline(self.tRtraj, self.oRtraj[:, 2], bc_type='clamped')
        # oRi=np.array([oRx,oRy,oRz])
        self.oCPx = CubicSpline(self.tCMtraj, self.oCPtraj[:, 0])
        self.oCPy = CubicSpline(self.tCMtraj, self.oCPtraj[:, 1])
        self.oCPz = CubicSpline(self.tCMtraj, self.oCPtraj[:, 2])
        # return oCMx, oCMy, oCMz, oLx, oLy, oLz, oRx, oRy, oRz

    def mpc2humn(self,dt,Tf,trn,sspbydsp,step_time,step_len,vis):
        i = 0
        ti = dt#sipdata[i][0]
        t0=0
        # dt=sipdata[i+1][0]-sipdata[i][0]
        tcyc = []
        ncyc = 0
        oLeft=self.o_left.copy()
        oRight=self.o_right.copy()
        oCM=self.r_com.copy()
        Stlr=self.Stlr.copy()
        spno=2 #self.spno #start with DSP in rest
        self.tCMtraj = np.array([self.ti])
        self.oCMtraj = np.array([oCM])
        self.tLtraj = np.array([self.ti])
        self.oLtraj = np.array([oLeft])
        self.tRtraj = np.array([self.ti])
        self.oRtraj = np.array([oRight])
        if Stlr[0] == 1:  # left foot is stance
            self.oCPtraj = np.array([self.o_left]) #COP
            rct = self.o_left
        else:
            self.oCPtraj = np.array([self.o_right])
            rct = self.o_right

        # LIPM motion
        r1=rct.copy()
        rcm0=self.r_com.copy()
        rct0=rct.copy()
        drcm0=self.v_com.copy()
        tcm = np.empty((0))
        ocm = np.empty((0, 3))
        oct = np.empty((0, 3))
        ftplac=[]

        #MPC control
        #Kajita's 2003 paper on LIPM preview control using Katayama et al. 1985 LQI control
        # LIPM parameters
        h=rcm0[2]-rct[2] #height of COM from ZMP
        g=9.81 #gravity

        Ts = np.sqrt(h / g) #Tc = np.sqrt(h/g)
        # step_time=np.around(Ts,1) #cycle time
        # dt = 0.005 #5ms
        N = 1000  # Preview horizon

        # State-space model for LIPM (discretized)
        A = np.array([[1, dt, dt**2/2],
                    [0, 1, dt],
                    [0, 0, 1]])
        B = np.array([[dt**3/6],
                    [dt**2/2],
                    [dt]])
        C = np.array([[1, 0, -h/g]])

        # Cost weights
        Qe = 1.0   # ZMP error weight
        Qx = np.zeros((3, 3))  # State weight        
        R = 1e-6   # Control input weight

        # Augmented system for preview control
        nx = A.shape[0]
        G = np.vstack((C, np.zeros((1, nx))))
        A_aug = np.vstack([
            np.hstack([np.eye(1), C @ A]),
            np.hstack([np.zeros((nx, 1)), A])
        ])
        B_aug = np.vstack([C @ B, B])

        # Riccati equation for augmented system
        Q = np.zeros((nx+1, nx+1))
        Q[0, 0] = Qe
        Q[1:, 1:] = Qx
        P = solve_discrete_are(A_aug, B_aug, Q, R)
        #print("Algebric Riccati Equation solution P:\n", P)

        # Compute feedback and preview gains
        K = np.linalg.inv(B_aug.T @ P @ B_aug + R) @ (B_aug.T @ P @ A_aug)
        Gi = K[0, 0]
        Gx = K[0, 1:]

        # Compute preview gains
        Gd = np.zeros(N)
        AcBK = A_aug - B_aug @ K
        X = -AcBK.T @ P @ np.array([[1], [0], [0], [0]])
        for i in range(N):
            Gd[i] = (np.linalg.inv(B_aug.T @ P @ B_aug + R) @ (B_aug.T @ X)).item()
            X = AcBK.T @ X

        # Plot preview gains
        # plt.figure(figsize=(8,3))
        # plt.plot(np.arange(1, N+1)*dt, Gd, marker='o')
        # plt.xlabel('Preview Time (s)')
        # plt.ylabel('Preview Gain $G_d$')
        # plt.title('Preview Gains vs. Preview Time (Kajita 2003)')
        # plt.grid()
        # plt.tight_layout()
        # plt.show()

        # --- Generate a sample ZMP (footstep) reference trajectory ---
        # step_time = 1 #cycle time
        num_steps = np.ceil((Tf+N*dt)/step_time).astype(int) #10
        zmp_ref_x = []
        zmp_ref_y = []
        zmp_ref_z = []
        com_z = [] #rcm0[2]
        zmp_crt=np.zeros(3)
        i=0
        #Current ZMP position
        zmp_crt[0]= rct0[0]+(i)*step_len #x_val
        zmp_crt[1]=0 #y_val
        zmp_crt[2]=0
        #Next ZMP position
        zmp_nxt=np.zeros(3)
        for i in range(num_steps):
            if (i+1) % 2 == 0:
                zmp_nxt[0]= rct0[0]+(i)*step_len #x_val
                zmp_nxt[1]= self.o_right[1] #y_step
                zmp_nxt[2]=0
            else:
                zmp_nxt[0]= rct0[0]+(i)*step_len #x_val
                zmp_nxt[1]=self.o_left[1] #-y_step
                zmp_nxt[2]=0

            # Check step height
            trn.cntplane(zmp_nxt, 1)
            zmp_nxt[2]=trn.cntpos[2]
            zmp_ref_x += [zmp_crt[0]]*int(step_time/dt)
            zmp_ref_y += [zmp_crt[1]]*int(step_time/dt)
            zmp_ref_z += [zmp_crt[2]]*int(step_time/dt)
            # print(zmp_ref_x)
            if i==0:
                com_z += [rcm0[2]+(zmp_nxt[2]-zmp_crt[2])*i/int(step_time/dt) for i in range(int(step_time/dt))] 
            else:
                # print("com_z last:", com_z[-1])
                com_z += [com_z[-1]+(zmp_nxt[2]-zmp_crt[2])*i/int(step_time/dt) for i in range(int(step_time/dt))]
            zmp_crt=zmp_nxt.copy()

        zmp_ref_x = np.array(zmp_ref_x)
        zmp_ref_y = np.array(zmp_ref_y)
        zmp_ref_z = np.array(zmp_ref_z)
        print("ZMP ref len:", len(zmp_ref_x), "Preview steps:", N)

        # --- Preview control simulation ---
        x = np.array([[rcm0[0]], [drcm0[0]], [0.0]])  # [CoM pos, vel, acc]
        com_x = []
        zmp_x = []
        e_sum = 0.0

        for k in range(len(zmp_ref_x)-N):
            # Output (current ZMP)
            p = (C @ x)[0, 0]
            # Error integration
            e = p - zmp_ref_x[k]
            e_sum += e

            # Preview control law (Kajita 2003)
            preview_sum = 0.0
            for j in range(N):
                if (k + j + 1) < len(zmp_ref_x):
                    preview_sum += float(Gd[j]) * zmp_ref_x[k + j + 1]
                else:
                    preview_sum += float(Gd[j]) * zmp_ref_x[-1]
            u = -Gi * e_sum - Gx @ x.flatten() - preview_sum

            # State update
            x = A @ x + B * u

            com_x.append(float(x[0, 0]))
            zmp_x.append(p)


        # --- Generate a sample ZMP (footstep) reference trajectory for y ---
        # For example, alternate left/right footsteps
        # y_step = 0.1
        # zmp_ref_y = []

        # --- Preview control simulation for y direction ---
        y = np.array([[rcm0[1]], [drcm0[1]], [0.0]])  # [CoM pos, vel, acc] in y
        com_y = []
        zmp_y = []
        e_sum_y = 0.0

        for k in range(len(zmp_ref_y)-N):
            # Output (current ZMP)
            p_y = (C @ y)[0, 0]
            # Error integration
            e_y = p_y - zmp_ref_y[k]
            e_sum_y += e_y

            # Preview control law (Kajita 2003) for y
            preview_sum_y = 0.0
            for j in range(N):
                if (k + j + 1) < len(zmp_ref_y):
                    preview_sum_y += float(Gd[j]) * zmp_ref_y[k + j + 1]
                else:
                    preview_sum_y += float(Gd[j]) * zmp_ref_y[-1]
            u_y = -Gi * e_sum_y - Gx @ y.flatten() - preview_sum_y

            # State update
            y = A @ y + B * u_y

            com_y.append(float(y[0, 0]))
            zmp_y.append(p_y)        
        if vis==1:
            plt.figure(figsize=(10,6))
            time_axis = np.arange(0, len(com_x)*dt, dt)
            plt.subplot(3,1,1)
            plt.plot(time_axis, com_x, label='CoM x')
            plt.plot(time_axis, zmp_x, label='ZMP x')
            plt.plot(time_axis, zmp_ref_x[:len(zmp_x)], '--', label='ZMP ref x')
            plt.xlabel('Time (s)')
            plt.ylabel('X Position (m)')
            plt.xlim(0, max(time_axis))
            plt.title('LIPM Preview Control in X Direction')
            plt.legend()
            plt.grid()

            plt.subplot(3,1,2)
            plt.plot(time_axis, com_y, label='CoM y')
            plt.plot(time_axis, zmp_y, label='ZMP y')
            plt.plot(time_axis, zmp_ref_y[:len(zmp_y)], '--', label='ZMP ref y')
            plt.xlabel('Time (s)')
            plt.ylabel('Y Position (m)')
            plt.xlim(0, max(time_axis))
            plt.title('LIPM Preview Control in Y Direction')
            plt.legend()
            plt.grid()

            plt.subplot(3,1,3)
            plt.plot(time_axis, com_z[:len(time_axis)], label='CoM z')
            plt.xlabel('Time (s)')
            plt.ylabel('Z Position (m)')
            plt.xlim(0, max(time_axis))
            plt.title('CoM Height Trajectory')
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.pause(2)
            # print(rcm0,com_x[0],com_y[0])
            # plt.show()

        # Generate Gait starting with DSP
        #spno=2
        i=0
        for ti in np.arange(dt,2*Tf,dt):
            if i>=len(com_x):
                print(ti,Tf,"End of ref traj")
                break
            rcm=np.array([com_x[i],com_y[i],com_z[i]])
            rct=np.array([zmp_ref_x[i],zmp_ref_y[i],rct[2]])
            tcm = np.append(tcm, np.array([ti]), axis=0)
            ocm = np.append(ocm, np.array([rcm]), axis=0)
            oct = np.append(oct, np.array([rct]), axis=0)
            #     rcp=drcm*Ts
            #     r2=rcm+(1/sspbydsp)*np.linalg.norm(rcm-rct)*drcm/np.linalg.norm(drcm)
            #     r2[2]=rcm[2]/sspbydsp
            #     r3=r1+2*(r2-r1)
            #     r3[2]=r1[2]

            # print(rcm,r1,r2,r3)
            if (i+1)%int(step_time/dt)==0:# or ti%step_time<dt: #time to change foot
                print(ncyc,ti)
                # Foot placement data
                if spno==2: #DSP no change in foot placement
                    r1=self.o_left.copy()
                    r3=self.o_right.copy()
                else:
                    r3=np.array([zmp_ref_x[i+1],zmp_ref_y[i+1],rct[2]])
                    trn.cntplane(r3, 1)
                    r3[2]=trn.cntpos[2]
                    print(r1,r3)

                ftplac.append([ti, r1, rct, r3])
                #print(ti,ti%step_time,ftplac[-1])

                self.updateGait(tcm, ocm, oct, oLeft, oRight, ncyc, Stlr, spno, ftplac)
                if spno==1:
                    Stlr = np.array([1, 1]) - Stlr 
                    # rct0 = rct.copy()      
                    rct[2]=r3[2]
                    r1 = r3.copy()          
                else: #No DSP in MPC after first DSP
                    spno=1
                t0 = ti
                tcyc.append(t0)
                ncyc = ncyc + 1
                #spno = 3 - spno
                # if spno==2:
                #     rct=r2.copy()
                # else:
                #     rct = r3.copy()
                #     r1 = r3.copy()
                rcm0 = rcm.copy()
                # drcm0 = drcm.copy() #(ocm[-1,:] - ocm[-2]) / dt

                # plt.figure(18)
                # # print(len(tcm[0:-1]),(np.diff([sublist[1] for sublist in ocm])))
                # plt.plot(tcm, (np.array([sublist[0] for sublist in ocm])))
                # print(oct[-1])
                # plt.plot(tcm[-1],oct[-1][0],'*r')
                # plt.plot(tcm, (np.array([sublist[1] for sublist in ocm])))
                # plt.plot(tcm[-1],oct[-1][1],'*b')
                # plt.figure(19)
                # # print(len(tcm[0:-1]),(np.diff([sublist[1] for sublist in ocm])))
                # plt.plot(tcm[0:-1],(np.diff([sublist[0] for sublist in ocm])/dt))
                # plt.plot(tcm[0:-1],(np.diff([sublist[1] for sublist in ocm])/dt))


                oLeft = self.oLtraj[-1]
                oRight = self.oRtraj[-1]
                tcm = np.empty((0))
                ocm = np.empty((0, 3))
                oct = np.empty((0, 3))
                if ti >= Tf:
                    break
            i=i+1
            # if vis==1 and (ti%0.01)<dt:
            #     print(ti) # ,drcm,rcm,r2)
            #     plt.figure(10)
            #     # plt.plot(ctime+data.time,qcp[2],'.r')
            #     # plt.plot(ti,drcm[0],'*r',ti,drcm[1],'*g',ti,drcm[2],'*b')
            #     plt.xlabel('Time')
            #     plt.ylabel('dq/dt')
            #     plt.pause(0.001)
        # plt.show()
        # print(self.tCMtraj)
        self.oCMx = CubicSpline(self.tCMtraj, self.oCMtraj[:, 0])
        self.oCMy = CubicSpline(self.tCMtraj, self.oCMtraj[:, 1])
        self.oCMz = CubicSpline(self.tCMtraj, self.oCMtraj[:, 2])
        # oCMi=np.array([oCMx,oCMy,oCMz])
        self.oLx = CubicSpline(self.tLtraj, self.oLtraj[:, 0], bc_type='clamped')
        self.oLy = CubicSpline(self.tLtraj, self.oLtraj[:, 1], bc_type='clamped')
        self.oLz = CubicSpline(self.tLtraj, self.oLtraj[:, 2], bc_type='clamped')
        # oLi=np.array([oLx,oLy,oLz])
        self.oRx = CubicSpline(self.tRtraj, self.oRtraj[:, 0], bc_type='clamped')
        self.oRy = CubicSpline(self.tRtraj, self.oRtraj[:, 1], bc_type='clamped')
        self.oRz = CubicSpline(self.tRtraj, self.oRtraj[:, 2], bc_type='clamped')
        # oRi=np.array([oRx,oRy,oRz])
        self.oCPx = CubicSpline(self.tCMtraj, self.oCPtraj[:, 0])
        self.oCPy = CubicSpline(self.tCMtraj, self.oCPtraj[:, 1])
        self.oCPz = CubicSpline(self.tCMtraj, self.oCPtraj[:, 2])
        # return oCMx, oCMy, oCMz, oLx, oLy, oLz, oRx, oRy, oRz

    # def LIPMmpc(self,):

    # Update gait from simplified model of one cycle
    def updateGait(self,tcm, ocm, oct, oLeft, oRight, ncyc, Stlr, spno, ftplac):
        self.tCMtraj = np.append(self.tCMtraj, tcm, axis=0)
        self.oCMtraj = np.append(self.oCMtraj, ocm, axis=0)
        self.oCPtraj = np.append(self.oCPtraj, oct, axis=0)
        if spno == 1:  # SSP
            if Stlr[0] == 1:  # left foot is stance
                if abs(oRight[2] - ftplac[ncyc][3][2])>self.zSw/2: #Increase Step height on stairs if needed
                    self.zSw=2*abs(oRight[2] - ftplac[ncyc][3][2])

                self.tLtraj = np.append(self.tLtraj, tcm, axis=0)
                self.oLtraj = np.append(self.oLtraj, oct, axis=0)
                self.tRtraj = np.append(self.tRtraj, np.array([0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), axis=0)
                self.oRtraj = np.append(
                    np.append(self.oRtraj, np.array([0.5 * (oRight + ftplac[ncyc][3]) + [0, 0, self.zSw]]), axis=0),
                    np.array([ftplac[ncyc][3]]), axis=0)
            else:
                if abs(oLeft[2] - ftplac[ncyc][3][2])>self.zSw/2: #increase step height if needed
                    self.zSw=2*abs(oLeft[2] - ftplac[ncyc][3][2])
                self.tRtraj = np.append(self.tRtraj, tcm, axis=0)
                self.oRtraj = np.append(self.oRtraj, oct, axis=0)
                self.tLtraj = np.append(self.tLtraj, np.array([0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), axis=0)
                self.oLtraj = np.append(
                    np.append(self.oLtraj, np.array([0.5 * (oLeft + ftplac[ncyc][3]) + [0, 0, self.zSw]]), axis=0),
                    np.array([ftplac[ncyc][3]]), axis=0)
        else:  # DSP
            Stlr = np.array([1, 1]) - Stlr
            self.tLtraj = np.append(self.tLtraj, tcm, axis=0)
            self.tRtraj = np.append(self.tRtraj, tcm, axis=0)
            if Stlr[0] == 1:  # left foot is stance
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[0], 0.5 * (oLeft[0] + ftplac[ncyc][3][0]), ftplac[ncyc][3][0]]),
                                    bc_type='clamped')
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[1], 0.5 * (oLeft[1] + ftplac[ncyc][3][1]), ftplac[ncyc][3][1]]),
                                    bc_type='clamped')
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[2], 0.5 * (oLeft[2] + ftplac[ncyc][3][2]), ftplac[ncyc][3][2]]),
                                    bc_type='clamped')
                self.oLtraj = np.append(self.oLtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[0], 0.5 * (oRight[0] + ftplac[ncyc][1][0]), ftplac[ncyc][1][0]]), bc_type='clamped')
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[1], 0.5 * (oRight[1] + ftplac[ncyc][1][1]), ftplac[ncyc][1][1]]), bc_type='clamped')
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[2], 0.5 * (oRight[2] + ftplac[ncyc][1][2]), ftplac[ncyc][1][2]]), bc_type='clamped')
                self.oRtraj = np.append(self.oRtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)
            else:
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[0], 0.5 * (oRight[0] + ftplac[ncyc][3][0]), ftplac[ncyc][3][0]]), bc_type='clamped')
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[1], 0.5 * (oRight[1] + ftplac[ncyc][3][1]), ftplac[ncyc][3][1]]), bc_type='clamped')
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[2], 0.5 * (oRight[2] + ftplac[ncyc][3][2]), ftplac[ncyc][3][2]]), bc_type='clamped')
                self.oRtraj = np.append(self.oRtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[0], 0.5 * (oLeft[0] + ftplac[ncyc][1][0]), ftplac[ncyc][1][0]]),
                                    bc_type='clamped')
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[1], 0.5 * (oLeft[1] + ftplac[ncyc][1][1]), ftplac[ncyc][1][1]]),
                                    bc_type='clamped')
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[2], 0.5 * (oLeft[2] + ftplac[ncyc][1][2]), ftplac[ncyc][1][2]]),
                                    bc_type='clamped')
                self.oLtraj = np.append(self.oLtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)

    
    # Joint angles from cartesian trajectories of Kondo humanoid robot
    def cart2joint(self,model, data0, ttraj, WLN, zeroAM):

        data = deepcopy(data0)
        qtraj = []
        AM_CM = []
        delt = ttraj[1] - ttraj[0]
        Lw = np.zeros(3)
        # self.tau = []
        rcom=data.subtree_com[0]
        gradH0=np.zeros([model.nv])            
        for ti in ttraj:
            mujoco.mj_fwdPosition(model, data);
            # Current joint angles
            q0 = self.data2q(data)
            drcom=(data.subtree_com[0]-rcom)/delt
            rcom = data.subtree_com[0].copy()
            # Desired traj
            ocm = np.array([self.oCMx(ti), self.oCMy(ti), self.oCMz(ti)])
            oLeft = np.array([self.oLx(ti), self.oLy(ti), self.oLz(ti)])
            oRight = np.array([self.oRx(ti), self.oRy(ti), self.oRz(ti)])
            qdes,gradH0 = self.numik(model, data, q0, delt, ocm, oLeft, oRight, self.ub_jnts, gradH0, WLN, self.k_ub, zeroAM)  # 0 - Upperbody locked and Nonzero ang. momentum about COM, 1 - ZAM abt COM using Arms swing
            data = self.q2data(data, qdes)
            qtraj.append(qdes)
            if ti%0.01<delt: print(ti)

            # Angular momentum matrix
            Iwb = np.zeros([3, model.nv])
            mujoco.mj_angmomMat(model, data, Iwb, 0)
            # Angular momentum
            AM_CM.append((Iwb @ (qdes-q0)/delt)) #tau is change in ang momentum
            # Lw = Iwb @ (qdes - q0) / delt
            # plt.figure(16)
            # plt.plot(drcom[0],AM_CM[-1][1],'*r')
            # plt.pause(0.0001)
        qi = []
        # dqi=[]
        self.CAMTraj=np.array(AM_CM)
        self.qTraj_des = np.append(self.qTraj_des, [qdes], axis=0)            
        self.AM_CMspl[0] = CubicSpline(ttraj, self.CAMTraj[:, 0], bc_type='clamped')
        self.AM_CMspl[1] = CubicSpline(ttraj, self.CAMTraj[:, 1], bc_type='clamped')
        self.AM_CMspl[2] = CubicSpline(ttraj, self.CAMTraj[:, 2], bc_type='clamped')

        self.qTraj_des = np.array(qtraj)
        # if len(ttraj) > 2:
        #     fig1 = plt.figure(1)
        #     fig2 = plt.figure(2)
        #     for i in np.arange(6, model.nv):
        #         plt.figure(1)
        #         plt.plot(ttraj, qtraj[:, i],label=f'th_{i}')
        #         plt.figure(2)
        #         plt.plot(ttraj, np.append(0, 1 / delt * np.diff(qtraj[:, i])),label=f'dth_{i}')
        #     plt.figure(1)
        #     plt.legend()
        #     fig1.savefig('qtraj.jpeg')
        #     plt.close(fig1)
        #     plt.figure(2)
        #     plt.legend()
        #     fig2.savefig('dqtraj.jpeg')
        #     plt.close(fig2)
        #     # Saving the data:
        #     with open('qtraj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        #         pickle.dump([ttraj, qtraj], f)

        #Plot position vs velocity in subplot
        # numrow=4
        # numcol=3
        # #Find joint velocities by numerical differentiation
        # dqtraj = np.zeros(qtraj.shape)
        # for i in range(model.nv):
        #     dqtraj[:, i] = np.append(data0.qvel[i], 1 / delt * np.diff(qtraj[:, i]))

        # fig2, ax2 = plt.subplots(nrows=numrow, ncols=numcol)
        # for i in range(12):
        #     ax2[i // numcol, i % numcol].plot(qtraj[:,i],dqtraj[:,i]) #, col[i % 6])
        #     ax2[i // numcol, i % numcol].set_title(f'Joint {i}')
        #     ax2[i // numcol, i % numcol].set_xlabel('Position (rad    )')
        #     ax2[i // numcol, i % numcol].set_ylabel('Velocity (rad/s)')
        #     ax2[i // numcol, i % numcol].grid(visible=None, which='major', axis='both')
        # # fig2.tight_layout()

        # plt.pause(1)
        # plt.show()

        return self.qTraj_des
    
    def optGait(self,ttraj): #Solve nonlinear optimization for gait generation        
        qtraj = []
        #Reference ZMP:
        Tf=ttraj[-1]
        dt=ttraj[1]-ttraj[0]
        step_time=self.step_len/self.vel
        N=1000
        rct0=self.o_left.copy()
        rcm0=self.r_com.copy()
        num_steps = np.ceil((Tf+N*dt)/step_time).astype(int) #10
        zmp_ref_x = []
        zmp_ref_y = []
        zmp_ref_z = []
        com_z = [] #rcm0[2]
        zmp_crt=np.zeros(3)
        i=0
        #Current ZMP position
        zmp_crt[0]= rct0[0]+(i)*self.step_len #x_val
        zmp_crt[1]=0 #y_val
        zmp_crt[2]=0
        #Next ZMP position
        zmp_nxt=np.zeros(3)
        for i in range(num_steps):
            if (i+1) % 2 == 0:
                zmp_nxt[0]= rct0[0]+(i)*self.step_len #x_val
                zmp_nxt[1]= self.o_right[1] #y_step
                zmp_nxt[2]=0
            else:
                zmp_nxt[0]= rct0[0]+(i)*self.step_len #x_val
                zmp_nxt[1]=self.o_left[1] #-y_step
                zmp_nxt[2]=0

            # Check step height
            self.trn.cntplane(zmp_nxt, 1)
            zmp_nxt[2]=self.trn.cntpos[2]
            zmp_ref_x += [zmp_crt[0]]*int(step_time/dt)
            zmp_ref_y += [zmp_crt[1]]*int(step_time/dt)
            zmp_ref_z += [zmp_crt[2]]*int(step_time/dt)
            # print(zmp_ref_x)
            if i==0:
                com_z += [rcm0[2]+(zmp_nxt[2]-zmp_crt[2])*i/int(step_time/dt) for i in range(int(step_time/dt))] 
            else:
                # print("com_z last:", com_z[-1])
                com_z += [com_z[-1]+(zmp_nxt[2]-zmp_crt[2])*i/int(step_time/dt) for i in range(int(step_time/dt))]
            zmp_crt=zmp_nxt.copy()

        self.zmp_ref_x = np.array(zmp_ref_x)
        self.zmp_ref_y = np.array(zmp_ref_y)
        self.zmp_ref_z = np.array(zmp_ref_z)
        self.com_z = np.array(com_z)
        print("ZMP ref len:", len(zmp_ref_x), "Preview steps:", N)

        q0=self.q0
        for t in ttraj:            
            sol = minimize(self.gaitCost, q0, method='SLSQP', options={'ftol': 1e-6, 'disp': False, 'maxiter': 100})
            q0 = sol.x
            qtraj.append(q0)

    def gaitCost(self, q):
        #Unpack gait variables
        # q = gaitVars
        cost = 0.0
        #Cost terms
        #Minimize joint torques and ZMP tracking error
        self.q2data(self.data, q)
        mujoco.mj_inverse(self.model, self.data)
        tau = self.data.qfrc_inverse
        #Find current ZMP tracking error        
        cost += np.sum(tau**2) + 1000.0 * ((self.data.subtree_com[0][0] - self.zmp_ref_x)**2 + (self.data.subtree_com[0][1] - self.zmp_ref_y)**2)
        return cost
        
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

