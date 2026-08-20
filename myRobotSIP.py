# This file contains functions used for footstep planning of Kondo khr3hv using SIP model
# Author : Sunil Gora, Shakti S. Gupta and Ashish Dutta

import numpy as np
import mujoco
import matplotlib.pyplot as plt
import os,time
import xml.etree.ElementTree as ET

# Kondo to SIP model
# Parameters of SIP
class humn2SIP:
    def __init__(self, humn,trn,model,data):
        self.m =mujoco.mj_getTotalmass(model) #1.379
        self.I = humn.I #1/1000*np.eye(3,3) #np.zeros((3,3))
        # self.I= data.crb[0] #1/1000*np.eye(3,3)
        # print('Inertia of Humanoid about COM:',self.I)
        self.qcm=data.subtree_com[0].copy() #np.array([-0.05, -0.025, 0.25])
        self.dqcm=data.subtree_linvel[0].copy() #np.array([0.39, 0.15, 0.0])
        self.Stlr=humn.Stlr.copy()
        self.plno=humn.plno
        self.trn=trn
        self.xlimft=humn.xlimft
        self.ylimft=humn.ylimft
        self.spno=humn.spno
        self.sspbydsp=humn.sspbydsp
        self.foot_size = humn.foot_size  # = np.array([length, width])
        self.vel=humn.vel
        self.Tsip=humn.Tsip
        self.zSw=humn.zSw
        self.AM_CMspl=humn.AM_CMspl
        self.En_des=humn.En_init

        if self.spno==1:
            if self.Stlr[0]==1:
                self.qcp=data.site('left_foot_site').xpos.copy()  # current Left foot position  #np.array([0.0, 0.0, 0.0])
                self.dqcp=0*self.dqcm.copy()
            else:
                self.qcp=data.site('right_foot_site').xpos.copy()  # current Left foot position  #np.array([0.0, 0.0, 0.0])
                self.dqcp=0*self.dqcm.copy()
        else:
            self.plno=1
            if self.Stlr[0]==1:
                # self.qcp=data.site('right_foot_site').xpos.copy()  # current Left foot position  #np.array([0.0, 0.0, 0.0])
                self.qcp=0.25*data.site('right_foot_site').xpos+0.75*data.site('left_foot_site').xpos
                self.qcp[2]=self.qcm[2]+1/humn.sspbydsp*(self.qcm[2]-self.qcp[2])
                self.qcp[1]=self.qcp[1]+self.foot_size[0]/2
                self.dqcp=0*self.dqcm.copy()
            else:
                # self.qcp=data.site('left_foot_site').xpos.copy()  # current Left foot position  #np.array([0.0, 0.0, 0.0])
                self.qcp=0.75*data.site('right_foot_site').xpos+0.25*data.site('left_foot_site').xpos
                self.qcp[2]=self.qcm[2]+1/humn.sspbydsp*(self.qcm[2]-self.qcp[2])
                self.qcp[1]=self.qcp[1]+self.foot_size[0]/10
                self.dqcp=0*self.dqcm.copy()

        self.l = np.linalg.norm(self.qcm - self.qcp)
        self.dth=data.qvel[3:6] #np.array([0,0,0])
        self.theulr = findeulr(self.qcm,self.qcp,self.l)
        self.qqt = euler2quat(self.theulr)

        self.rc = self.qcm
        self.r1 = self.qcp
        self.r2 = 0
        self.r3 = 0

    # Generate SIP trajectory given the initial pos and velocity of COM
    def siptraj(self, simend, simfreq, vis):
        # Parameters of SIP
        m=self.m
        I=self.I
        #qcm=self.qcm.copy()
        #dqcm=self.dqcm.copy()
        #qcp=self.qcp.copy()
        l = np.linalg.norm(self.qcm - self.qcp)
        theulr = findeulr(self.qcm, self.qcp, l)
        qqt = euler2quat(theulr)
        # dqcp = np.array([0, 0, 0])
        dth = np.matmul(np.linalg.pinv(l * np.array(
            [[0, np.cos(theulr[1]), 0],
            [-np.cos(theulr[0]) * np.cos(theulr[1]), np.sin(theulr[0]) * np.sin(theulr[1]), 0],
            [-np.sin(theulr[0]) * np.cos(theulr[1]), -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), self.dqcm - 0 * self.dqcm)

        # zpln = trn[plno].zpln
        # solref = trn[plno].solref
        # solimp = trn[plno].solimp
        # nocp = trn[plno].nocp

        self.trn.cntplane(self.qcp, self.spno)
        xml_str, r = modifysip(m, I, l, self.qcm, self.qcp, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp, self.trn.cntsolref, self.trn.cntsolimp)  # create new xml file from basic sip

        # MuJoCo data structures
        model = mujoco.MjModel.from_xml_string(xml_str)  # MuJoCo model
        data = mujoco.MjData(model)  # MuJoCo data
        cam = mujoco.MjvCamera()  # Abstract camera
        opt = mujoco.MjvOption()  # visualization options

        # Example on how to set camera configuration
        # cam.azimuth = 90
        # cam.elevation = -45
        # cam.distance = 2
        # cam.lookat = np.array([0.0, 0.0, 0])

        # print(DepthvsForce(model,data,0))

        ctime = 0  # total time of all sip motions
        if self.spno == 1:
            rc = self.qcm.copy()
            r1 = self.qcp.copy()
            r2 = 0*self.qcp
            r3 = 0*self.qcp
        else:
            rc = self.qcm.copy()
            r1 = self.qcm.copy()
            r1[2]=0
            r2 = self.qcp.copy()
            r3 = 2*self.qcp-self.qcm
            r3[0]=self.qcp[0]
            r3[1]=self.qcp[1]
            r3[2]=0
        print('qcm=',self.qcm,'qcp=',self.qcp,'l=',l)
        # Run walking pattern generation
        sipdata = []
        ftplac = []
        while ctime < simend:
            # data.contact.solref
            # print('dqcm =',self.dqcm)
            data, sipdata, tf, self.qcm, self.dqcm, rc, r1, r2, r3 = sipmotion(model, data, simend, simfreq, self.spno, m, l, self.qcm, qqt,
                                                                    self.qcp, self.dqcm, dth, rc, r1, r2, r3, self.sspbydsp, self.xlimft,
                                                                    self.ylimft, self.En_des, self.AM_CMspl, ctime,
                                                                    sipdata, vis)
            # try:
            #     # stiffness
            #     kn = data.contact.solimp[0][0] / ((data.contact.solimp[0][1] ** 2) * (data.contact.solref[0][0] ** 2) * (
            #             data.contact.solref[0][1] ** 2))
            #     print('stiffness of SIP',kn)  # stiffness
            #     # damping
            #     print('damping of SIP',2 / (data.contact.solimp[0][1] * data.contact.solref[0][0]))
            # except:
            #     print('stiffness/dampting error')

            # Increment time
            ctime = ctime + tf
            # Change SSP/DSP
            self.spno = 3 - self.spno
            if self.spno == 2: # DSP
                self.plno = 1  # change contact plane
                # print('Angular momentum abt r1', np.cross(self.qcm - self.qcp, m * self.dqcm))
                self.qcp = r2.copy()
                # print('foot placement position r3', r3)
                # Contact parameters
                #Switch leg support
                self.Stlr = np.array([1, 1]) - self.Stlr

                # deformation of contact point:
                #   dz_r1 =  self.trn.cntsolimp[2]/2
                #   print('Error in z deformation, i.e, dz_qcp=',dz_r1)
                # Swing foot pos at the start of DSP: r1
                # Swing foot pos at the end of DSP
                self.rsw = np.append(r1[0:2], self.trn.pos[self.trn.cntgeomid][2] + self.trn.size[self.trn.cntgeomid][2])
                # Stance foot pos at the start of DSP
                self.trn.cntplane(r3, self.spno)
                self.rst = np.append(r3[0:2], self.trn.pos[self.trn.cntgeomid][2] + self.trn.size[self.trn.cntgeomid][2])
                #Stance foot pos at the end of DSP: r3
                # Foot placement data
                ftplac.append([ctime, r1, r2, self.rst])
                self.trn.cntplane(self.qcp, self.spno)

                # plt.plot(self.qcp[0], self.qcp[1], 'bo')

            else: # SSP
                # print('Angular momentum abt r2', np.cross(self.qcm - self.qcp, m * self.dqcm))
                # Contact parameters
                if self.Stlr[0] == 1:
                    self.plno = 0  # change contact plane
                else:
                    self.plno = 2  # change contact plane
                self.qcp = r3.copy()  # Change contact point
                #self.qcp[2] = 0 - self.trn.cntsolimp[2]  # Contact point deformation
                # Foot placement data
                self.trn.cntplane(self.qcp, self.spno)
                #if same pendulum length in each SSP
                # self.qcp[2] = (self.qcm[2]-np.sqrt(np.linalg.norm(rc - r1)**2-np.linalg.norm(self.qcm[0:2]-r3[0:2])**2)) +  self.trn.cntpos[2] - self.trn.cntsolimp[2]/2 -r1[2]  # Contact point deformation
                #else Change pendulum length in each SSP based on terrain height
                self.qcp[2] = self.trn.cntpos[2] - self.trn.cntsolimp[2]/2 #(self.qcm[2]-np.sqrt(np.linalg.norm(rc - r1)**2-np.linalg.norm(self.qcm[0:2]-r3[0:2])**2)) +  self.trn.cntpos[2] - self.trn.cntsolimp[2]/2 -r1[2]  # Contact point deformation
                ftplac.append([ctime, self.rsw, r2, self.qcp])

                # plt.plot(self.qcp[0], self.qcp[1], 'ko')

            # print('Angular momentum abt qcp', np.cross(self.qcm - self.qcp, m * self.dqcm))
            print('qcp=',self.qcp,'qcm=',self.qcm,'dqcm=',self.dqcm)
            print('l=',l,'r1=',r1,'r2=',r2,'r3=',r3)
            l = np.linalg.norm(self.qcm - self.qcp)
            theulr = findeulr(self.qcm, self.qcp, l)
            qqt = euler2quat(theulr)
            dqcp = np.array([0, 0, 0])
            dth = np.matmul(np.linalg.pinv(l * np.array([[0, np.cos(theulr[1]), 0],
                                                        [-np.cos(theulr[0]) * np.cos(theulr[1]),
                                                        np.sin(theulr[0]) * np.sin(theulr[1]), 0],
                                                        [-np.sin(theulr[0]) * np.cos(theulr[1]),
                                                        -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), self.dqcm - dqcp)
            xml_str, r = modifysip(m, I, l, self.qcm, self.qcp, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp,self.trn.cntsolref, self.trn.cntsolimp)  # create new xml file from basic sip

            # MuJoCo data structures
            model = mujoco.MjModel.from_xml_string(xml_str)  # MuJoCo model
            data = mujoco.MjData(model)  # MuJoCo data
            # cam = mujoco.MjvCamera()  # Abstract camera
            # opt = mujoco.MjvOption()  # visualization options
            # plt.plot(qcm[0], qcm[1], 'go')  # Plot COM position at leg transition
            # cam.lookat = np.array([qcp[0], qcp[1], 3.0])

        # print(sipdata)
        # Plot qcm --- SIP COM position
        if vis==1:
            fig, ax = plt.subplots(nrows=2, ncols=2)
            fig.suptitle('COM position')
            for item in sipdata:
                # plt.plot(item[0],item[1][0],'ro')
                # plt.xlabel('Time (s)')
                ax[0][1].plot(item[1][0], item[1][1], 'r.')
                # ax[0][1].set_xlabel('X (m)')
                ax[0][1].set_ylabel('Y (m))')
                ax[1][0].plot(item[1][1], item[1][2], 'r.')
                ax[1][0].set_xlabel('Y (m)')
                ax[1][0].set_ylabel('Z (m))')
                ax[1][1].plot(item[1][0], item[1][2], 'r.')
                ax[1][1].set_xlabel('X (m)')
                # ax[1][1].set_ylabel('Z (m))')
            # plt.savefig('SIPxyz.png')
            # plt.show()
            plt.close()

        # glfw.terminate()
        # sipdata.append([ctime+data.time, data.qpos.copy(), data.qvel.copy(), qcp.copy(), fc.copy()])
        # np.savez('siptraj.npz', sipdata=sipdata, ftplac=ftplac)
        # Saving the data:
        # with open('siptraj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        #     pickle.dump([sipdata, ftplac], f)

        return sipdata, ftplac

    def sipstep(self, simstart, simend, simfreq, vis):
        # Parameters of SIP
        m=self.m
        I=self.I
        #qcm=self.qcm.copy()
        #dqcm=self.dqcm.copy()
        #qcp=self.qcp.copy()
        #l = self.l.copy() #np.linalg.norm(qcm - qcp)

        l = np.linalg.norm(self.qcm - self.qcp)
        theulr = findeulr(self.qcm, self.qcp, l)
        qqt = euler2quat(theulr)
        dqcp = np.array([0, 0, 0])
        self.dth = np.matmul(np.linalg.pinv(l * np.array(
            [[0, np.cos(theulr[1]), 0],
            [-np.cos(theulr[0]) * np.cos(theulr[1]), np.sin(theulr[0]) * np.sin(theulr[1]), 0],
            [-np.sin(theulr[0]) * np.cos(theulr[1]), -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), self.dqcm - dqcp)

        self.trn.cntplane(self.qcp,self.spno)
        xml_str, r = modifysip(m, I, l, self.qcm, self.qcp, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp, self.trn.cntsolref, self.trn.cntsolimp)

        # MuJoCo data structures
        model = mujoco.MjModel.from_xml_string(xml_str)  # MuJoCo model
        data = mujoco.MjData(model)  # MuJoCo data
        cam = mujoco.MjvCamera()  # Abstract camera
        opt = mujoco.MjvOption()  # visualization options

        # Example on how to set camera configuration
        # cam.azimuth = 90
        # cam.elevation = -45
        # cam.distance = 2
        # cam.lookat = np.array([0.0, 0.0, 0])

        ctime = simstart  # total time of all sip motions
        # Run walking pattern generation
        sipdata = []
        ftplac = []
        while ctime < simend and len(ftplac)<1:
            # data.contact.solref
            data, sipdata, tf, self.qcm, self.dqcm, self.rc, self.r1, self.r2, self.r3 = sipmotion(model, data, simend, simfreq, self.spno, m, l, self.qcm, qqt,
                                                                    self.qcp, self.dqcm, self.dth, self.rc, self.r1, self.r2, self.r3, self.sspbydsp, self.xlimft,
                                                                    self.ylimft, self.En_des, self.AM_CMspl, ctime,
                                                                    sipdata, vis)
            try:
                # stiffness
                kn = data.contact.solimp[0][0] / ((data.contact.solimp[0][1] ** 2) * (data.contact.solref[0][0] ** 2) * (
                        data.contact.solref[0][1] ** 2))
                print(kn)  # stiffness
                # damping
                print(2 / (data.contact.solimp[0][1] * data.contact.solref[0][0]))
            except:
                print('stiffness/dampting error')

            # Increment time
            ctime = ctime + tf
            # Change SSP/DSP
            self.spno = 3 - self.spno
            if self.spno == 2:
                self.plno = 1  # change contact plane
                print('Angular momentum abt r1', np.cross(self.qcm - self.qcp, m * self.dqcm))
                self.qcp = self.r2.copy()
                print('foot placement position r3', self.r3)
                plt.plot(self.qcp[0], self.qcp[1], 'bo')
                # Contact parameters
                self.Stlr = np.array([1, 1]) - self.Stlr
                # zpln = self.qcp[2] + self.trn[self.plno].solimp[2]  # -r+(0*9.81/kn)
                # solref = self.trn[self.plno].solref
                # solimp = self.trn[self.plno].solimp
                # nocp = self.trn[self.plno].nocp
                # Swing foot pos at the end of DSP
                self.rsw=np.append(self.r1[0:2], self.trn.pos[self.trn.cntgeomid][2]+self.trn.size[self.trn.cntgeomid][2])
                # Foot placement position
                self.trn.cntplane(self.r3, self.spno)
                # Stance foot pos at the start of DSP
                self.rst=np.append(self.r3[0:2], self.trn.pos[self.trn.cntgeomid][2]+self.trn.size[self.trn.cntgeomid][2])

                # Contact point data for DSP
                self.trn.cntplane(self.qcp, self.spno)
                ftplac.append([ctime, self.r1, self.r2, self.rst])
            else:
                print('Angular momentum abt r2', np.cross(self.qcm - self.qcp, m * self.dqcm))
                # Contact parameters
                if self.Stlr[0] == 1:
                    self.plno = 0  # change contact plane
                else:
                    self.plno = 2  # change contact plane
                # zpln = self.trn[self.plno].zpln
                # solref = self.trn[self.plno].solref
                # solimp = self.trn[self.plno].solimp
                # nocp = self.trn[self.plno].nocp
                self.qcp = self.r3.copy()  # Change contact point
                # Foot placement data
                self.trn.cntplane(self.qcp, self.spno)
                self.qcp[2] = self.trn.cntpos[2] - self.trn.cntsolimp[2]  # Contact point deformation

                ftplac.append([ctime, self.rsw, self.r2, self.qcp])
                plt.plot(self.qcp[0], self.qcp[1], 'ko')

            print('Angular momentum abt qcp', np.cross(self.qcm - self.qcp, m * self.dqcm))
            l = np.linalg.norm(self.qcm - self.qcp)
            theulr = findeulr(self.qcm, self.qcp, l)
            qqt = euler2quat(theulr)
            dqcp = np.array([0, 0, 0])
            self.dth = np.matmul(np.linalg.pinv(l * np.array([[0, np.cos(theulr[1]), 0],
                                                        [-np.cos(theulr[0]) * np.cos(theulr[1]),
                                                        np.sin(theulr[0]) * np.sin(theulr[1]), 0],
                                                        [-np.sin(theulr[0]) * np.cos(theulr[1]),
                                                        -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), self.dqcm - dqcp)
            xml_str, r = modifysip(m, I, l, self.qcm, self.qcp, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp, self.trn.cntsolref, self.trn.cntsolimp)
            # MuJoCo data structures
            model = mujoco.MjModel.from_xml_string(xml_str)  # MuJoCo model
            data = mujoco.MjData(model)  # MuJoCo data
            # cam = mujoco.MjvCamera()  # Abstract camera
            # opt = mujoco.MjvOption()  # visualization options
            # plt.plot(qcm[0], qcm[1], 'go')  # Plot COM position at leg transition
            # cam.lookat = np.array([qcp[0], qcp[1], 3.0])

        # print(sipdata)
        # Plot qcm --- SIP COM position
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.suptitle('COM position')
        for item in sipdata:
            # plt.plot(item[0],item[1][0],'ro')
            # plt.xlabel('Time (s)')
            ax[0][1].plot(item[1][0], item[1][1], 'r.')
            # ax[0][1].set_xlabel('X (m)')
            ax[0][1].set_ylabel('Y (m))')
            ax[1][0].plot(item[1][1], item[1][2], 'r.')
            ax[1][0].set_xlabel('Y (m)')
            ax[1][0].set_ylabel('Z (m))')
            ax[1][1].plot(item[1][0], item[1][2], 'r.')
            ax[1][1].set_xlabel('X (m)')
            # ax[1][1].set_ylabel('Z (m))')
        #plt.savefig('SIPxyz.png')
        # plt.show()
        plt.close()

        # glfw.terminate()
        # sipdata.append([ctime+data.time, data.qpos.copy(), data.qvel.copy(), qcp.copy(), fc.copy()])
        # np.savez('siptraj.npz', sipdata=sipdata, ftplac=ftplac)
        # Saving the data:
        #with open('siptraj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        #    pickle.dump([sipdata, ftplac], f)

        return sipdata, ftplac

    def modifyASIP(self, l, plno, plnpos, plnsize, nocp, solref, solimp):    # def modifysip(m,I,l,qcm,qcp,plno,plnpos,plnsize,nocp,solref,solimp):
        # Load basic xml file of SIP
        xml_path = 'ASIP.xml'  # xml file (assumes this is in the same folder as this file)
        plno=np.minimum(2-plno,plno)

        # get the full path
        dirname = os.path.dirname(__file__) #os.getcwd() 
        # dirname = os.getcwd() #os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath

        xmltree = ET.parse(xml_path)
        root = xmltree.getroot()
        # Change mass,pos, orientation and length of pendulum
        bodyeul=np.zeros([1,3])
        #model.geom_size[2,1]=l/2 # length of cylindrical rod
        for tag_wb in root.findall("worldbody"):
            tagwb_geom=tag_wb.findall('geom')
            tagwb_geom[plno].attrib['pos']=' '.join(map(str, np.array(plnpos))) #change contact plane pos
            tagwb_geom[plno].attrib['size']=' '.join(map(str, np.array(plnsize))) #change contact plane pos
            tagwb_geom[plno].attrib['contype']=' '.join(map(str, np.array([1]))) #activate contact type
            tagwb_geom[plno].attrib['conaffinity']=' '.join(map(str, np.array([1]))) #activate contact affinity
            tagwb_geom[plno].attrib['solref']=' '.join(map(str, solref)) #change solref
            tagwb_geom[plno].attrib['solimp']=' '.join(map(str, solimp)) #change solref

            tag_com=tag_wb.findall('body')
            
            tag_com[0].attrib['pos']=' '.join(map(str, self.qcm)) #change COM pos

            tagbd_inert=tag_com[0].findall('inertial')
            tagbd_inert[0].attrib['mass']=' '.join(map(str, np.array([self.m]))) # change mass
            tagbd_inert[0].attrib['fullinertia']= ' '.join(map(str, np.array([self.I[0,0],self.I[1,1],self.I[2,2],self.I[0,1],self.I[0,2],self.I[1,2]]))) # change inertia self.I to upper triangular array

            tagcom_geom = tag_com[0].findall('geom')

            tag_leg = tag_com[0].findall('body')
            tagleg_geom= tag_leg[0].findall('geom')
            # tagbd_geom[0].attrib['mass']=' '.join(map(str, np.array([self.m]))) # change mass
            r=tagleg_geom[0].attrib['size'].split(" ", 1)[0] #radius of cylinder
            print('rad:',r)
            body_pos = np.zeros([len(tagleg_geom), 3])
            k=3
            for i in np.arange(0, int((len(tagleg_geom))/k)):
                body_pos[i, 2] = -(l-float(r)) / 2  # pos of bodyfrmae of cylindrical rod
                tagleg_geom[i].attrib['pos']=' '.join(map(str, body_pos[i,:])) # change cylinder frame pos
                tagleg_geom[i].attrib['size'] = ' '.join(map(str, np.array([r, (l - float(r)) / 2])))  # change length of cylinder to (l-r)/2
            r=tagleg_geom[-1].attrib['size'].split(" ", 1)[0] #radius of contact sphere
            for i in np.arange(int((len(tagleg_geom))/k+1), len(tagleg_geom)):
                body_pos[i, 2] = -(l-float(r))  # pos of bodyframe of point contact sphere
                tagleg_geom[i].attrib['pos']=' '.join(map(str, body_pos[i,:])) # change contact sphere frame pos
                tagleg_geom[i].attrib['size'] = ' '.join(map(str, np.array([r])))  # change length of cylinder to (l-r)/2

            for i in np.arange(0, len(tagleg_geom)):
                tagleg_geom[i].attrib['contype'] = ' '.join(map(str, np.array([0])))  # deactivate contact type
                tagleg_geom[i].attrib['conaffinity'] = ' '.join(map(str, np.array([0])))  # deactivate contact affinity

            if plno==0:
                tagcom_geom[0].attrib['contype'] = ' '.join(map(str, np.array([1])))  # activate contact type
                tagcom_geom[0].attrib['conaffinity'] = ' '.join(map(str, np.array([1])))  # activate contact affinity
                for i in np.arange(0, nocp): #int((len(tag3) - 1) /k+1)):
                    tagleg_geom[i].attrib['contype'] = ' '.join(map(str, np.array([1])))  # activate contact type
                    tagleg_geom[i].attrib['conaffinity'] = ' '.join(map(str, np.array([1])))  # activate contact affinity
                    tagleg_geom[i].attrib['mass'] = ' '.join(map(str, np.array([0.01*self.m])))  # change mass to 0.01*m kg

            else:
                tagcom_geom[0].attrib['contype'] = ' '.join(map(str, np.array([0])))  # activate contact type
                tagcom_geom[0].attrib['conaffinity'] = ' '.join(map(str, np.array([0])))  # activate contact affinity
                for i in np.arange(int((len(tagleg_geom)) /k), int((len(tagleg_geom)) /k ) + nocp ): #len(tag3)):
                    tagleg_geom[i].attrib['contype'] = ' '.join(map(str, np.array([1])))  # activate contact type
                    tagleg_geom[i].attrib['conaffinity'] = ' '.join(map(str, np.array([1])))  # activate contact affinity


            #for geotag2 in tag1.findall('geom'):
                #geotag2.attrib['size']
        # Write the xml tree to sipmotion.xml file and return path of the file
        # xmltree.write('sipmotion.xml')
        xml_str = ET.tostring(root)
        # print(xml_str)
        # ET.dump(root)
        # xml_path = 'sipmotion.xml'
        return xml_str,float(r)

    def ASIPtraj(self, simend, simfreq, vis):
        # Parameters of SIP
        m=self.m
        I=self.I
        #qcm=self.qcm.copy()
        #dqcm=self.dqcm.copy()
        #qcp=self.qcp.copy()
        l = np.linalg.norm(self.qcm - self.qcp)
        theulr = findeulr(self.qcm, self.qcp, l)
        # print('theulr:',theulr)
        qqt = euler2quat(0*theulr)
        # dqcp = np.array([0, 0, 0])
        dth = np.matmul(np.linalg.pinv(l * np.array(
            [[0, np.cos(theulr[1]), 0],
            [-np.cos(theulr[0]) * np.cos(theulr[1]), np.sin(theulr[0]) * np.sin(theulr[1]), 0],
            [-np.sin(theulr[0]) * np.cos(theulr[1]), -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), 0*self.dqcm - 0 * self.dqcm)

        dthcm = 0*dth

        # zpln = trn[plno].zpln
        # solref = trn[plno].solref
        # solimp = trn[plno].solimp
        # nocp = trn[plno].nocp

        self.trn.cntplane(self.qcp, self.spno)
        xml_str, r = self.modifyASIP(l, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp, self.trn.cntsolref, self.trn.cntsolimp)  # create new xml file from basic sip
        print(xml_str)

        # MuJoCo data structures
        model = mujoco.MjModel.from_xml_string(xml_str)  # MuJoCo model
        data = mujoco.MjData(model)  # MuJoCo data
        cam = mujoco.MjvCamera()  # Abstract camera
        opt = mujoco.MjvOption()  # visualization options

        # Example on how to set camera configuration
        # cam.azimuth = 90
        # cam.elevation = -45
        # cam.distance = 2
        # cam.lookat = np.array([0.0, 0.0, 0])

        # print(DepthvsForce(model,data,0))

        ctime = 0  # total time of all sip motions
        if self.spno == 1:
            rc = self.qcm.copy()
            r1 = self.qcp.copy()
            r2 = 0*self.qcp
            r3 = 0*self.qcp
            sspbydsp=min(self.sspbydsp,self.zSw/self.trn.cntsolimp[2]) 
        else:
            rc = self.qcm.copy()
            r1 = self.qcm.copy()
            r1[2]=0
            r2 = self.qcp.copy()
            r3 = 2*self.qcp-self.qcm
            r3[0]=self.qcp[0]
            r3[1]=self.qcp[1]
            r3[2]=0
        print('qcm=',self.qcm,'qcp=',self.qcp,'l=',l)
        # Run walking pattern generation
        sipdata = []
        ftplac = []
        while ctime < simend:
            # data.contact.solref
            # print('dqcm =',self.dqcm)
            data, sipdata, tf, self.qcm, self.dqcm, rc, r1, r2, r3 = self.ASIPmotion(model, data, simend, simfreq, self.spno, m, l, self.qcm, qqt,
                                                                    self.qcp, self.dqcm, dthcm, dth, rc, r1, r2, r3, sspbydsp, ctime,
                                                                    sipdata, vis)
            # try:
            #     # stiffness
            #     kn = data.contact.solimp[0][0] / ((data.contact.solimp[0][1] ** 2) * (data.contact.solref[0][0] ** 2) * (
            #             data.contact.solref[0][1] ** 2))
            #     print('stiffness of SIP',kn)  # stiffness
            #     # damping
            #     print('damping of SIP',2 / (data.contact.solimp[0][1] * data.contact.solref[0][0]))
            # except:
            #     print('stiffness/dampting error')

            # Increment time
            ctime = ctime + tf
            # Change SSP/DSP
            self.spno = 3 - self.spno
            if self.spno == 2: # DSP
                self.plno = 1  # change contact plane
                # print('Angular momentum abt r1', np.cross(self.qcm - self.qcp, m * self.dqcm))
                self.qcp = r2.copy()
                # print('foot placement position r3', r3)
                # Contact parameters
                #Switch leg support
                self.Stlr = np.array([1, 1]) - self.Stlr

                # deformation of contact point:
                #   dz_r1 =  self.trn.cntsolimp[2]/2
                #   print('Error in z deformation, i.e, dz_qcp=',dz_r1)
                # Swing foot pos at the start of DSP: r1
                # Swing foot pos at the end of DSP
                self.rsw = np.append(r1[0:2], self.trn.pos[self.trn.cntgeomid][2] + self.trn.size[self.trn.cntgeomid][2])
                # Stance foot pos at the start of DSP
                self.trn.cntplane(r3, self.spno)
                self.rst = np.append(r3[0:2], self.trn.pos[self.trn.cntgeomid][2] + self.trn.size[self.trn.cntgeomid][2])
                #Stance foot pos at the end of DSP: r3
                # Foot placement data
                ftplac.append([ctime, r1, r2, self.rst])
                self.trn.cntplane(self.qcp, self.spno)

                # plt.plot(self.qcp[0], self.qcp[1], 'bo')

            else: # SSP
                # print('Angular momentum abt r2', np.cross(self.qcm - self.qcp, m * self.dqcm))
                # Contact parameters
                if self.Stlr[0] == 1:
                    self.plno = 0  # change contact plane
                else:
                    self.plno = 2  # change contact plane
                self.qcp = r3.copy()  # Change contact point
                #self.qcp[2] = 0 - self.trn.cntsolimp[2]  # Contact point deformation
                # Foot placement data
                self.trn.cntplane(self.qcp, self.spno)
                #if same pendulum length in each SSP
                # self.qcp[2] = (self.qcm[2]-np.sqrt(np.linalg.norm(rc - r1)**2-np.linalg.norm(self.qcm[0:2]-r3[0:2])**2)) +  self.trn.cntpos[2] - self.trn.cntsolimp[2]/2 -r1[2]  # Contact point deformation
                #else Change pendulum length in each SSP based on terrain height
                self.qcp[2] = self.trn.cntpos[2] - self.trn.cntsolimp[2]/2 #(self.qcm[2]-np.sqrt(np.linalg.norm(rc - r1)**2-np.linalg.norm(self.qcm[0:2]-r3[0:2])**2)) +  self.trn.cntpos[2] - self.trn.cntsolimp[2]/2 -r1[2]  # Contact point deformation
                ftplac.append([ctime, self.rsw, r2, self.qcp])
                sspbydsp=min(self.sspbydsp,self.zSw/self.trn.cntsolimp[2]) 


                # plt.plot(self.qcp[0], self.qcp[1], 'ko')

            # print('Angular momentum abt qcp', np.cross(self.qcm - self.qcp, m * self.dqcm))
            l = np.linalg.norm(self.qcm - self.qcp)
            print('qcp=',self.qcp,'qcm=',self.qcm,'dqcm=',self.dqcm)
            print('l=',l,'r1=',r1,'r2=',r2,'r3=',r3)
            theulr = findeulr(self.qcm, self.qcp, l)
            qqt = euler2quat(0*theulr)
            dqcp = np.array([0, 0, 0])
            dthcm=data.qvel[3:6].copy()
            dth = data.qvel[6:].copy()

            xml_str, r = self.modifyASIP(l, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp, self.trn.cntsolref, self.trn.cntsolimp)  # create new xml file from basic sip
            print(xml_str)


            # MuJoCo data structures
            model = mujoco.MjModel.from_xml_string(xml_str)  # MuJoCo model
            data = mujoco.MjData(model)  # MuJoCo data
            # cam = mujoco.MjvCamera()  # Abstract camera
            # opt = mujoco.MjvOption()  # visualization options
            # plt.plot(qcm[0], qcm[1], 'go')  # Plot COM position at leg transition
            # cam.lookat = np.array([qcp[0], qcp[1], 3.0])

        # print(sipdata)
        # Plot qcm --- SIP COM position
        if vis==1:
            fig, ax = plt.subplots(nrows=2, ncols=2)
            fig.suptitle('COM position')
            for item in sipdata:
                # plt.plot(item[0],item[1][0],'ro')
                # plt.xlabel('Time (s)')
                ax[0][1].plot(item[1][0], item[1][1], 'r.')
                # ax[0][1].set_xlabel('X (m)')
                ax[0][1].set_ylabel('Y (m))')
                ax[1][0].plot(item[1][1], item[1][2], 'r.')
                ax[1][0].set_xlabel('Y (m)')
                ax[1][0].set_ylabel('Z (m))')
                ax[1][1].plot(item[1][0], item[1][2], 'r.')
                ax[1][1].set_xlabel('X (m)')
                # ax[1][1].set_ylabel('Z (m))')
            # plt.savefig('SIPxyz.png')
            # plt.show()
            plt.close()

        # glfw.terminate()
        # sipdata.append([ctime+data.time, data.qpos.copy(), data.qvel.copy(), qcp.copy(), fc.copy()])
        # np.savez('siptraj.npz', sipdata=sipdata, ftplac=ftplac)
        # Saving the data:
        # with open('siptraj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        #     pickle.dump([sipdata, ftplac], f)

        return sipdata, ftplac
    
    def ASIPmotion(self,model,data,simend,freq,spno,m,l,qcm,qqt,qcp,dqcm,dthcm,dth,rc,r1,r2,r3,sspbydsp,ctime,sipdata,vis):
        #sipdata.qcm=[]
        #sipdata.dqcm=[]
        # Set pos and orientation of com
        data.qpos[0:3]=qcm # position of com
        data.qpos[3:7]=qqt # orientation of com
        theulr = findeulr(qcm,qcp,l)
        data.qpos[7:]=theulr[:2]
        data.qvel[0:3]=dqcm  # vel of com
        data.qvel[3:6] = dthcm  # ang vel of com
        dth = np.matmul(np.linalg.pinv(l * np.array([[0, np.cos(theulr[1]), 0],
                                                        [-np.cos(theulr[0]) * np.cos(theulr[1]),
                                                        np.sin(theulr[0]) * np.sin(theulr[1]), 0],
                                                        [-np.sin(theulr[0]) * np.cos(theulr[1]),
                                                        -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), dqcm - 0*dqcm)
        data.qvel[6:]=dth[0:2]

        #Set orientation of legs

        #Energy differnce for actuation
        dEn = self.En_des - ( 1/2*m*np.linalg.norm(dqcm)**2 + m*9.81*(qcm[2]-r1[2]) )
        print('Energy diff=',dEn)


        self.r1x_err=0
        #Find actuation torques
        if spno==2 and dEn>0:
            #Find angle of pendulum from qcm-r2 line
            print('qcm_z',qcm[2],'l',l)
            # thx=(np.arctan2(qcm[0]-r2[0],np.sqrt(l**2-np.linalg.norm(qcm[0]-r2[0])**2)))
            # thy=(np.arctan2(qcm[1]-r2[1],np.sqrt(l**2-np.linalg.norm(qcm[1]-r2[1])**2)))
            thxy=(np.arctan2(np.linalg.norm(qcm[0:2]-r2[0:2]),np.sqrt(l**2-np.linalg.norm(qcm[0]-r2[0])**2)))
            tauxy = 1*dEn /(2*abs(thxy))/10
            # Torque splines about x and y axis of body frame
            # taux = dEn/(2*abs(thy))/100
            # tauy = dEn/(2*abs(thx))/1000
            # print('Pendulum angles:',thx,thy)
            print('torques needed:',tauxy)
        else:
            tauxy=0
            # taux=0
            # tauy=0

        #initialize the controller
        #init_controller(model,data)

        #set the controller
        #mujoco.set_mjcb_control(controller)

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
            viewer.opt.flags[16] = 1  # Contact Forces
            # viewer.opt.flags[18] = 1  # Transparent
            viewer.opt.flags[20] = 1 #COM

            # Update scene and render
            # viewer.sync()

            # print("Press any key to proceed.")
            # key = keyboard.read_key()

            # print("Simulation starting.....")
            # time.sleep(5)

            while viewer.is_running() and data.time < simend:
                time_prev = data.time

                clock_start = time.time()
                while ((vis==0)+((data.time - time_prev) < (1.0 / freq)))>0 and data.time < simend:
                    #print(vis==0,1/freq,(data.time - time_prev),((vis==0)+((data.time - time_prev) < (1.0 / freq)))>0)
                    mujoco.mj_step(model, data)
                    # Update sipdata
                    qcm = data.qpos[0:3]
                    dqcm = data.qvel[0:3]
                    qqt = data.qpos[3:7]                    
                    # Euler angles
                    # thFW = quat2euler(qqt)
                    # T_FW = quat2mat(qqt)
                    # print('thFW:',thFW)
                    theulr = data.qpos[7:]
                    # theulr[0:2] += thFW[0:2]
                    # theulr=quat2euler(qqt)
                    # position of contact point
                    qcp=qcm - quat2mat(qqt) @ quat2mat(euler2quat([theulr[0],theulr[1],0])) @ np.array([0,0,l])  #l*np.array([np.sin(theulr[1]),-np.cos(theulr[1])*np.sin(theulr[0]),np.cos(theulr[1])*np.cos(theulr[0])])
                    # print(qcp)
                    #Body torque on COM
                    # if spno==1:
                    #     data.qfrc_applied[3:6] = [AMspl[0](ctime + data.time, 1),AMspl[1](ctime + data.time, 1),AMspl[2](ctime + data.time, 1)]
                    #     data.qfrc_applied[3:6] = 0.01*data.qfrc_applied[3:6]
                    # data.qfrc_applied[4] = 100*( -1*(qcm[2]>qcp[2]) + 1*(qcm[2]<qcp[2]) )*(data.qvel[0] - 0.1)

                    #RAte of change of AM_CMspl is the torque needed
                    data.ctrl[0]= self.AM_CMspl[0](data.time,1)
                    data.ctrl[1]= self.AM_CMspl[1](data.time,1)
                    print('torques:',data.ctrl[0],data.ctrl[1])

                    # if spno==1:
                        # print(data.qvel[0])
                        # data.ctrl[0] = 0.05*np.sign(qcm[1]-qcp[1]) #Torque to maintain cyclicity
                        # data.ctrl[1]= 1000*(self.vel-data.qvel[0]) #0.005*np.sign(qcm[1]-qcp[1])
                    #     # print(m*9.81*(qcm[0]-qcp[0])/l)
                    #     data.ctrl[1]= -0.001 #-m*9.81*(qcm[0]-qcp[0])/l
                    #     tau_FW = self.I @ np.array([data.qacc[6],data.qacc[7],0]) + np.array([[0, 0, data.qvel[7]], [0, 0, -data.qvel[6]], [-data.qvel[7], data.qvel[6], 0]]) @ (self.I @ np.array([data.qvel[6],data.qvel[7],0])) 
                    #     print('tau:',tau_FW)
                    #     data.ctrl[0]= -(tau_FW[0]/5) #*np.sign(qcm[1]-qcp[1])
                    #     data.ctrl[1]= -(tau_FW[1]/5) #*np.sign(qcm[0]-qcp[0])
                    # else:
                    #     data.ctrl[1]= 1*(self.vel-data.qvel[0]) #0.005*np.sign(qcm[1]-qcp[1])
                    #     data.ctrl[0] = data.ctrl[1]*(qcm[0]-qcp[0])/(qcm[1]-qcp[1]) #0.05*np.sign(qcm[1]-qcp[1]) #Torque to maintain cyclicity in suspended pendulum


                    # Actuation to input energy if required -- it will destroy the symetry of SIP gait
                    # if spno==2 and dEn>0:
                    #     data.ctrl[0] = tauxy*abs(dqcm[1])/np.linalg.norm(dqcm[0:2]) #taux #dEn/(2*abs(rc[0]-r2[0]))*l/2 #0.005 #Torque to pitch forward in DSP 
                    #     data.ctrl[1] = -tauxy*abs(dqcm[0])/np.linalg.norm(dqcm[0:2])*np.sign(r3[1]-r1[1]) #tauy*np.sign(r3[1]-r1[1]) #dEn/(2*abs(rc[1]-r2[1]))*l/2 #0.0001*np.sign(r3[1]-r1[1]) #Torque to roll towards r3 in DSP

                    #Error in periodic SIP motion
                    if abs(data.qvel[1])<0.001:
                        self.r1x_err = qcm[0]-qcp[0]
                        print(qcm[0]-qcp[0])
                        print('r1x_err,qcp',self.r1x_err,qcp)
                        # self.r1x_err = 0 
                    # print(data.qvel[0])
                    #print(data.qfrc_applied)

                    # Contact force
                    fc = np.zeros([6])
                    for i in np.arange(0, data.ncon):
                        #conid = data.contact[i].geom1
                        fci = np.zeros([6])
                        try:
                            mujoco.mj_contactForce(model, data, i, fci)
                            fc = fc + fci
                        except:
                            print('no contact')
                    sipdata.append([ctime+data.time, data.qpos.copy(), data.qvel.copy(), qcp.copy(), fc.copy(), data.subtree_angmom[0]])
                    # Check condition for leg transition
                    # print('hw:',data.subtree_angmom[0])
                    if spno == 1:
                        rc, r1, r2, r3 = ftstep(m,qcm, dqcm, qcp,sspbydsp, self.r1x_err)
                    legtrans_check = legtrans(rc,r1,r2,r3,qcm,qcp,dqcm,spno,self.xlimft,self.ylimft)
                    if legtrans_check == 1 and len(sipdata)>1:
                        break
                if legtrans_check == 1 and len(sipdata)>1:
                    print(ctime + data.time)
                    # print(data.qvel)
                    break

                if (data.time>=simend):
                    break
                if vis==1:
                    print(ctime + data.time)
                    plt.figure(10)
                    #Subplot of size 3x1  with common x-axis
                    plt.subplots_adjust(hspace=0.5)
                    for i in range(3):
                        plt.subplot(3,1,i+1)
                        plt.plot(ctime+data.time,data.qvel[i],'.k')
                        # plt.plot(ctime+data.time,data.subtree_angmom[1][i],'.k')
                    # plt.plot(ctime+data.time,qcp[2],'.r')
                    # plt.plot(ctime+data.time,data.qpos[0],'*r',ctime+data.time,data.qpos[1],'*g')
                    # plt.plot(ctime+data.time,data.qvel[0],'*r',ctime+data.time,data.qvel[1],'*g',ctime+data.time,data.qvel[2],'*b')
                    # plt.plot(ctime+data.time,data.subtree_angmom[1][0],'*r',ctime+data.time,data.subtree_angmom[1][1],'*g',ctime+data.time,data.subtree_angmom[1][2],'*b')
                    # plt.plot(ctime+data.time,data.ctrl[1],'ok')
                    # plt.plot(ctime+data.time,r2[1],'ok')
                    plt.xlabel('Time')
                    # plt.ylabel('dq/dt')
                    plt.ylabel('hw')
                    plt.pause(0.001)

                # Update scene and render
                viewer.sync()
                time_until_next_step = 1 / freq - (time.time() - clock_start)
                if time_until_next_step > 0 and vis==1:
                    time.sleep(time_until_next_step)
                    time.sleep(1)

        return data,sipdata, data.time, qcm,dqcm, rc,r1,r2,r3

# Find foot placement position when in SSP
def ftstep(m,rc,drc,qcp,sspbydsp,r1x_err=0):
    r1=qcp
    l=np.linalg.norm(rc-r1)
    #print(np.cross(rc-r1,m*drc)) # angular momentum abot cnt pt
    Lz=m*((rc[0]-r1[0])*drc[1]-(rc[1]-r1[1])*drc[0]) #angular momentum about Z axis
    try:
        dr=np.array([drc[1]*Lz/(m*np.linalg.norm(drc[0:2])**2), -drc[0]*Lz/(m*np.linalg.norm(drc[0:2])**2), 0]) # diff of COP in leg transition
    except:
        dr=0*rc
    pd=(l/sspbydsp)/np.linalg.norm(r1+dr-rc)
    r2=rc-(r1+dr-rc)*pd
    # r3=np.array([r1[0]+2*(r2[0]-r1[0]), r1[1]+2*(r2[1]-r1[1]), r1[2]])
    #Correction in r3 to avoid drift in SIP motion
    r3=np.array([r1[0]+r1x_err/2+2*(r2[0]-r1[0]-r1x_err/2), r1[1]+2*(r2[1]-r1[1]), r1[2]])
    # print('r1,r2,r3:',r1,r2,r3)
    return rc, r1, r2, r3


# Check condition to transition from SSP to DSP or vice versa
def legtrans(rc,r1,r2,r3,qcm,qcp,dqcm,spno,xlimft,ylimft):
    if spno==1:
        #print(abs(r3[0]-r1[0])>xlimft)
        cond= (np.dot(dqcm[0:2],qcm[0:2]-qcp[0:2])>0) and (abs(r3[0]-r1[0])>xlimft or abs(r3[1]-r1[1])>ylimft)
    else:
        #Sym about r2
        cond= ( ((abs(r2[0]-qcm[0])>abs(r2[0]-rc[0])) and (abs(r2[1]-qcm[1])>abs(r2[1]-rc[1]))) ) and np.linalg.norm(qcm-r3)<=np.linalg.norm(qcm-r1)
    return cond

# Generate SIP motion in MuJoCo for one phase (SSP or DSP) and break the loop if leg transition cond is true
def sipmotion(model,data,simend,freq,spno,m,l,qcm,qqt,qcp,dqcm,dth,rc,r1,r2,r3,sspbydsp,xlimft,ylimft,En_des,AMspl,ctime,sipdata,vis):
    #sipdata.qcm=[]
    #sipdata.dqcm=[]
    # Set pos and orientation
    data.qpos[0:3]=qcm # position of com
    data.qpos[3:]=qqt # orientation of com
    data.qvel[0:3]=dqcm  # vel of com
    data.qvel[3:] = dth  # ang vel of com

    #Energy differnce for actuation
    dEn = En_des - ( 1/2*m*np.linalg.norm(dqcm)**2 + m*9.81*(qcm[2]-r1[2]) )
    print('Energy diff=',dEn)

    #Find angle of pendulum from qcm-r2 line
    print('qcm_z',qcm[2],'l',l)
    thx=(np.arctan2(qcm[0]-r2[0],np.sqrt(l**2-np.linalg.norm(qcm[0]-r2[0])**2)))
    thy=(np.arctan2(qcm[1]-r2[1],np.sqrt(l**2-np.linalg.norm(qcm[1]-r2[1])**2)))

    #Find actuation torques
    if spno==2 and dEn>0:
        # Torque splines about x and y axis of body frame
        taux = 0*dEn/(2*abs(thy))/100
        tauy = 0*dEn/(2*abs(thx))/1000
        print('Pendulum angles:',thx,thy)
        print('torques needed:',taux,tauy)
    else:
        taux=0
        tauy=0

    #initialize the controller
    #init_controller(model,data)

    #set the controller
    #mujoco.set_mjcb_control(controller)

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
        viewer.opt.flags[16] = 1  # Contact Forces
        # viewer.opt.flags[18] = 1  # Transparent
        viewer.opt.flags[20] = 1 #COM

        # Update scene and render
        # viewer.sync()

        # print("Press any key to proceed.")
        # key = keyboard.read_key()

        # print("Simulation starting.....")
        # time.sleep(5)

        while viewer.is_running() and data.time < simend:
            time_prev = data.time

            clock_start = time.time()
            while ((vis==0)+((data.time - time_prev) < (1.0 / freq)))>0 and data.time < simend:
                #print(vis==0,1/freq,(data.time - time_prev),((vis==0)+((data.time - time_prev) < (1.0 / freq)))>0)
                mujoco.mj_step(model, data)
                # Update sipdata
                qcm = data.qpos[0:3]
                dqcm = data.qvel[0:3]
                qqt = data.qpos[3:]
                # Euler angles
                theulr=quat2euler(qqt)
                # position of contact point
                qcp=qcm-l*np.array([np.sin(theulr[1]),-np.cos(theulr[1])*np.sin(theulr[0]),np.cos(theulr[1])*np.cos(theulr[0])])
                #Body torque on COM
                # if spno==1:
                #     data.qfrc_applied[3:6] = [AMspl[0](ctime + data.time, 1),AMspl[1](ctime + data.time, 1),AMspl[2](ctime + data.time, 1)]
                #     data.qfrc_applied[3:6] = 0.01*data.qfrc_applied[3:6]
                # data.qfrc_applied[4] = 100*( -1*(qcm[2]>qcp[2]) + 1*(qcm[2]<qcp[2]) )*(data.qvel[0] - 0.1)

                # Actuation to input energy if required
                # if spno==2 and dEn>0:
                #     data.qfrc_applied[4] = taux #dEn/(2*abs(rc[0]-r2[0]))*l/2 #0.005 #Torque to pitch forward in DSP 
                #     data.qfrc_applied[5] = tauy*np.sign(r3[1]-r1[1]) #dEn/(2*abs(rc[1]-r2[1]))*l/2 #0.0001*np.sign(r3[1]-r1[1]) #Torque to roll towards r3 in DSP


                # print(data.qvel[0])
                #print(data.qfrc_applied)

                # Contact force
                fc = np.zeros([6])
                for i in np.arange(0, data.ncon):
                    #conid = data.contact[i].geom1
                    fci = np.zeros([6])
                    try:
                        mujoco.mj_contactForce(model, data, i, fci)
                        fc = fc + fci
                    except:
                        print('no contact')
                sipdata.append([ctime+data.time, data.qpos.copy(), data.qvel.copy(), qcp.copy(), fc.copy(), data.subtree_angmom[0]])
                # Check condition for leg transition
                # print('hw:',data.subtree_angmom[0])
                if spno == 1:
                    rc, r1, r2, r3 = ftstep(m,qcm, dqcm, qcp,sspbydsp)
                legtrans_check = legtrans(rc,r1,r2,r3,qcm,qcp,dqcm,spno,xlimft,ylimft)
                if legtrans_check == 1 and len(sipdata)>1:
                    break
            if legtrans_check == 1 and len(sipdata)>1:
                print(ctime + data.time)
                # print(data.qvel)
                break

            if (data.time>=simend):
                break
            if vis==1:
                print(ctime + data.time)
                plt.figure(10)
                #Subplot of size 3x1  with common x-axis
                plt.subplots_adjust(hspace=0.5)
                for i in range(3):
                    plt.subplot(3,1,i+1)
                    # plt.plot(ctime+data.time,data.qvel[i],'.k')
                    plt.plot(ctime+data.time,data.subtree_angmom[1][i],'.k')

                # plt.plot(ctime+data.time,qcp[2],'.r')
                # plt.plot(ctime+data.time,data.qvel[0],'*r',ctime+data.time,data.qvel[1],'*g',ctime+data.time,data.qvel[2],'*b')
                # plt.plot(ctime+data.time,data.subtree_angmom[1][0],'*r',ctime+data.time,data.subtree_angmom[1][1],'*g',ctime+data.time,data.subtree_angmom[1][2],'*b')
                plt.xlabel('Time')
                plt.ylabel('dq/dt')
                plt.pause(0.001)

            # Update scene and render
            viewer.sync()
            time_until_next_step = 1 / freq - (time.time() - clock_start)
            if time_until_next_step > 0 and vis==1:
                time.sleep(time_until_next_step)
                time.sleep(1)

    return data,sipdata, data.time, qcm,dqcm, rc,r1,r2,r3


# Modify sip.xml file to change the parameters of the pendulum (m,I,l,qcm,qcp) and terrain in SSP and DSP (solref, solimp)
def modifysip(m,I,l,qcm,qcp,plno,plnpos,plnsize,nocp,solref,solimp):
    # Load basic xml file of SIP
    xml_path = 'sip.xml'  # xml file (assumes this is in the same folder as this file)
    plno=np.minimum(2-plno,plno)

    # get the full path
    dirname = os.path.dirname(__file__) #os.getcwd() 
    # dirname = os.getcwd() #os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath

    xmltree = ET.parse(xml_path)
    root = xmltree.getroot()
    # Change mass,pos, orientation and length of pendulum
    bodyeul=np.zeros([1,3])
    #model.geom_size[2,1]=l/2 # length of cylindrical rod
    for tag_wb in root.findall("worldbody"):
        tagwb_geom=tag_wb.findall('geom')
        tagwb_geom[plno].attrib['pos']=' '.join(map(str, np.array(plnpos))) #change contact plane pos
        tagwb_geom[plno].attrib['size']=' '.join(map(str, np.array(plnsize))) #change contact plane pos
        tagwb_geom[plno].attrib['contype']=' '.join(map(str, np.array([1]))) #activate contact type
        tagwb_geom[plno].attrib['conaffinity']=' '.join(map(str, np.array([1]))) #activate contact affinity
        tagwb_geom[plno].attrib['solref']=' '.join(map(str, solref)) #change solref
        tagwb_geom[plno].attrib['solimp']=' '.join(map(str, solimp)) #change solref

        tag_body=tag_wb.findall('body')
        tag_body[0].attrib['pos']=' '.join(map(str, qcm)) #change COM pos
        tagbd_geom= tag_body[0].findall('geom')
        tagbd_geom[0].attrib['mass']=' '.join(map(str, np.array([m]))) # change mass
        r=tagbd_geom[-1].attrib['size'].split(" ", 1)[0] #radius of cylinder and contact sphere
        body_pos = np.zeros([len(tagbd_geom), 3])
        k=3
        for i in np.arange(1, int((len(tagbd_geom)-1)/k+1)):
            body_pos[i, 2] = -(l-float(r)) / 2  # pos of bodyfrmae of cylindrical rod
            tagbd_geom[i].attrib['pos']=' '.join(map(str, body_pos[i,:])) # change cylinder frame pos
            tagbd_geom[i].attrib['size'] = ' '.join(map(str, np.array([r, (l - float(r)) / 2])))  # change length of cylinder to (l-r)/2
        for i in np.arange(int((len(tagbd_geom)-1)/k+1), len(tagbd_geom)):
            body_pos[i, 2] = -(l-float(r))  # pos of bodyframe of point contact sphere
            tagbd_geom[i].attrib['pos']=' '.join(map(str, body_pos[i,:])) # change contact sphere frame pos
            tagbd_geom[i].attrib['size'] = ' '.join(map(str, np.array([r])))  # change length of cylinder to (l-r)/2

        for i in np.arange(1, len(tagbd_geom)):
            tagbd_geom[i].attrib['contype'] = ' '.join(map(str, np.array([0])))  # deactivate contact type
            tagbd_geom[i].attrib['conaffinity'] = ' '.join(map(str, np.array([0])))  # deactivate contact affinity

        if plno==0:
            tagbd_geom[0].attrib['contype'] = ' '.join(map(str, np.array([1])))  # activate contact type
            tagbd_geom[0].attrib['conaffinity'] = ' '.join(map(str, np.array([1])))  # activate contact affinity
            for i in np.arange(1, 1+nocp): #int((len(tagbd_geom) - 1) /k+1)):
                tagbd_geom[i].attrib['contype'] = ' '.join(map(str, np.array([1])))  # activate contact type
                tagbd_geom[i].attrib['conaffinity'] = ' '.join(map(str, np.array([1])))  # activate contact affinity
        else:
            tagbd_geom[0].attrib['contype'] = ' '.join(map(str, np.array([0])))  # activate contact type
            tagbd_geom[0].attrib['conaffinity'] = ' '.join(map(str, np.array([0])))  # activate contact affinity
            for i in np.arange(int((len(tagbd_geom) - 1) /k + 1), int((len(tagbd_geom) - 1) /k + 1) + nocp ): #len(tagbd_geom)):
                tagbd_geom[i].attrib['contype'] = ' '.join(map(str, np.array([1])))  # activate contact type
                tagbd_geom[i].attrib['conaffinity'] = ' '.join(map(str, np.array([1])))  # activate contact affinity


        #for geotag2 in tag1.findall('geom'):
            #geotag2.attrib['size']
    # Write the xml tree to sipmotion.xml file and return path of the file
    # xmltree.write('sipmotion.xml')
    xml_str = ET.tostring(root)
    # print(xml_str)
    # ET.dump(root)
    # xml_path = 'sipmotion.xml'
    return xml_str,float(r)

# Find Euler angles from COM position and Contact position of SIP model
def findeulr(qcm,qcp,l):
    return np.array([np.arctan2(qcp[1] - qcm[1], qcm[2] - qcp[2]), np.arctan2(qcm[0] - qcp[0], np.linalg.norm(qcm[1:] - qcp[1:]) ), 0.0])

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

