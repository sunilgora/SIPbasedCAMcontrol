import numpy as np
import matplotlib.pyplot as plt

#Plot set
fsize=11 # font size
plt.rcParams['figure.figsize'] = [4, 4]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'cmr10'  # Computer Modern Roman
# To ensure correct rendering of minus signs in mathematical expressions
plt.rcParams["axes.formatter.use_mathtext"] = True        
plt.rcParams['pdf.fonttype'] = 42  # avoid type3 font
# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# Set the default linewidth to 2
plt.rcParams['lines.linewidth'] = 2
plt.rc('font', size=fsize)  # controls default text sizes
plt.rc('axes', titlesize=fsize)  # fontsize of the axes title
plt.rc('axes', labelsize=fsize)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=fsize)  # fontsize of the tick labels
plt.rc('legend', fontsize=fsize)  # legend fontsize
plt.rc('figure', titlesize=fsize+2)  # fontsize of the figure title
plt.rcParams["axes.spines.right"] = "False"
plt.rcParams["axes.spines.top"] = "False"
plt.rcParams['axes.autolimit_mode'] = 'round_numbers' # to avoid offset in axis
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['axes.grid'] = True

def myDataplots(DesData,ActData,humn):
    # Saved data = [t, q, dq, rcom, drcom, oL, oR, rcop, fcl, fcr, tau, I * dq, WD]
    # Plots
    #plt.rcParams.update({'font.size': 12})
    #col = ['r', 'g', 'b', 'k', 'c', 'y']
    col = ['r', 'g', 'b', 'c', 'm', 'k']

    # plt.figure(1)  # Joint position     # Act vs Des q
    fig1, ax1 = plt.subplots(nrows=5, sharex=True)  # joint traj
    #fig1.suptitle('Joint angle traj')
    Xdata = np.empty((0))
    Y1data = np.empty((0, len(DesData[0][1]))) #qdes
    Y2data = np.empty((0, len(ActData[0][1]))) #qact
    Y3data = np.empty((0, len(DesData[0][2]))) #dqdes
    Y4data = np.empty((0, len(ActData[0][2]))) #dqact

    for idata in DesData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[1]]), axis=0) #qdes
        Y3data=np.append(Y3data,np.array([idata[2]]), axis=0) #dqdes
    for idata in ActData:
        # Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y2data=np.append(Y2data,np.array([idata[1]]), axis=0) #q
        Y4data=np.append(Y4data,np.array([idata[2]]), axis=0) #dq

    for i in humn.left_legjnts:  # model.nv):
        ax1[0].plot(Xdata, Y1data[:,i] * 180 / np.pi, 'o', color=col[i - min(humn.left_legjnts)], label='')
        ax1[0].plot(Xdata, Y2data[:,i] * 180 / np.pi, '-', color=col[i - min(humn.left_legjnts)], label='$\u03B8_{'+str(i - min(humn.left_legjnts) + 1)+'}$')
    for i in humn.right_legjnts:  # model.nv):
        ax1[1].plot(Xdata, Y1data[:,i] * 180 / np.pi, 'o', color=col[i - min(humn.right_legjnts)], label='')
        ax1[1].plot(Xdata, Y2data[:,i] * 180 / np.pi, '-', color=col[i - min(humn.right_legjnts)], label='$\u03B8_{'+str(i - min(humn.right_legjnts) + 1+len(humn.left_legjnts))+'}$')
    for i in humn.ub_jnts[0:2]:  # model.nv):
        ax1[2].plot(Xdata, Y1data[:,i] * 180 / np.pi, 'o', color=col[(i - min(humn.ub_jnts))%6], label='')
        ax1[2].plot(Xdata, Y2data[:,i] * 180 / np.pi, '-', color=col[(i - min(humn.ub_jnts))%6], label='$\u03B8_{'+str(i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts))+'}$')
    #Kondo Torso, left and right arms
    for i in humn.ub_jnts[2:6]:  # model.nv):
        ax1[3].plot(Xdata, Y1data[:,i] * 180 / np.pi, 'o', color=col[(i - min(humn.ub_jnts)-2)%6], label='')
        ax1[3].plot(Xdata, Y2data[:,i] * 180 / np.pi, '-', color=col[(i - min(humn.ub_jnts)-2)%6], label='$\u03B8_{'+str(i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts))+'}$')
    for i in humn.ub_jnts[6:10]:  # model.nv):
        ax1[4].plot(Xdata, Y1data[:,i] * 180 / np.pi, 'o', color=col[(i - min(humn.ub_jnts)-6)%6], label='')
        ax1[4].plot(Xdata, Y2data[:,i] * 180 / np.pi, '-', color=col[(i - min(humn.ub_jnts)-6)%6], label='$\u03B8_{'+str(i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts))+'}$')

    ax1[len(ax1)-1].set_xlabel('Time (s)')
    for i in range(5):
        ax1[i].set_ylabel('Angle (deg)')
        # ax1[i].grid(visible=None, which='major', axis='both')
        ax1[i].legend(loc='upper center',frameon=False, bbox_to_anchor=(0.5, 1.2), ncol = max(1,len(ax1[i].lines)) )


    # plt.figure(1000)
    # plt.plot(Xdata, Y2data[:,9]* 180 / np.pi)
    # plt.plot(Xdata, Y2data[:,15]* 180 / np.pi)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angle (deg)')

    # plt.figure(11)  # Joint velocity     # Act vs Des dq/dt
    fig11, ax11 = plt.subplots(nrows=3, sharex=True)  # joint rates

    for i in humn.left_legjnts:  # model.nv):
        ax11[0].plot(Xdata, Y3data[:,i], 'o', color=col[i - min(humn.left_legjnts)], label=f'dth_des_{i - min(humn.left_legjnts) + 1}')
        ax11[0].plot(Xdata, Y4data[:,i], '-', color=col[i - min(humn.left_legjnts)], label=f'dth_act_{i - min(humn.left_legjnts) + 1}')
        ax11[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax11[0].lines)) )
    for i in humn.right_legjnts:  # model.nv):
        ax11[1].plot(Xdata, Y3data[:,i], 'o', color=col[i - min(humn.right_legjnts)], label=f'dth_des_{i - min(humn.right_legjnts) + 1+len(humn.left_legjnts)}')
        ax11[1].plot(Xdata, Y4data[:,i], '-', color=col[i - min(humn.right_legjnts)], label=f'dth_act_{i - min(humn.right_legjnts) + 1+len(humn.left_legjnts)}')
        ax11[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax11[1].lines)) )
    for i in humn.ub_jnts:  # model.nv):
        ax11[2].plot(Xdata, Y3data[:,i], 'o', color=col[(i - min(humn.ub_jnts))%6], label=f'dth_des_{i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts)}')
        ax11[2].plot(Xdata, Y4data[:,i], '-', color=col[(i - min(humn.ub_jnts))%6], label=f'dth_act_{i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts)}')
        ax11[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax11[2].lines)) )

    ax11[2].set_xlabel('Time (s)')
    ax11[0].set_ylabel('Joint rate (rad/s)')
    # ax11[0].grid(visible=None, which='major', axis='both')
    ax11[1].set_ylabel('Joint rate (rad/s)')
    # ax11[1].grid(visible=None, which='major', axis='both')
    ax11[2].set_ylabel('Joint rate (rad/s)')
    # ax11[2].grid(visible=None, which='major', axis='both')


    # plt.figure(2) # COM, COP position
    fig2, ax2 = plt.subplots(nrows=3, sharex=True)  # COM and COP Position with time
    #fig2.suptitle('COM and COP traj')

    Xdata = np.empty((0))
    Y1data = np.empty((0, 3))
    Y2data = np.empty((0, 3))
    Y3data = np.empty((0, 3))
    Y4data = np.empty((0, 3))
    for idata in DesData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[3]]),axis=0) #rcom
        Y2data=np.append(Y2data,np.array([idata[7]]),axis=0) #rcop

    for idata in ActData:
        # Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y3data=np.append(Y3data,np.array([idata[3]]),axis=0) #rcom
        Y4data=np.append(Y4data,np.array([idata[7]]),axis=0) #rcop

    print('Tracking RMS error in COM_x = ', np.linalg.norm(Y1data[:,0]-Y3data[:,0])/np.sqrt(len(Y1data[:,0]))*1e3,'mm')
    print('Tracking RMS error in COM_y = ', np.linalg.norm(Y1data[:,1]-Y3data[:,1])/np.sqrt(len(Y1data[:,1]))*1e3,'mm')
    print('Tracking RMS error in COM_z = ', np.linalg.norm(Y1data[:,2]-Y3data[:,2])/np.sqrt(len(Y1data[:,2]))*1e3,'mm')
    # print('Tracking RMS error in ZMP_x = ', np.linalg.norm(Y2data[:,0]-Y4data[:,0])/np.sqrt(len(Y2data[:,0]))*1e3,'mm')
    # print('Tracking RMS error in ZMP_y = ', np.linalg.norm(Y2data[:,1]-Y4data[:,1])/np.sqrt(len(Y2data[:,1]))*1e3,'mm')
    # print('Tracking RMS error in ZMP_z = ', np.linalg.norm(Y2data[:,2]-Y4data[:,2])/np.sqrt(len(Y2data[:,2]))*1e3,'mm')

    for i in np.arange(0, 3):
        ax2[i].plot(Xdata, Y1data[:,i], '.b', label=f'COM_des')  # COM_des
        ax2[i].plot(Xdata, Y3data[:,i], '-b', label=f'COM_act')  # COM
        ax2[i].plot(Xdata, Y2data[:,i], '.g', label=f'COP_des')  # COP_des
        ax2[i].plot(Xdata, Y4data[:,i], '-g', label=f'COP_act')  # COP
        ax2[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax2[0].lines)) )
    # plt.figure()
    for i in np.arange(0, 3):
        ax22 = ax2[i].twinx()
        ax22.plot(Xdata,Y1data[:,i]-Y3data[:,i],'-r')
        ax22.set_ylabel('Error (m)')

    errxCOMdata=Y1data[:,0]-Y3data[:,0]
    ax2[2].set_xlabel('Time (s)')
    ax2[0].set_ylabel('X (m)')
    # ax2[0].grid(visible=None, which='major', axis='both')
    ax2[1].set_ylabel('Y (m)')
    # ax2[1].grid(visible=None, which='major', axis='both')
    ax2[2].set_ylabel('Z (m)')
    # ax2[2].grid(visible=None, which='major', axis='both')


    plt.figure()  # COM-COP
    # fig4 = plt.figure(4)  # COM and COP
    plt.plot(Y1data[:,0], Y1data[:,1], '.r', label=f'COM_des')
    plt.plot(Y3data[:,0], Y3data[:,1], '-r', label=f'COM_act')
    plt.plot(Y2data[:,0], Y2data[:,1], '.g', label=f'COP_des')
    plt.plot(Y4data[:,0], Y4data[:,1], '-g', label=f'COP_act')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = 4)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    # plt.grid(visible=None, which='major', axis='both')


    # plt.figure() # Position with time
    fig3, ax3 = plt.subplots(nrows=4, sharex=True)  # Hip, Leg Position with time
    #fig3.suptitle('Hip and Foot traj')
    # Hip
    Xdata = np.empty((0))
    Y1data = np.empty((0, len(DesData[0][1])))
    Y2data = np.empty((0, len(ActData[0][1])))
    for idata in DesData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[1]]), axis=0) #qdes
    for idata in ActData:
        # Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y2data=np.append(Y2data,np.array([idata[1]]), axis=0) #q

    for i in np.arange(0, 3):
        ax3[i].plot(Xdata, Y1data[:,i], '.r', label=f'Hip_des')  # Hip-des
        ax3[i].plot(Xdata, Y2data[:,i], '-r', label=f'Hip_act')  # Hip-act
        ax3[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax3[0].lines)) )


    # Xdata = np.empty((0))
    Y3data = np.empty((0, len(ActData[0][5])))
    Y4data = np.empty((0, len(ActData[0][6])))
    Y5data = np.empty((0, len(ActData[0][5])))
    Y6data = np.empty((0, len(ActData[0][6])))
    for idata in DesData:
        # Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y3data=np.append(Y3data,np.array([idata[5]]), axis=0) #oLeft
        Y4data=np.append(Y4data,np.array([idata[6]]), axis=0) #oRight

    for idata in ActData:
        # Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y5data=np.append(Y5data,np.array([idata[5]]), axis=0) #oLeft
        Y6data=np.append(Y6data,np.array([idata[6]]), axis=0) #oRight

    for i in np.arange(0, 2):
        ax3[i].plot(Xdata, Y3data[:,i], '.g', label=f'Left-Foot_des')  # Left leg
        ax3[i].plot(Xdata, Y5data[:,i], '-g', label=f'Left-Foot_act')  # Left leg
        ax3[i].plot(Xdata, Y4data[:,i], '.b', label=f'Right-Foot_des')  # Right leg
        ax3[i].plot(Xdata, Y6data[:,i], '-b', label=f'Right-Foot_act')  # Right leg
        ax3[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax3[0].lines)) )
    i=2
    ax3[i+1].plot(Xdata, Y3data[:,i], '.g', label=f'Left-Foot_des')  # Left leg
    ax3[i+1].plot(Xdata, Y5data[:,i], '-g', label=f'Left-Foot_act')  # Left leg
    ax3[i+1].plot(Xdata, Y4data[:,i], '.b', label=f'Right-Foot_des')  # Right leg
    ax3[i+1].plot(Xdata, Y6data[:,i], '-b', label=f'Right-Foot_act')  # Right leg
    # ax3[i+1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax3[i+1].lines))

    # plt.figure()
    ax3[3].set_xlabel('Time (s)')
    ax3[0].set_ylabel('X (m)')
    # ax3[0].grid(visible=None, which='major', axis='both')
    ax3[1].set_ylabel('Y (m)')
    # ax3[1].grid(visible=None, which='major', axis='both')
    ax3[2].set_ylabel('Z (m)')
    # ax3[2].grid(visible=None, which='major', axis='both')
    ax3[3].set_ylabel('Z (m)')
    # ax3[3].grid(visible=None, which='major', axis='both')

    # plt.figure() # XY position
    fig33, ax33 = plt.subplots(nrows=2, sharex=True)  # HIP and Foot Traj XY
    ax33[0].plot(Y1data[:,0], Y1data[:,2], '.r', label=f'Hip_des')  # Hip-des
    ax33[0].plot(Y2data[:,0], Y2data[:,2], '-r', label=f'Hip_act')  # Hip-act
    ax33[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax33[0].lines)) )

    ax33[1].plot(Y3data[:,0], Y3data[:,2], '.g', label=f'Left-Foot_des')  # Left leg
    ax33[1].plot(Y5data[:,0], Y5data[:,2], '-g', label=f'Left-Foot_act')  # Left leg
    ax33[1].plot(Y4data[:,0], Y4data[:,2], '.b', label=f'Right-Foot_des')  # Right leg
    ax33[1].plot(Y6data[:,0], Y6data[:,2], '-b', label=f'Right-Foot_act')  # Right leg
    ax33[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax33[1].lines)) )
    # plt.figure()
    ax33[1].set_xlabel('X (m)')
    ax33[0].set_ylabel('Z (m)')
    # ax33[0].grid(visible=None, which='major', axis='both')
    ax33[1].set_ylabel('Z (m)')
    # ax33[1].grid(visible=None, which='major', axis='both')

    plt.figure()  # Normal Contact force
    # fig5 = plt.figure(5)  # Contact force
    Xdata = np.empty((0))
    Y1data = np.empty((0, len(ActData[0][8])))
    Y2data = np.empty((0, len(ActData[0][9])))
    for idata in ActData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[8]]), axis=0) #fcl
        Y2data=np.append(Y2data,np.array([idata[9]]), axis=0) #fcr
    plt.plot(Xdata, Y1data[:,0], '-g', label=f'Left foot')
    plt.plot(Xdata, Y2data[:,0], '-b', label=f'Right foot')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = 2)

    #plt.figure(5)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    # plt.grid(visible=None, which='major', axis='both')


    # plt.figure(6) # Applied joint torque
    fig6, ax6 = plt.subplots(nrows=3, sharex=True)  # Torque
    # fig6.suptitle('Joint torque')
    Xdata = np.empty((0))
    Y1data = np.empty((0, len(ActData[0][10])))
    for idata in ActData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[10]]), axis=0) #data.ctrl

    for i in humn.left_legjnts:
        # plt.plot(data.time, tauid[i], '.r')
        ax6[0].plot(Xdata, Y1data[:,i - 6], color=col[i - min(humn.left_legjnts)], label=r'$\tau_{' + str(i - min(humn.left_legjnts) + 1) + '}$')
        ax6[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax6[0].lines)) )
    for i in humn.right_legjnts:
        # plt.plot(data.time, tauid[i], '.r')
        ax6[1].plot(Xdata, Y1data[:,i - 6], color=col[i - min(humn.right_legjnts)], label=r'$\tau_{'+str(i - min(humn.right_legjnts) + 1+len(humn.left_legjnts))+'}$')
        ax6[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax6[1].lines)) )
    for i in humn.ub_jnts:
        # plt.plot(data.time, tauid[i], '.r')
        ax6[2].plot(Xdata, Y1data[:,i - 6], color=col[(i - min(humn.ub_jnts))%6], label=r'$\tau_{'+str(i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts))+'}$')
        ax6[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = max(1,len(ax6[2].lines)) )

    #plt.figure(6)
    ax6[2].set_xlabel('Time (s)')
    ax6[0].set_ylabel('Torque (Nm)')
    # ax6[0].grid(visible=None, which='major', axis='both')
    ax6[1].set_ylabel('Torque (Nm)')
    # ax6[1].grid(visible=None, which='major', axis='both')
    ax6[2].set_ylabel('Torque (Nm)')
    # ax6[2].grid(visible=None, which='major', axis='both')


    # plt.figure(7)
    fig7, ax71 = plt.subplots(nrows=3, sharex=True)  # Hip, Leg Position with time
    # fig7.suptitle('Angular momentum')
    Xdata = np.empty((0))
    Y1data = np.empty((0, len(ActData[0][11]))) #Lxyz
    Y2data = np.empty((0)) #k_L
    for idata in ActData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[11]]), axis=0) # Iw*dq
        Y2data=np.append(Y2data,np.array([idata[14]]), axis=0) # k_L

    for i in range(3):
        ax71[i].plot(Xdata, Y1data[:,i], '-k')  # Ang momentum
    # plt.figure(7)
    ax71[2].set_xlabel('Time (s)')
    ax71[0].set_ylabel('$h_x$ (kg-m²/s)')
    # ax71[0].grid(visible=None, which='major', axis='both')
    ax71[1].set_ylabel('$h_y$ (kg-m²/s)')
    # ax71[1].grid(visible=None, which='major', axis='both')
    ax71[2].set_ylabel('$h_z$ (kg-m²/s)')
    # ax71[2].grid(visible=None, which='major', axis='both')

    # gain k_L
    # ax72=ax71[0].twinx()
    # ax72.plot(Xdata,Y2data,'--')
    # ax72.set_ylabel('k_L')
    # ax73 = ax71[1].twinx()
    # ax73.plot(Xdata, Y2data, '--')
    # ax73.set_ylabel('k_L')
    # ax74 = ax71[2].twinx()
    # ax74.plot(Xdata, Y2data, '--')
    # ax74.set_ylabel('k_L')

    #plt.figure() #COT
    # fig8 = plt.figure(8)  # COT
    # Create the figure and the first set of axes
    fig8, ax81 = plt.subplots()

    Xdata = np.empty((0))
    Y1data = np.empty((0))
    #Y2data = np.empty((0, 3))
    rcmX0=ActData[0][3][0]
    for idata in ActData:
        rcmX=idata[3][0]-rcmX0
        Xdata = np.append(Xdata, np.array([idata[0]]), axis=0)  # time
        if rcmX>0:
            Y1data=np.append(Y1data,np.array([idata[12]/(humn.m*9.81*rcmX)]),axis=0) #WD/(m*g*rcomX)
        else:
            Y1data = np.append(Y1data, np.array([np.nan]),axis=0)
            #Y2data=np.append(Y2data,np.array([idata[12]]),axis=0) #WD
        #Y3data=np.append(Y3data,np.array([idata[11]]),axis=0) #drcom
    ax81.plot(Xdata, Y1data, '-')
    # plt.legend(loc="upper right")
    ax81.set_xlabel('Time (s)')
    ax81.set_ylabel('COT')
    # ax81.grid(visible=None, which='major', axis='both')

    # Tracking error
    ax82=ax81.twinx()
    ax82.plot(Xdata,errxCOMdata,'--')
    ax82.set_ylabel('Error (m)')

    print("COT = ",Y1data[-1])

    plt.show()
