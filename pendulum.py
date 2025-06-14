import pybullet as p
import time
import pybullet_data
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from control.matlab import place, lqr

guiFlag = True

dt = 1 / 240
th0 = 0.1
thd = 2.0
kp = 40.0
kd = 20.0
g = 10
L = 0.8
m = 1
T_transition = 3.0

A = np.array([[0, 1], [-g / L, 0]])
B = np.array([[0], [1 / (m * L * L)]])
Q = np.array([[1e3, 0], [0, 1e-2]])
R = 1e-2
K, *_ = lqr(A, B, Q, R)
K = -K


def fifth_order_traj(t, T, q0, qf, qd0=0, qdf=0, qdd0=0, qddf=0):
    if t <= 0:
        return q0, qd0, qdd0
    if t >= T:
        return qf, qdf, qddf
    
    tau = t / T

    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    a0 = q0
    a1 = qd0 * T
    a2 = qdd0 * T * T / 2
    a3 = 10 * (qf - q0) - (6 * qd0 + 4 * qdf) * T - (1.5 * qdd0 - 0.5 * qddf) * T * T
    a4 = -15 * (qf - q0) + (8 * qd0 + 7 * qdf) * T + (1.5 * qdd0 - qddf) * T * T
    a5 = 6 * (qf - q0) - 3 * (qd0 + qdf) * T - 0.5 * (qdd0 - qddf) * T * T

    q = a0 + a1 * tau + a2 * tau2 + a3 * tau3 + a4 * tau4 + a5 * tau5
    qd = (a1 + 2 * a2 * tau + 3 * a3 * tau2 + 4 * a4 * tau3 + 5 * a5 * tau4) / T
    qdd = (2 * a2 + 6 * a3 * tau + 12 * a4 * tau2 + 20 * a5 * tau3) / (T * T)

    return q, qd, qdd


physicsClient = p.connect(p.GUI if guiFlag else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("./simple.urdf.xml", useFixedBase=True)

p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=th0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

maxTime = 5
logTime = np.arange(0, maxTime, dt)
sz = len(logTime)
logThetaSim = np.zeros(sz)
logVelSim = np.zeros(sz)
logTauSim = np.zeros(sz)
logThetaRef = np.zeros(sz)
logVelRef = np.zeros(sz)
logAccRef = np.zeros(sz)

idx = 0

for t in logTime:
    th, vel = p.getJointState(boxId, 1)[:2]
    logThetaSim[idx] = th
    logVelSim[idx] = vel

    th_ref, vel_ref, acc_ref = fifth_order_traj(t, T_transition, th0, thd)

    logThetaRef[idx] = th_ref
    logVelRef[idx] = vel_ref
    logAccRef[idx] = acc_ref

    tau_fb = m * L * L * (acc_ref - kp * (th - th_ref) - kd * (vel - vel_ref))
    tau_ff = m * L * L * (g / L * np.sin(th_ref))
    tau = tau_fb + tau_ff

    logTauSim[idx] = tau

    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, force=tau, controlMode=p.TORQUE_CONTROL)
    p.stepSimulation()

    idx += 1
    if guiFlag:
        time.sleep(dt)

p.disconnect()

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(logTime, logThetaSim, 'b', label="Sim Pos")
plt.plot(logTime, logThetaRef, 'r--', label="Ref Pos")
plt.grid(True)
plt.legend()
plt.ylabel('Position (rad)')

plt.subplot(3, 1, 2)
plt.plot(logTime, logVelSim, 'b', label="Sim Vel")
plt.plot(logTime, logVelRef, 'r--', label="Ref Vel")
plt.grid(True)
plt.legend()
plt.ylabel('Velocity (rad/s)')

plt.subplot(3, 1, 3)
plt.plot(logTime, logTauSim, 'g', label="Sim Tau")
plt.grid(True)
plt.legend()
plt.ylabel('Torque (Nm)')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()