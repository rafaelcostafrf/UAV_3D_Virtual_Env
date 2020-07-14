import numpy as np
"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    Quaternion and euler utility algorithms.
    Converts euler to quaternions
    Converts quaternions to euler
    Computes the quaternions derivative (Useful for time integration of quaternions)
    Converts quaternions to rotatio matrix
"""
def euler_quat(ang):
    #ROTACAO 3-2-1
    phi = ang[0]
    theta = ang[1]
    psi = ang[2]
    
    cp = np.cos(phi/2)
    sp = np.sin(phi/2)
    ct = np.cos(theta/2)
    st = np.sin(theta/2)
    cps = np.cos(psi/2)
    sps = np.sin(psi/2)

    q0 = cp*ct*cps+sp*st*sps
    q1 = sp*ct*cps-cp*st*sps
    q2 = cp*st*cps+sp*ct*sps
    q3 = cp*ct*sps-sp*st*cps
    q = np.array([[q0, q1, q2, q3]]).T
    q = q/np.linalg.norm(q)
    return q


def quat_euler(q):
    phi = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]**2+q[2]**2))
    theta = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
    psi = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
    phi = phi[0]
    theta = theta[0]
    psi = psi[0]
    if any(np.isnan([phi, theta, psi])):
        print('Divergencia na conversao Quaternion - Euler')
    return np.array([phi, theta, psi])


def deriv_quat(w, q):
    w = w.flatten()
    q.reshape((4,1))
    wx = w[0]
    wy = w[1]
    wz = w[2]   
    omega = np.array([[0, -wx, -wy, -wz],
                      [wx, 0, wz, -wy],
                      [wy, -wz, 0, wx],
                      [wz, wy, -wx, 0]])
    dq = 1/2*np.dot(omega,q).flatten()
    return dq

def quat_rot_mat(q):
    q = q.flatten()
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]
    R = np.array([[a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
                  [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
                  [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]])
    return R