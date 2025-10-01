"""
Library for interacting with and visualizing coordinate frames. Modified to use C106A Lab 3 Starter Code
Author: Jay Monga
"""

import numpy as np
import vpython as vp
import kin_func_skeleton

def hat(omega):
    """
    Compute the skew-symmetric matrix
    corresponding to the hat operation on
    vector omega
    """
    return kin_func_skeleton.skew_3d(omega)

def rotX(theta):
    """
    Compute a rotation matrix for
    rotating theta radians about
    the x axis
    """
    rot = np.eye(3)
    rot[1:, 1:] = np.array([[
        np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return rot

def rotY(theta):
    """
    Compute a rotation matrix for
    rotating theta radians about
    the y axis
    """
    rot = np.eye(3)
    rot[0, 0] = np.cos(theta)
    rot[0, 2] = -np.sin(theta)
    rot[2, 0] = np.sin(theta)
    rot[2, 2] = np.cos(theta)
    return rot

def rotZ(theta):
    """
    Compute a rotation matrix for
    rotating theta radians about
    the z axis
    """
    rot = np.eye(3)
    rot[:2, :2] = np.array([[
        np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return rot

def rot_from_axis_angle(axis, angle):
    """
    Implements Rodrigues formula, normalizes the axis
    so always computes rotation matrix for rotation
    of angle radians about axis
    """
    axis = axis/np.linalg.norm(axis)
    return kin_func_skeleton.rotation_3d(axis, angle)

def axis_angle_from_rot(rot):
    """
    Return the axis-angle representaiton
    of a rotation described by a rotation matrix
    """
    axis = np.array([
        rot[2, 1] - rot[1, 2],
        rot[0, 2] - rot[2, 0],
        rot[1, 0] - rot[0, 1]])
    angle = np.arccos((np.trace(rot) - 1)/2)
    return (axis, angle)
    # if np.allclose(rot, rot_from_axis_angle(axis, angle), rtol=1e-2):
        # return (axis, angle)
    # else:
        # return (axis, -angle)

def g_from_vec_rot(vec, rot):
    """
    Shapes a position vector and
    rotation matrix into a homogenous
    transform matrix
    """
    g = np.eye(4)
    g[:3, :3] = rot
    g[:3, 3] = vec
    return g
    
def draw(g, scale=1, name=""):
    """
    Draws the specified coordinate frame
    """
    x_vec = g[:3, 0]
    y_vec = g[:3, 1]
    z_vec = g[:3, 2]
    vec = g[:3, 3]
    x_ax = vp.arrow(pos=vp.vector(*vec), axis=vp.vector(*x_vec), color=vp.vector(1, 0, 0), length=scale)
    y_ax = vp.arrow(pos=vp.vector(*vec), axis=vp.vector(*y_vec), color=vp.vector(0, 1, 0), length=scale)
    z_ax = vp.arrow(pos=vp.vector(*vec), axis=vp.vector(*z_vec), color=vp.vector(0, 0, 1), length=scale)
    text = vp.text(text=name, align="center", pos=vp.vector(*(vec - .4 * z_vec)), up=vp.vector(*z_vec), height=.3)
    frame = vp.compound([x_ax, y_ax, z_ax, text], origin=vp.vector(*vec))
    return frame

def integrate_twist(twist, t):
    """
    Computes the rigid body transform
    from exponentiating the twist for t seconds
    """

    return kin_func_skeleton.homog_3d(twist, t)

def twist_gfunc(twist):
    return lambda t: integrate_twist(twist, t)

def animate_g(g_funcs, t, hz=100, scales=[], names=[]):
    """
    Given a list of functions,
    each describing the motion of some coordinate frame,
    draws out all coordinate frames
    over t seconds
    """
    last_gs = [g_func(0) for g_func in g_funcs]
    if not scales:
        scales = [1] * len(g_funcs)
    if not names:
        names = ["none"] * len(g_funcs)
    frames = [draw(g_func(0), scale=scale, name=name) for g_func, scale, name in zip(g_funcs, scales, names)]
    t_cur = 0
    while(t_cur < t):
        vp.rate(hz)
        t_cur += 1/hz
        for i in range(len(g_funcs)):
            g_func = g_funcs[i]
            frame = frames[i]
            last_g = last_gs[i]
            g_cur = g_func(t_cur)
            g_diff = np.matmul(g_cur, np.linalg.inv(last_g))
            axis, angle = axis_angle_from_rot(g_diff[:3, :3])
            frame.rotate(angle=angle, axis=vp.vector(*axis))
            frame.pos = vp.vector(*g_cur[:3, 3])
            last_gs[i] = g_cur