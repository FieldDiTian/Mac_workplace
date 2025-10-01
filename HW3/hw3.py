import numpy as np
import CoordFrame

def g02(t, v2, R2):
    '''
    Problem (b)
        Returns the 4x4 rigid body pose g_02 at time t given that the
        satellite is orbitting at radius R2 and linear velocity v2.

        Args:
            t: time at which the configuration is computed.
            v2: linear speed of the satellite, in meters per second.
            R2: radius of the orbit, in meters.
        Returns:
            4x4 rigid pose of frame {2} as seen from frame {0} as a numpy array.

        Functions you might find useful:
            numpy.sin
            numpy.cos
            numpy.array
            numpy.sqrt
            numpy.matmul
            numpy.eye

        Note: Feel free to use one or more of the other functions you have implemented
        in this file.
    '''
    # Calculate the angle: theta = (v2/R2) * t
    theta = (v2 / R2) * t
    
    # Calculate trigonometric values
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Build the transformation matrix T_02(t)
    g = np.array([[sin_theta,  0, -cos_theta, -R2 * sin_theta],
                  [-cos_theta, 0, -sin_theta,  R2 * cos_theta],
                  [0,          1,  0,          0],
                  [0,          0,  0,          1]])
    return g

def g01(t, v1, R1):
    '''
    Problem (c)
        Returns the 4x4 rigid body pose g_01 at time t given that the
        satellite is orbitting at radius R1 and linear velocity v1.

        Args:
            t: time at which the configuration is computed.
            v1: linear speed of the satellite, in meters per second.
            R1: radius of the orbit, in meters.
        Returns:
            4x4 rigid pose of frame {1} as seen from frame {0} as a numpy array.

        Functions you might find useful:
            numpy.sin
            numpy.cos
            numpy.array
            numpy.sqrt
            numpy.matmul
            numpy.eye

        Note: Feel free to use one or more of the other functions you have implemented
        in this file.
    '''
    # Calculate the angle: theta_1 = (v1/R1) * t
    theta_1 = (v1 / R1) * t
    
    # Calculate trigonometric values
    sin_theta1 = np.sin(theta_1)
    cos_theta1 = np.cos(theta_1)
    
    # Constants
    sqrt3_over_2 = np.sqrt(3) / 2
    half = 1.0 / 2
    
    # Build the transformation matrix T_01(t)
    g = np.array([
        [sin_theta1, 0, -cos_theta1, -R1 * sin_theta1],
        [-sqrt3_over_2 * cos_theta1, -half, -sqrt3_over_2 * sin_theta1, sqrt3_over_2 * R1 * cos_theta1],
        [-half * cos_theta1, sqrt3_over_2, -half * sin_theta1, half * R1 * cos_theta1],
        [0, 0, 0, 1]
    ])
    return g

def g21(t, v1, R1, v2, R2):
    '''
    Problem (d)
        Returns the 4x4 rigid body pose g_21 at time t given that the
        first satellite is orbitting at radius R1 and linear velocity v1
        and the second satellite is orbitting at radius R2 and linear 
        velocity v2.

        Args:
            t: time at which the configuration is computed.
            v1: linear speed of satellite 1, in meters per second.
            R1: radius of the orbit of satellite 1, in meters.
            v2: linear speed of satellite 2, in meters per second.
            R2: radius of the orbit of satellite 2, in meters.
        Returns:
            4x4 rigid pose of frame {2} as seen from frame {0} as a numpy array.

        Functions you might find useful:
            numpy.matmul
            numpy.linalg.inv

        Note: Feel free to use one or more of the other functions you have implemented
        in this file.
    '''
    # 计算 g_01: 从坐标系{0}到{1}的变换
    g_01 = g01(t, v1, R1)
    
    # 计算 g_02: 从坐标系{0}到{2}的变换  
    g_02 = g02(t, v2, R2)
    
    # 计算 g_21 = g_02^(-1) * g_01
    # 从坐标系{2}到{1}的变换 = 从{2}到{0}的变换 * 从{0}到{1}的变换
    g_02_inv = np.linalg.inv(g_02)
    g = np.matmul(g_02_inv, g_01)
    
    return g

def xi02(v2, R2):
    '''
    Problem (e)
        Returns the 6x1 twist describing the motion of satellite 2
        given that it is rotating at radius R2 with linear velocity v2.

        Args:
            v2: linear speed of satellite 2, in meters per second.
            R2: radius of the orbit of satellite 2, in meters.
        Returns:
            6x1 twist describing the motion of frame {2} relative to frame {0}

        Functions you might find useful:
            numpy.array

        Note: Feel free to use one or more of the other functions you have implemented
        in this file.
    '''
    # 卫星2绕z0轴（过原点）旋转
    # 角速度 omega_2 = v2/R2
    # 扭转向量 xi_02 = [v; omega] = [0, 0, 0, 0, 0, v2/R2]^T
    xi = np.array([
        0,           # vx = 0 (平移速度x分量)
        0,           # vy = 0 (平移速度y分量)  
        0,           # vz = 0 (平移速度z分量)
        0,           # omega_x = 0 (角速度x分量)
        0,           # omega_y = 0 (角速度y分量)
        v2 / R2      # omega_z = v2/R2 (角速度z分量)
    ])
    return xi

def xi01(v1, R1):
    '''
    Problem (f)
        Returns the 6x1 twist describing the motion of satellite 1
        given that it is rotating at radius R1 with linear velocity v1.

        Args:
            v1: linear speed of satellite 1, in meters per second.
            R1: radius of the orbit of satellite 1, in meters.
        Returns:
            6x1 twist describing the motion of frame {1} relative to frame {0}

        Functions you might find useful:
            numpy.array

        Note: Feel free to use one or more of the other functions you have implemented
        in this file.
    '''
    # 卫星1绕 u1=[0, -1/2, sqrt(3)/2]^T 轴（过原点）旋转
    # 角速度 omega_1 = v1/R1
    # 角速度向量 omega = omega_1 * u1 = (v1/R1) * [0, -1/2, sqrt(3)/2]^T
    # 扭转向量 xi_01 = [v; omega] = [0, 0, 0, 0, -v1/(2*R1), sqrt(3)*v1/(2*R1)]^T
    xi = np.array([
        0,                          # vx = 0 (平移速度x分量)
        0,                          # vy = 0 (平移速度y分量)
        0,                          # vz = 0 (平移速度z分量)
        0,                          # omega_x = 0 (角速度x分量)
        -v1 / (2 * R1),            # omega_y = -v1/(2*R1) (角速度y分量)
        np.sqrt(3) * v1 / (2 * R1) # omega_z = sqrt(3)*v1/(2*R1) (角速度z分量)
    ])
    return xi