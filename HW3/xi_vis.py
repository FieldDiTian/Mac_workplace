import hw3
import vpython as vp
import numpy as np
import CoordFrame

v1 = 2
R1 = 3
v2 = 3
R2 = 5

vp.scene.forward = vp.vector(-1, -1, -.5)
vp.scene.up = vp.vector(0, 0, 1)
vp.scene.range = 5

g_0 = CoordFrame.g_from_vec_rot(np.zeros(3), np.eye(3))
CoordFrame.draw(g_0, name="0")

g_01_init = hw2.g01(0, v1, R1)
g_02_init = hw2.g02(0, v2, R2)
print(CoordFrame.integrate_twist(hw2.xi01(v1, R1), 0))
g_funcs= [
lambda t: np.matmul(CoordFrame.integrate_twist(hw2.xi01(v1, R1), t), g_01_init),
lambda t: np.matmul(CoordFrame.integrate_twist(hw2.xi02(v2, R2), t), g_02_init)
]
names = ["1", "2"]
CoordFrame.animate_g(g_funcs, 30, names=names)