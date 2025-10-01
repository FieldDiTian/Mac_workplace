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
g_funcs = [lambda t: hw1.g01(t, v1, R1), lambda t: hw1.g02(t, v2, R2)]
names = ["1", "2"]
CoordFrame.animate_g(g_funcs, 30, names=names)