# import sys
# import numpy as np
#
# from pymedit import Mesh3D, trunc3DMesh, connectedComponents3D, cube, P1Function3D
# from pymedit import mmg3d, mshdist, advect, P0Function3D, P1Vector3D, saveToVtk
#
#
# if len(sys.argv)>=3 and sys.argv[1]=='--debug':
#     debug = int(sys.argv[2])
# else:
#     debug = 0
# if len(sys.argv)>=4 and sys.argv[3]=='--noplot':
#     import medit.external as ext
#     def medit(*args,**kwargs):
#         pass
#     ext.medit = medit
#
# M = cube(30, 30, 30, debug=debug)
# M.plot("Plotting initial mesh",keys=["b","c","e","Z","Z","Z","Z"])#, silent=False)
# phi =\
#     P1Function3D(M, lambda x: np.cos(10*x[0])*np.cos(10*x[1])*np.cos(10*x[2])-0.3)
# newM = mmg3d(M, 0.02, 0.05, 1.3, sol=phi, ls=True)
#
# regions = P0Function3D(newM, connectedComponents3D(newM))
# regions.plot(title="Plotting Connected components of the new mesh obtained with mmg3d", keys=["b","m","F1","F2","Z","Z","Z","Z"])
#
# phi = mshdist(newM,debug=debug)
# phi.plot(title="Plotting Signed distance function computed with mshdist", keys=["b","m","Z","Z","Z","Z"])
#
# theta = P1Vector3D(newM, lambda x : [0.3,0,0])
# phiAdvected = advect(newM, phi, theta, T=1,debug=debug)
# phiAdvected.plot(title="Advected level set function with advect",keys=["b","m","Z","Z","Z","Z"])
#
#
# saveToVtk(newM, [regions,phi,theta,phiAdvected],
#           ["regions","phi","theta",
#            "phiAdvected"], [0,1,1,1], "out.vtu")


from pymedit import P1Function3D, mmg3d, Mesh3D, cube, trunc3DMesh, saveToVtk
import numpy as np

N = 10

c = 0.2
M = cube(N, N, N, lambda x, y, z :  [x-0.5, y-0.5, z-0.5])

# saveToVtk(M, output="/run/media/ttsesm/external_data/data_for_testing/out_.vtu")

# def phi(x): return np.abs(x[0])+np.abs(x[1])+np.abs(x[2])-c
#
#
# def pphi(x):
#     res = 1
#     for i in range(-2, 3):
#         for j in range(-2,3):
#             for k in range(-2,3):
#                 vect = np.array([i, j, k, 0])*2*c
#                 res = np.minimum(res, phi(x-vect))
#     return res
#
#
# p = P1Function3D(M, pphi)

hmin = 0.0005
hmax = 0.001
hgrad = 1.3
hausd = 0.1*hmin
# Mnew = mmg3d(M, hmin, hmax, hgrad, hausd, sol=p, ls=True, nr=False, debug=10)
# Mnew = mmg3d(M, hmin, hmax, hgrad, hausd, nr=False, debug=10, extra_args="-nosurf")
Mnew = mmg3d(M, hmin, hmax, hgrad, hausd, nr=False, debug=10)
# Ms=trunc3DMesh(Mnew, 3)
# Ms.plot(keys=["b","c","e","Z","Z","Z","Z"])
Mnew.plot(keys=["b","c","e","Z","Z","Z","Z"])

saveToVtk(Mnew, output="/run/media/ttsesm/external_data/data_for_testing/out1.vtu")
