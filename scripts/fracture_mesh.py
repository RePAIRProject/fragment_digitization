# from vedo import *
# import numpy as np
# np.random.seed(0)
#
# # s = Sphere(quads=False, res=15).clean()
# # s = Cube().clean()
# s = Mesh("../data/RPf_00047.obj").scale(0.06)
# res = 0.2  #control the tetras resolution
#
# # fill the space w/ points
# pts = (np.random.rand(10000, 3)-0.5)*2
# # fillpts = s.insidePoints(pts).subsample(res).scale(0.9)  # make pts uniform
# fillpts = s.insidePoints(pts).scale(0.9)  # make pts uniform
# seeds = s.clone().subsample(0.1).ps(12).c('black') # pick uniform pts on sphere
# printc("# of pieces, #points in sphere:", seeds.N(), fillpts.N())
#
# tmesh = delaunay3D(merge(fillpts,s))
# # assign a closest point to each tetrahedron
# cids = []
# for p in tmesh.cellCenters():
# 	cid = seeds.closestPoint(p, returnPointId=True)
# 	cids.append(cid)
#
# tmesh.celldata["fragment"] = np.array(cids)
#
# pieces = []
# for i in range(seeds.NPoints()):
# 	tc = tmesh.clone().threshold(above=i-0.1, below=i+0.1)
# 	mc = tc.tomesh(fill=False).color(i)
# 	pieces.append(mc)
#
# ############### animate
# plt = Plotter(interactive=1)
# plt.show(pieces, seeds, "press q to make it explode")
# for i in range(15):
# 	for pc in pieces:
# 		cm = pc.centerOfMass()
# 		pc.shift(cm/15)
# 	plt.render()
# plt.interactive().close()

"""Add a custom scalar to a TetMesh to segment it.
Press q to make it explode"""
from vedo import show, Mesh, Sphere, TetMesh, Plotter, dataurl, printc, Text2D
# from vedo import *
import pygalmesh
import tetgen
import pymeshfix

n = 20000
f1 = 0.005  # control the tetras resolution
f2 = 0.15   # control the nr of seeds

s = Sphere(quads=False, res=15).clean()

# mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
#     # "../data/elephant.vtu",
# 	'../data/Tombstone2.obj',
#     min_facet_angle=25.0,
#     max_radius_surface_delaunay_ball=0.15,
#     max_facet_distance=0.008,
#     max_circumradius_edge_ratio=3.0,
#     verbose=False,
# )

# # repair and tetralize the closed surface
# amesh = Mesh(dataurl+'bunny.obj')
# # amesh = Mesh('../data/Tombstone2.obj')
# meshfix = pymeshfix.MeshFix(amesh.points(), amesh.faces())
# meshfix.repair() # will make it manifold
# repaired = Mesh(meshfix.mesh)
# tet = tetgen.TetGen(repaired.points(), repaired.faces())
# tet.tetrahedralize(order=1, mindihedral=210, minratio=1.2, maxvolume=0.01)
# tmesh = TetMesh(tet.grid)

tmesh = TetMesh('/run/media/ttsesm/external_data/data_for_testing/out.vtu')
surf = tmesh.tomesh(fill=False)

# pick uniform pts on the surface
seeds = surf.clone().subsample(f2).ps(10).c('black')
printc("#pieces:", seeds.N())

# assign to each tetrahedron the id of the closest seed point
cids = []
for p in tmesh.cellCenters():
	cid = seeds.closestPoint(p, returnPointId=True)
	cids.append(cid)
tmesh.celldata["fragment"] = cids

pieces = []
for i in range(seeds.NPoints()):
	tc = tmesh.clone().threshold(name="fragment", above=i-0.1, below=i+0.1)
	mc = tc.tomesh(fill=False).color(i)
	pieces.append(mc)

############### animate
plt = Plotter(size=(1200,800), axes=1)
plt.show(__doc__, pieces)
for i in range(20):
	for pc in pieces:
		cm = pc.centerOfMass()
		pc.shift(cm/25)
	plt.render()
plt.interactive().close()