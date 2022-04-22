from vedo import *
import numpy as np

def find_correspondences(meshes):

    m1_pts = meshes[0].points()
    m2_pts = meshes[1].points()

    correspondences = []

    for i, p in enumerate(m1_pts):
        iclos = meshes[1].closestPoint(p, returnPointId=True)
        correspondences.append([i, iclos])

    dist_thres = 0.05
    correspondences = np.asarray(correspondences).reshape(-1, 2)
    correspondences = correspondences[np.where(mag(m1_pts[correspondences[:,0]] - m2_pts[correspondences[:,1]]) < dist_thres), :]

    return correspondences.squeeze()

def transform_mesh(m):
    cm = m.centerOfMass()
    m.shift(-cm)
    elli = pcaEllipsoid(m, pvalue=0.5)

    ax1 = versor(elli.axis1)
    ax2 = versor(elli.axis2)
    ax3 = versor(elli.axis3)

    T = np.array([ax1, ax2, ax3])  # the transposed matrix is already the inverse
    # print(T)
    # print(T@ax1)

    return m.applyTransform(T, reset=True)


def main():
    # mesh_files = load(['/run/media/ttsesm/external_data/repair_dataset/tuwien/cake/pcd/cake_part01.xyz', '/run/media/ttsesm/external_data/repair_dataset/tuwien/cake/pcd/cake_part02.xyz'])
    files = ['/run/media/ttsesm/external_data/repair_dataset/tuwien/cake/pcd/cake_part01.xyz', '/run/media/ttsesm/external_data/repair_dataset/tuwien/cake/pcd/cake_part02.xyz']

    mesh_files = []
    mesh_models = []

    for i, file in enumerate(files):
        m = Mesh(file).color(i)
        mesh_files.append(m)
        # m.color(i)
        mesh = transform_mesh(m.clone())
        mesh_models.append(mesh)


    corrs = find_correspondences(mesh_files)

    pts1 = Points(mesh_files[0].points()[list(corrs[:,0])], r=5, c='blue')
    pts2 = Points(mesh_files[1].points()[list(corrs[:, 1])], r=5, c='red')

    pts1_ = Points(mesh_models[0].points()[list(corrs[:, 0])], r=5, c='blue')
    pts2_ = Points(mesh_models[1].points()[list(corrs[:, 1])], r=5, c='red')

    # vd.show(mesh_models_original, pts1, pts2, axes=1, interactive=1).close()
    show(mesh_files, pts1, pts2, "Original", at=0, N=2, axes=1, sharecam=False)
    show(mesh_models, pts1_, pts2_, "Testing", at=1, interactive=True, sharecam=False).close()
    # show(Points(mesh_files[0].points()), Points(mesh_files[1].points()), pts1, pts2, "Before", at=0, N=2, axes=1, sharecam=False)
    # show(Points(mesh_models[0].points()), Points(mesh_models[1].points()), pts1_, pts2_, "After", at=1, interactive=True, sharecam=False).close()

    return 0

if __name__ == "__main__":
    main()