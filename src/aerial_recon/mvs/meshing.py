"""Surface meshing of fused point clouds with Open3D. Scaffold (`uv sync --extra mesh`).

Screened Poisson reconstruction (Kazhdan & Hoppe, 2013) fits an indicator function whose
gradient matches the oriented normals, then extracts its level set. It needs consistently
oriented normals: for aerial scenes, orienting toward +z (up) or toward the cameras works.
Poisson happily invents surface where there is no data; trim by density.

Alternatives worth trying: Delaunay + graph cut (OpenMVS ReconstructMesh), TSDF fusion of
depth maps (open3d.pipelines.integration), and for splats, 2DGS / Gaussian-surfel meshing.
"""

from __future__ import annotations

import numpy as np


def poisson_mesh(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    depth: int = 9,
    normal_radius: float | None = None,
    density_quantile: float = 0.02,
    orient_up: bool = True,
):
    """Returns an open3d.geometry.TriangleMesh."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, np.float64)))
    if colors is not None:
        c = np.asarray(colors, dtype=np.float64)
        c = c / 255.0 if c.max() > 1.0 else c
        pcd.colors = o3d.utility.Vector3dVector(c.reshape(-1, 3) if c.ndim > 1 else
                                                np.repeat(c[:, None], 3, axis=1))
    if normal_radius is None:
        extent = np.ptp(np.asarray(points), axis=0).max()
        normal_radius = 0.01 * extent
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    if orient_up:
        pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))
    return mesh


def write_mesh(path, mesh) -> None:
    import open3d as o3d

    o3d.io.write_triangle_mesh(str(path), mesh)
