#%% Importing Libraries
from pathlib import Path
import pandas as pd
import copy
import cv2 as cv
import numpy as np
import open3d as o3d
from tqdm import tqdm
import pickle
from termcolor import colored

#%% Defining Subroutines
def triangulationEngine(pts1, pts2, calib):
    ''' Triangulates 2D stereo points into 3D space, removing optical distortions. '''
    pts1, pts2 = np.asarray(pts1, dtype=np.float32), np.asarray(pts2, dtype=np.float32)
    if pts1.size == 0 or pts2.size == 0:
        return np.empty((0, 3))
    
    udist1 = cv.undistortPoints(pts1.reshape(-1, 1, 2), calib['K1'], calib['D1'], P=calib['K1']) 
    udist2 = cv.undistortPoints(pts2.reshape(-1, 1, 2), calib['K2'], calib['D2'], P=calib['K2'])
    
    P1 = calib['K1'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = calib['K2'] @ np.hstack((calib['R'], calib['T']))
    
    points_4d_h = cv.triangulatePoints(P1, P2, udist1.reshape(-1, 2).T, udist2.reshape(-1, 2).T)
    return (points_4d_h[:3] / points_4d_h[3]).T

def localFilter(pcd, s):
    ''' Accumulation phase lightweight filtering. '''
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=s.Stats.nbNeighbors, std_ratio=s.Stats.stdRatio)
    pcd, _ = pcd.remove_radius_outlier(nb_points=s.Radius.nbPoints, radius=s.Radius.radius)
    return pcd

def globalFilter(pcd, s):
    ''' Preprocessing: Positional cropping, Voxel downsampling, Outlier removal, and PCA filtering. '''
    if getattr(s.Positional, 'enabled', False):
        points = np.asarray(pcd.points)
        keep_mask = np.ones(len(points), dtype=bool)
        for zone in s.Positional.exclusion_zones:
            bbox = o3d.geometry.AxisAlignedBoundingBox(np.array(zone[0]), np.array(zone[1]))
            idx_inside = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
            keep_mask[idx_inside] = False
        pcd = pcd.select_by_index(np.where(keep_mask)[0])
        
    if s.Size.voxelSize > 0:
        pcd = pcd.voxel_down_sample(voxel_size=s.Size.voxelSize)
        
    if getattr(s.Stats, 'enabled', False):
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=s.Stats.nbNeighbors, std_ratio=s.Stats.stdRatio)
        
    if getattr(s.Radius, 'enabled', False):
        pcd, _ = pcd.remove_radius_outlier(nb_points=s.Radius.nbPoints, radius=s.Radius.radius)
        
    if getattr(s.PCA, 'enabled', False):
        pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(radius=s.PCA.search_radius, max_nn=s.PCA.max_nn))
        covariances = np.asarray(pcd.covariances)
        eigenvalues, _ = np.linalg.eigh(covariances)
        surface_variation = eigenvalues[:, 0] / (np.sum(eigenvalues, axis=1) + 1e-6)
        pcd = pcd.select_by_index(np.where(surface_variation < s.PCA.threshold)[0])
        
    if len(pcd.points) > 0:
        eps_val = s.Size.voxelSize * 5.0 if s.Size.voxelSize > 0 else 10.0
        labels = np.array(pcd.cluster_dbscan(eps=eps_val, min_points=20, print_progress=False))
        if len(labels) > 0:
            unique_labels, counts = np.unique(labels, return_counts=True)
            valid_mask = unique_labels >= 0
            if np.any(valid_mask):
                largest_cluster_label = unique_labels[valid_mask][np.argmax(counts[valid_mask])]
                pcd = pcd.select_by_index(np.where(labels == largest_cluster_label)[0])
                
    if len(pcd.points) == 0:
        print(colored("ERROR: Empty cloud after global filters.", "red")) 
    return pcd

def meshEngine(pcd, s):
    ''' Generates, cleans, and optimally smooths the mesh from a filtered point cloud. '''
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=s.NormalOrient.radius, max_nn=s.NormalOrient.max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=s.NormalOrient.max_nn)
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=s.Poisson.depth, width=s.Poisson.width, scale=s.Poisson.scale, linear_fit=s.Poisson.linear_fit)
        
    if getattr(s.Poisson, 'density_trim', 0) > 0:
        density_threshold = np.quantile(np.asarray(densities), s.Poisson.density_trim * 0.1)
        mesh.remove_vertices_by_mask(np.asarray(densities) < density_threshold)
        
    if getattr(s.Cleaning, 'max_distance_from_source', 0) > 0:
        mesh_vertices_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh.vertices))
        dists = np.asarray(mesh_vertices_pcd.compute_point_cloud_distance(pcd))
        mesh.remove_vertices_by_mask(dists > s.Cleaning.max_distance_from_source)
        
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    if len(cluster_n_triangles) > 0:
        mesh.remove_triangles_by_mask(np.asarray(triangle_clusters) != np.asarray(cluster_n_triangles).argmax())
        mesh.remove_unreferenced_vertices()

    if len(mesh.triangles) == 0:
        print(colored("ERROR: Empty mesh after cleaning.", "red"))
        return mesh, np.array([]), np.array([])

    # 1. Topological Regularization (Decimation as Isotropic proxy)
    if getattr(s.Smoothing.Regularization, 'enabled', False):
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=s.Smoothing.Regularization.target_triangles)
    
    # 2. Organic Subdivision (Loop)
    if getattr(s.Smoothing.Subdivision, 'enabled', False) and s.Smoothing.Subdivision.iterations > 0:
        if s.Smoothing.Subdivision.method == 'loop':
            mesh = mesh.subdivide_loop(number_of_iterations=s.Smoothing.Subdivision.iterations)
        else:
            mesh = mesh.subdivide_midpoint(number_of_iterations=s.Smoothing.Subdivision.iterations)

    # 3. Volume-Preserving Relaxation (Optimized Taubin)
    if getattr(s.Smoothing.Taubin, 'enabled', False) and s.Smoothing.Taubin.iterations > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=s.Smoothing.Taubin.iterations,
            lambda_filter=s.Smoothing.Taubin.lambda_filter,
            mu=s.Smoothing.Taubin.mu
        )

    mesh.compute_vertex_normals()
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    centroids = np.mean(vertices[triangles], axis=1)
    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    return mesh, centroids, areas

#%% Defining Main Function
def main(Config):    
    task_conf = Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3
    if not task_conf.General.Activation:
        print('.... Task3:', colored('Offline ⚠️', 'yellow'))
        return

    print('.... Task3:', colored('Running ℹ️', 'cyan'))
    main_root = Path(Config.Paths.mainRooot)
    base_path = main_root / Config.Paths.DataRoots.ResourcesRoot / Config.Paths.DataRoots.StreamRoot / Config.Paths.DataRoots.CaseStudyRoot() / Config.Packages.Drivers.__name__
    
    stereoRoot = base_path / Config.Packages.Drivers.Phases.Phase2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module1.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task1.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task1.MetaData.OutputExt
    poseRoot = base_path / Config.Packages.Drivers.Phases.Phase2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2.MetaData.OutputExt
    calibRoot = base_path / Config.Packages.Drivers.Phases.Phase1.__name__ / Config.Packages.Drivers.Phases.Phase1.Modules.Module3.__name__ / Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.__name__ / Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.MetaData.OutputExt
    
    dstRoot = base_path / Config.Packages.Drivers.Phases.Phase2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / task_conf.__name__ / task_conf.MetaData.OutputExt
    meshRoot = base_path / Config.Packages.Drivers.Phases.Phase2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / task_conf.__name__ / task_conf.MetaData.MeshExt
    
    settings = task_conf.Settings
    calib = pd.read_pickle(calibRoot)[settings.Calib.Dataset][settings.Calib.Pair][settings.Calib.Model]
    poses = pd.read_pickle(poseRoot)['BackwardSync']
    
    try:
        stereo_data = pd.read_pickle(stereoRoot)
        valid_phases = set(p for bounds in settings.Bounds for p in range(bounds[0], bounds[1]))
        all_points = []

        # MEMORY FIX: Only triangulate and process phases within requested Bounds
        for phase, group in stereo_data.groupby('Phase'):
            if phase not in valid_phases:
                continue
                
            pts_list = [triangulationEngine(np.array(row['Camera1']), np.array(row['Camera2']), calib) for _, row in group.iterrows()]
            pts_array = np.concatenate(pts_list) if pts_list else np.empty((0, 3))
            
            if pts_array.size == 0: 
                continue
                
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_array))
            filtered_pcd = localFilter(pcd, settings.LocalFilter)
            filtered_pcd.transform(poses[phase])
            all_points.append(filtered_pcd)

        if not all_points:
            raise ValueError("No points found in the specified Bounds.")

        sync = o3d.geometry.PointCloud()
        for p in all_points:
            sync += p
            
        sync = sync.voxel_down_sample(voxel_size=settings.LocalFilter.Size.voxelSize)
        globalPcd = globalFilter(sync, settings.GlobalFilter)
        
        mesh, centroids, areas = meshEngine(globalPcd, settings.Mesh)
        o3d.io.write_triangle_mesh(str(meshRoot), mesh)
        pickle.dump({'areas': areas, 'pts': centroids}, open(dstRoot, 'wb'))
        
        print('.... Task3:', colored('Executed ✅', 'green'))
        
    except Exception as e:
        print('.... Task3:', colored(f'Error: {e} ❌', 'red'))
        raise e