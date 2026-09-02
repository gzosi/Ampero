#%% Importing Libreries
from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
import open3d as o3d
from tqdm import tqdm
from termcolor import colored 
#%% Defining Subroutines
def surfaceCarver(masks: list, calib: dict, pose: np.ndarray, config) -> tuple:
    """Intaglia la Superficie dell'elica usando le maschere di occlusione."""
    filled = []
    for i, mask in enumerate(masks):
        imgH, imgW = mask.shape
        ys, xs = np.where(mask > 0)
        points = np.stack([xs, ys], axis=-1)  
        zero = np.zeros_like(mask, dtype=np.uint8)
        if points.size > 0:
            K, D = calib.get(f"K{i+1}"), calib.get(f"D{i+1}")
            undistorted = cv.undistortPoints(points.astype(np.float32).reshape(-1, 1, 2), K, D, None, K).reshape(-1, 2)
            undistorted = np.round(undistorted).astype(int)
            valid = ((undistorted[:, 0] >= 0) & (undistorted[:, 0] < imgW) & (undistorted[:, 1] >= 0) & (undistorted[:, 1] < imgH))
            undistorted = undistorted[valid]
            zero[undistorted[:, 1], undistorted[:, 0]] = 1
        poseH = np.hstack([pose, np.ones((pose.shape[0], 1))])
        P = calib['K1'] @ np.hstack((np.eye(3), np.zeros((3, 1)))) if i == 0 else calib['K2'] @ np.hstack((calib['R'], calib['T']))
        uvs = P @ poseH.T
        uvs /= uvs[2, :]
        uvs = np.round(uvs).astype(int)
        good = (uvs[0, :] >= 0) & (uvs[0, :] < imgW) & (uvs[1, :] >= 0) & (uvs[1, :] < imgH)
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        fill[indices] = zero[sub_uvs[1, :], sub_uvs[0, :]]
        filled.append(fill) 
    occupancy = np.sum(np.vstack(filled), axis=0)
    good_indices = np.where(occupancy >= config.occupancyLimit)[0]
    return good_indices, pose[good_indices]
#%% Defining Main Function
def main(Config): 
    task_conf = Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1
    if task_conf.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        poseRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.MetaData.OutputExt)
        motionRoot = (
            main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ /
            Config.Packages.Drivers.Phases.Phase2.__name__ /
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ /
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2.__name__/
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2.MetaData.OutputExt)
        objRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task2.__name__ /
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task2.MetaData.OutputExt)
        calibRoot = (
            main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase1.__name__ / 
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.__name__ /
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.__name__ / 
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.MetaData.OutputExt)
        dstRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.Modules.Module2.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.MetaData.OutputExt)
        settings = task_conf.Settings
        ppr = Config.Settings.Acquisition.PPR
        blades = Config.Settings.Acquisition.Blades
        poses, areas = [pd.read_pickle(poseRoot)[key] for key in ['single', 'areas']]
        motion = pd.read_pickle(motionRoot)['ForwardSync']
        calib = pd.read_pickle(calibRoot)[settings.Calib.Dataset][settings.Calib.Pair][settings.Calib.Model]
        objects = pd.read_pickle(objRoot)
        datasets = list(objects.keys())
        try:
            data = {}
            global_reference_poses = {} # Conserva le Pose_0 di tutte le pale per il Dataset 0
            cumulative_ids = {b: set() for b in range(1, blades + 1)}
            resync_list = settings.Resync 
            time_list = settings.Time # Recupero la lista Time dai settings
            for d_idx, dataset in enumerate(datasets):
                shift = resync_list[d_idx]
                current_time = time_list[d_idx] # Tempo corrente per questo dataset
                reference_poses = {} 
                for key in tqdm(list(objects[dataset].keys()), desc=colored(f'Surface Carving {dataset} 🚀', 'magenta'), ncols=100): 
                    phase = int(key) % ppr 
                    raw_blade = int((phase // (ppr / blades)) + 1)
                    actual_blade = int(((raw_blade - 1 + shift) % blades) + 1)
                    pose, masks = poses[phase], objects[dataset][key]
                    if actual_blade not in reference_poses:
                        moved = np.asarray(
                            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pose))
                            .transform(np.linalg.inv(motion[phase])).points)
                        reference_poses[actual_blade] = moved
                        global_reference_poses[actual_blade] = moved # Salva globalmente per il Dataset_0
                    for _, (mask1, mask2) in masks.iterrows():
                        id_res, _ = surfaceCarver([mask1, mask2], calib, pose, settings.Carver)
                        if id_res is not None and id_res.size: 
                            cumulative_ids[actual_blade].update(id_res.tolist())
                results = []
                for b in range(1, blades + 1):
                    indices = list(cumulative_ids[b])
                    if b in reference_poses:
                        ref_pose = reference_poses[b]
                        damage_pts = ref_pose[indices] if indices else np.array([])
                    else:
                        ref_pose = np.array([])
                        damage_pts = np.array([])
                    results.append({
                        'Blade': b,
                        'Time': current_time, 
                        'Pose': ref_pose,
                        'Damage': damage_pts,
                        'Area': sum(areas[i] for i in indices) if indices else 0.0})
                data[dataset] = pd.DataFrame(results).set_index('Blade')
            results_0 = []
            for b in range(1, blades + 1):
                ref_pose = global_reference_poses.get(b, np.array([]))
                results_0.append({
                    'Blade': b,
                    'Time': 0.0,
                    'Pose': ref_pose,
                    'Damage': np.array([]), 
                    'Area': 0.0              
                })
            final_data = {'Dataset0': pd.DataFrame(results_0).set_index('Blade')}
            final_data.update(data)
            pickle.dump(final_data, open(dstRoot, 'wb'))
            print('.... Task1:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return