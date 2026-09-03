#%% Importing Libraries
from pathlib import Path
import pandas as pd
import numpy as np
import open3d as o3d
import cv2 as cv
from scipy.spatial import cKDTree
from scipy.signal import butter, filtfilt, savgol_filter
from tqdm import tqdm
from termcolor import colored

#%% Defining Subroutines
def kinematic(values, name, config):
    vs = lambda l: np.unique(np.vstack(l), axis=0) if len(l) else np.array([])
    pose = vs([a for a in values.get('Pose', pd.Series()).dropna() if isinstance(a, (list, np.ndarray)) and len(a)])
    inner = vs([a['Inner'] for a in values.get(name, pd.Series()).dropna() if isinstance(a, dict) and 'Inner' in a and len(a['Inner'])])
    outer = vs([a['Mesh'] for a in values.get(name, pd.Series()).dropna() if isinstance(a, dict) and 'Mesh' in a and len(a['Mesh'])])
    if len(pose) == 0 or len(inner) < 3 or len(outer) == 0: return np.nan, np.nan, np.nan, np.nan
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inner)
        pcd_tiles = pcd.voxel_down_sample(getattr(config, 'TileSize', 0.01))
        if len(pcd_tiles.points) < 3: pcd_tiles = pcd
        pcd_tiles.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=getattr(config, 'TileSize', 0.01)*2.5, max_nn=getattr(config, 'Max_nn', 30)))
        pcd_tiles.orient_normals_towards_camera_location(np.array(getattr(config, 'CameraPose', [0,0,0])))
        _, tile_indices = cKDTree(np.asarray(pcd_tiles.points)).query(inner, k=1) 
        normals, poseC, innerC, outerC = np.asarray(pcd_tiles.normals)[tile_indices], np.mean(pose, axis=0), np.mean(inner, axis=0), np.mean(outer, axis=0)
        _, idx = cKDTree(inner).query(poseC, k=1)
        nearest_surface_point = inner[idx]
        reference = normals[idx] / (np.linalg.norm(normals[idx]) + 1e-10)
        distNorm = np.abs(np.dot(poseC - nearest_surface_point, reference))
        return np.linalg.norm(poseC - innerC), np.linalg.norm(poseC - outerC), np.linalg.norm(innerC - outerC), distNorm
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

def optical(img1, img2):
    def imgData(img):
        mask = img > 0
        area = np.sum(mask)
        if area == 0: return 0, 0, np.zeros(256), 0, 0 
        pixels = img[mask]
        intensity, (hist, _) = np.sum(pixels), np.histogram(pixels, bins=256, range=(0, 255))
        magnitudo = np.sqrt(cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)**2 + cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)**2)
        erode = cv.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1).astype(bool)
        return area, intensity, hist, np.sum(magnitudo[erode]) if np.any(erode) else np.sum(magnitudo[mask]), np.sum(erode) if np.any(erode) else area
    
    a1, si1, hist1, sg1, ag1 = imgData(img1)
    a2, si2, hist2, sg2, ag2 = imgData(img2)    
    areaTot = a1 + a2
    
    if areaTot == 0: return np.nan, np.nan, np.nan
    intensityTot = (si1 + si2) / areaTot
    probability = (hist1 + hist2) / areaTot
    probability = probability[probability > 0] 
    entropyTot = -np.sum(probability * np.log2(probability)) if len(probability) > 0 else np.nan
    gradientTot = (sg1 + sg2) / (ag1 + ag2) if (ag1 + ag2) > 0 else np.nan
    return intensityTot, entropyTot, gradientTot

def compute(Ds, dt, targets=['Area', 'Volume'], conf=None):
    def apply(df, group_col, targets, config, dt):
        out = df.copy()
        
        def smooth(x, config):
            s = pd.Series(np.asarray(x, dtype=float)).interpolate(method='linear', limit_direction='both')
            x_arr = s.to_numpy()
            if len(x_arr) < 5 or np.isnan(x_arr).all(): 
                return x_arr
            b, a = butter(getattr(config, 'Order', 3), getattr(config, 'Cutoff', 0.05), btype='low')
            return filtfilt(b, a, x_arr, padlen=min(max(3, int(len(x_arr) * getattr(config, 'PadPerc', 0.10))), len(x_arr) - 1))
        def derive(x, dt, config=None):
            x_arr = np.asarray(x, dtype=float)
            if np.isnan(x_arr).all() or len(x_arr) < 5: 
                return np.gradient(x_arr, dt, edge_order=getattr(config, 'EdgeOrder', 2) if config else 2) if len(x_arr) > 1 else np.zeros_like(x_arr)
            window = getattr(config, 'SavgolWindow', 11) 
            window = min(window, len(x_arr) - 1 if len(x_arr) % 2 == 0 else len(x_arr) - 2)
            window = max(3, window)
            window = int(window) | 1 
            poly = getattr(config, 'SavgolPoly', 3)
            poly = min(poly, window - 1)
            try:
                return savgol_filter(x_arr, window_length=window, polyorder=poly, deriv=1, delta=dt)
            except Exception:
                return np.gradient(x_arr, dt, edge_order=getattr(config, 'EdgeOrder', 2) if config else 2)

        for t in [t for t in targets if t in df.columns]:
            if group_col and group_col in df.columns:
                out[f'{t}D0'] = df.groupby(group_col)[t].transform(lambda x: smooth(x, config))
                out[f'{t}D1'] = out.groupby(group_col)[f'{t}D0'].transform(lambda x: derive(x, dt, config))
                out[f'{t}D2'] = out.groupby(group_col)[f'{t}D1'].transform(lambda x: derive(x, dt, config))
            else:
                out[f'{t}D0'] = smooth(df[t], config)
                out[f'{t}D1'] = derive(out[f'{t}D0'], dt, config)
                out[f'{t}D2'] = derive(out[f'{t}D1'], dt, config)
        return out
    
    return apply(Ds[0], 'Event', targets, conf, dt), apply(Ds[1], 'Blade', targets, conf, dt), apply(Ds[2], 'Prop', targets, conf, dt)

def processor(data, files, name, Config):
    settings = getattr(Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task2, 'Settings', None)
    c_norm, c_smooth = getattr(settings, 'Normals', None), getattr(settings, 'Smoothing', None)
    ppr, blade_num = getattr(Config.Settings.Acquisition, 'PPR', 360), getattr(Config.Settings.Acquisition, 'Blades', 4)
    analysis_settings = getattr(settings, 'Analysis', None)
    dt = getattr(analysis_settings, 'dt', 0.00015625)
    blade_period, records = ppr / blade_num, []
    safe_sum = lambda s, k: np.sum([x[k] for x in s.dropna() if isinstance(x, dict) and k in x and not pd.isna(x[k])]) if not s.empty else 0.0
    
    def getMask(df_masks, camera_name):
        """Estrae le maschere in sicurezza evitando errori su liste vuote o celle NaN"""
        if camera_name not in df_masks.columns:
            return np.zeros((2, 2))
        valid_cells = [
            cell[0] for cell in df_masks[camera_name].dropna() 
            if isinstance(cell, (list, tuple, np.ndarray)) and len(cell) > 0]
        return np.maximum.reduce(valid_cells) if valid_cells else np.zeros((2, 2))
    
    for k, v in tqdm(data.items(), total=len(data), desc=colored(f'Data Analysis {name} 🚀', 'magenta'), ncols=100):
        k_int, cav = int(k), v.get(name, pd.Series(dtype=object))
        phase_p, phase_b = k_int % ppr, k_int % blade_period
        areaC, volumeC, distRel, distNorm = kinematic(v, name, c_norm)
        mask_path = next((path for path, id in files if id == int(k)), None)
        masks = pd.read_pickle(mask_path) if mask_path else pd.DataFrame()
        img1 = getMask(masks, 'Camera1')
        img2 = getMask(masks, 'Camera2')
        whiteness, entropy, gradient = optical(img1, img2)
        records.append({'Index': k, 'Prop': 0, 'Blade': int(phase_p // blade_period) + 1, 'Event': int(k_int // blade_period),
                        'DegP': phase_p * (360.0 / ppr), 'DegB': phase_b * (360.0 / ppr), 'PhaseP': phase_p, 'PhaseB': phase_b,
                        'Area': safe_sum(cav, 'Area'), 'Volume': safe_sum(cav, 'Volume'),
                        'AreaC': areaC, 'VolumeC': volumeC, 'DistRel': distRel, 'DistNorm': distNorm,
                        'Whiteness': whiteness, 'Entropy': entropy, 'Gradient': gradient})  
    
    DATAraw = pd.DataFrame(records).set_index('Index')
    return compute([DATAraw, 
        DATAraw.drop(columns=['DegP']).groupby(DATAraw['DegP'].round(4)).median(numeric_only=True).reset_index(),
        DATAraw.drop(columns=['DegB']).groupby(DATAraw['DegB'].round(4)).median(numeric_only=True).reset_index()],
        dt, ['Area', 'Volume', 'AreaC', 'VolumeC', 'DistRel', 'DistNorm', 'Whiteness', 'Entropy', 'Gradient'], conf=c_smooth)

#%% Defining Main Function
def main(Config):       
    task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task2
    if task_conf.General.Activation is True:
        print('.... Task2:', colored('Running ℹ️', 'cyan'))
        base = Path(Config.Paths.mainRooot) / Config.Paths.DataRoots.ResourcesRoot / Config.Paths.DataRoots.StreamRoot / Config.Paths.DataRoots.CaseStudyRoot() / Config.Packages.Drivers.__name__ / Config.Packages.Drivers.Phases.Phase3.__name__
        m1, m2 = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__, Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__
        t1, t2 = Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.__name__, Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task2.__name__
        ext = Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.MetaData.OutputExt
        srcRoot = base / m2 / t1 / ext
        maskRoot = base / m1 / Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task2.__name__
        dstRoot = base / m2 / t2 / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task2.MetaData.OutputExt
        data = pd.read_pickle(srcRoot)
        files = [(str(f), int(f.stem.split('_')[1])) for f in maskRoot.iterdir()]
        CAVraw, CAVprp, CAVbld = processor(data, files, 'Cavity', Config)
        CLDraw, CLDprp, CLDbld = processor(data, files, 'Cloud', Config)
        pd.to_pickle({'CAVraw': CAVraw, 'CAVprp': CAVprp, 'CAVbld': CAVbld, 'CLDraw': CLDraw, 'CLDprp': CLDprp, 'CLDbld': CLDbld}, dstRoot)
    elif task_conf.General.Activation is False:
        print('.... Task2:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task2 Switch (on/off) ❌')
    return