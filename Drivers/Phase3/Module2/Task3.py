from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
import gc
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import cKDTree
import open3d as o3d
from tqdm import tqdm
from termcolor import colored

def getImgData(mask): 
    mask_u8 = img_as_ubyte(mask) if mask.dtype != np.uint8 else mask    
    grad_x, grad_y = cv.Sobel(mask, cv.CV_32F, 1, 0, ksize=3), cv.Sobel(mask, cv.CV_32F, 0, 1, ksize=3)
    return mask.astype(np.float32), cv.magnitude(grad_x, grad_y), entropy(mask_u8, disk(1)).astype(np.float32)

def dataAssignment(masks1, masks2, inner, calib): 
    gray1, grad1, ent1 = masks1
    gray2, grad2, ent2 = masks2
    h, w = gray1.shape
    val_gray, val_grad, val_ent = (np.zeros((len(inner), 2), dtype=np.float32) for _ in range(3))
    pts1 = np.round(cv.projectPoints(inner, np.zeros((3,1)), np.zeros((3,1)), calib['K1'], calib['D1'])[0]).astype(int).reshape(-1, 2)
    pts2 = np.round(cv.projectPoints(inner, cv.Rodrigues(calib['R'])[0], calib['T'], calib['K2'], calib['D2'])[0]).astype(int).reshape(-1, 2)
    u1, v1, u2, v2 = pts1[:,0], pts1[:,1], pts2[:,0], pts2[:,1]
    valid1 = (u1 >= 0) & (u1 < w) & (v1 >= 0) & (v1 < h)
    valid2 = (u2 >= 0) & (u2 < w) & (v2 >= 0) & (v2 < h)
    val_gray[valid1, 0], val_gray[valid2, 1] = gray1[v1[valid1], u1[valid1]], gray2[v2[valid2], u2[valid2]]
    val_grad[valid1, 0], val_grad[valid2, 1] = grad1[v1[valid1], u1[valid1]], grad2[v2[valid2], u2[valid2]]
    val_ent[valid1, 0], val_ent[valid2, 1] = ent1[v1[valid1], u1[valid1]], ent2[v2[valid2], u2[valid2]]
    return val_gray, val_grad, val_ent

def compute(Ds, targets, conf=None):
    def apply(df_inner, targets):
        def get_X(df_data, name):
            return np.stack(df_data[name].values).astype(np.float32).reshape(len(df_data), -1)
        def time_stats(df_data, name, dt):
            X = get_X(df_data, name)
            v, rms = np.var(X, axis=0, dtype=np.float32), np.sqrt(np.mean(X**2, axis=0, dtype=np.float32))
            dX_dt = np.diff(X, axis=0) / dt
            pv, prms = np.var(dX_dt, axis=0, dtype=np.float32), np.sqrt(np.mean(dX_dt**2, axis=0, dtype=np.float32))
            del X, dX_dt
            gc.collect() 
            return {'variance': v.reshape(-1, 2), 'rms': rms.reshape(-1, 2), 'power_variance': pv.reshape(-1, 2), 'power_rms': prms.reshape(-1, 2)}
        def pod(df_data, name, cfg, dt):
            X = get_X(df_data, name)
            T_steps = X.shape[0]
            X_mean = np.mean(X, axis=0)
            X -= X_mean
            svd = TruncatedSVD(n_components=getattr(cfg, 'n_modes', 25), algorithm='randomized', random_state=42)
            C_full = svd.fit_transform(X) 
            valid = svd.explained_variance_ > getattr(cfg, 'eig_th', 1e-10)
            if not np.any(valid): valid[0] = True 
            C, eig, en_pct, V = C_full[:, valid], svd.explained_variance_[valid], (svd.explained_variance_ratio_ * 100)[valid], svd.components_[valid, :]
            spatial_energy = np.sum((V ** 2) * eig[:, np.newaxis], axis=0)
            del X
            gc.collect()                
            return {'mean_field': X_mean.reshape(-1, 2), 'spatial_modes': [m for m in V.reshape(np.sum(valid), -1, 2)], 'energy_ratio': en_pct, 'time_coeffs': C, 'spatial_energy_map': spatial_energy.reshape(-1, 2), 'power_map': (spatial_energy / (T_steps * dt)).reshape(-1, 2)}
        out_dict = {}
        for t in [t for t in targets if t in df_inner.columns]:    
            stats = time_stats(df_inner, t, conf.Settings.Analysis.dt)
            out_dict.update({f'{t}_TimeVar': stats['variance'], f'{t}_RMS': stats['rms'], f'{t}_PowerVar': stats['power_variance'], f'{t}_PowerRMS': stats['power_rms'], f'{t}_POD': pod(df_inner, t, conf.Settings.Analysis.POD, conf.Settings.Analysis.dt)})
        return out_dict
    return apply(Ds[0], targets), apply(Ds[1], targets), apply(Ds[2], targets)
def processor(data, files, motion, pose, calib, name, Config):
    ppr, blade_num = Config.Settings.Acquisition.PPR, Config.Settings.Acquisition.Blades
    blade_period, records, pose_tree = ppr / blade_num, [], cKDTree(pose)
    def getMask(df_masks, cam):
        if df_masks.empty or cam not in df_masks.columns: return None
        v = [c[0] for c in df_masks[cam].dropna() if isinstance(c, (list, tuple, np.ndarray)) and len(c) > 0]
        return np.maximum.reduce(v) if v else None
    for k, v in tqdm(data.items(), total=len(data), desc=colored(f'Data Analysis {name} 🚀', 'magenta'), ncols=100):
        k_int, p_p, p_b = int(k), int(k) % ppr, int(k) % blade_period
        fp = next((p for p, i in files if i == k_int), None)
        masks = pd.read_pickle(fp) if fp else pd.DataFrame()
        m1, m2 = getMask(masks, 'Camera1'), getMask(masks, 'Camera2')
        i_list = [a['Inner'] for a in v.get(name, pd.Series()).dropna() if isinstance(a, dict) and 'Inner' in a and len(a['Inner'])]
        inner = np.unique(np.vstack(i_list), axis=0) if len(i_list) else np.array([])
        b_rec = {'Index': k, 'Prop': 0, 'Blade': int(p_p // blade_period) + 1, 'Event': int(k_int // blade_period), 'DegP': p_p * (360.0 / ppr), 'DegB': p_b * (360.0 / ppr), 'PhaseP': p_p, 'PhaseB': p_b}
        if m1 is None or m2 is None or inner.size == 0:
            records.append({**b_rec, 'Grayscale': np.nan, 'Gradient': np.nan, 'Entropy': np.nan})
            continue
        comp_gray, comp_grad, comp_ent = dataAssignment([*getImgData(m1)], [*getImgData(m2)], inner, calib)
        dist, idx = pose_tree.query(np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inner)).transform(np.linalg.inv(motion[p_p])).points), k=1)
        order, pose_gray, pose_grad, pose_ent = np.argsort(dist), np.zeros((len(pose), 2), dtype=np.float32), np.zeros((len(pose), 2), dtype=np.float32), np.zeros((len(pose), 2), dtype=np.float32)
        valid = dist[order] < 5.0
        u_idx, u_pos = np.unique(idx[order][valid], return_index=True)
        pose_gray[u_idx], pose_grad[u_idx], pose_ent[u_idx] = comp_gray[order][valid][u_pos], comp_grad[order][valid][u_pos], comp_ent[order][valid][u_pos]
        records.append({**b_rec, 'Grayscale': pose_gray, 'Gradient': pose_grad, 'Entropy': pose_ent})      
    cols = ['Prop', 'Blade', 'Event', 'DegP', 'DegB', 'PhaseP', 'PhaseB', 'Grayscale', 'Gradient', 'Entropy']
    if not records: return compute([pd.DataFrame(columns=['Index']+cols).set_index('Index'), pd.DataFrame(columns=['DegP']+cols), pd.DataFrame(columns=['DegB']+cols)], ['Grayscale', 'Gradient', 'Entropy'], conf=Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task3)
    D1 = pd.DataFrame(records).set_index('Index')
    for c in ['Grayscale', 'Gradient', 'Entropy']: D1[c] = D1[c].ffill().bfill()
    grp = lambda df, col: pd.DataFrame([{col: k, **{c: np.median(np.stack(df.loc[idx, c].tolist()), axis=0) for c in ['Grayscale', 'Gradient', 'Entropy']}} for k, idx in df.groupby(col).groups.items()])
    return compute([D1, grp(D1, 'DegP'), grp(D1, 'DegB')], ['Grayscale', 'Gradient', 'Entropy'], conf=Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task3)
def main(Config):       
    tc = Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task3
    if tc.General.Activation is False: return print('.... Task3:', colored('Offline ⚠️', 'yellow'))
    elif tc.General.Activation is not True: raise ValueError('Please Set the Task3 Switch (on/off) ❌')
    print('.... Task3:', colored('Running ℹ️', 'cyan'))
    mr, dr, st, cs = Path(Config.Paths.mainRooot), Config.Paths.DataRoots.ResourcesRoot, Config.Paths.DataRoots.StreamRoot, Config.Paths.DataRoots.CaseStudyRoot()
    base = mr / dr / st / cs / Config.Packages.Drivers.__name__
    p1, p2, p3 = base / Config.Packages.Drivers.Phases.Phase1.__name__, base / Config.Packages.Drivers.Phases.Phase2.__name__, base / Config.Packages.Drivers.Phases.Phase3.__name__
    data = pd.read_pickle(p3 / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.__name__ / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.MetaData.OutputExt)
    maskRoot = p3 / Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__ / Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task2.__name__
    pose = pd.read_pickle(p2 / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.MetaData.OutputExt)['pts']
    motion = pd.read_pickle(p2 / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2.MetaData.OutputExt)['ForwardSync']
    calib = pd.read_pickle(p1 / Config.Packages.Drivers.Phases.Phase1.Modules.Module3.__name__ / Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.__name__ / Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.MetaData.OutputExt)[tc.Settings.Calib.Dataset][tc.Settings.Calib.Pair][tc.Settings.Calib.Model]
    files = [(str(f), int(f.stem.split('_')[1])) for f in maskRoot.iterdir()]
    pd.to_pickle({'CAVraw': (c:=processor(data, files, motion, pose, calib, 'Cavity', Config))[0], 'CAVprp': c[1], 'CAVbld': c[2], 'CLDraw': (cl:=processor(data, files, motion, pose, calib, 'Cloud', Config))[0], 'CLDprp': cl[1], 'CLDbld': cl[2]}, p3 / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task3.__name__ / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task3.MetaData.OutputExt)