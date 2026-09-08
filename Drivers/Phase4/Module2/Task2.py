from pathlib import Path
import warnings
import pickle
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import mutual_info_score
from tqdm.auto import tqdm
from termcolor import colored
def sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))
def main(Config):
    task_conf = Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task2
    if task_conf.General.Activation is not True:
        if task_conf.General.Activation is False:
            print('.... Task2:', colored('Offline ⚠️', 'yellow'))
            return
        raise ValueError('Please Set the Task2 Switch (on/off) ❌')
    print('.... Task2:', colored('Running ℹ️', 'cyan'))
    mainRoot = Path(Config.Paths.mainRooot)
    base = mainRoot / Config.Paths.DataRoots.ResourcesRoot / Config.Paths.DataRoots.StreamRoot / Config.Paths.DataRoots.CaseStudyRoot() / Config.Packages.Drivers.__name__
    damageRoot = base / Config.Packages.Drivers.Phases.Phase4.__name__ / Config.Packages.Drivers.Phases.Phase4.Modules.Module2.__name__ / Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.__name__ / Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.MetaData.OutputExt
    energyRoot = base / Config.Packages.Drivers.Phases.Phase3.__name__ / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task3.__name__ / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task3.MetaData.OutputExt
    poseRoot = base / Config.Packages.Drivers.Phases.Phase2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.__name__ / Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.MetaData.OutputExt
    dstRoot = base / Config.Packages.Drivers.Phases.Phase4.__name__ / Config.Packages.Drivers.Phases.Phase4.Modules.Module2.__name__ / Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task2.__name__ / Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task2.MetaData.OutputExt
    damages = pd.read_pickle(damageRoot)
    energies = pd.read_pickle(energyRoot)
    areas = pd.read_pickle(poseRoot)['areas']
    try:
        geom_cache = {}
        for dataset, damageDF in damages.items():
            geom_cache[dataset] = []
            for row in damageDF.itertuples():
                pose = np.array(row.Pose, dtype=float)
                damageRAW = row.Damage
                mask = np.zeros(len(pose), dtype=int)
                if len(damageRAW) > 0:
                    tree = cKDTree(pose)
                    _, idx = tree.query(np.array(damageRAW, dtype=float), k=1)
                    mask[np.unique(idx)] = 1
                innerID = np.where(mask == 1)[0]
                outerID = np.where(mask == 0)[0]
                geom_cache[dataset].append({'Blade': getattr(row, 'Blade', row.Index), 'Area': row.Area, 'Time': getattr(row, 'Time', np.nan), 'innerID': innerID, 'outerID': outerID, 'num_pts': len(pose)})
            if 'Time' in damageDF.columns:
                for time_val, df_t in damageDF.groupby('Time'):
                    masks, areas_list, pose = [], [], None
                    for row in df_t.itertuples():
                        if pose is None: pose = np.array(row.Pose, dtype=float)
                        areas_list.append(row.Area)
                        damageRAW = row.Damage
                        mask = np.zeros(len(pose), dtype=int)
                        if len(damageRAW) > 0:
                            tree = cKDTree(pose)
                            _, idx = tree.query(np.array(damageRAW, dtype=float), k=1)
                            mask[np.unique(idx)] = 1
                        masks.append(mask)
                    if len(masks) > 0:
                        median_mask = (np.median(np.vstack(masks), axis=0) >= 0.5).astype(int)
                        innerID = np.where(median_mask == 1)[0]
                        outerID = np.where(median_mask == 0)[0]
                        geom_cache[dataset].append({'Blade': 'Median', 'Area': np.median(areas_list), 'Time': time_val, 'innerID': innerID, 'outerID': outerID, 'num_pts': len(pose)})
        levels = {}
        for name, dt in tqdm(energies.items(), desc="Ciclo Energies"):
            if name not in levels: levels[name] = {}
            for path in tqdm(task_conf.Settings.Categories, desc=f"Categories per {name}", leave=False):
                try:
                    current = dt
                    for step in path: current = current[step]
                except (KeyError, IndexError, TypeError): continue
                energy = np.linalg.norm(current, axis=-1)
                datasetsDICT = {}
                for dataset, active_cache in geom_cache.items():
                    records = []
                    for geom_data in active_cache:
                        innerID, outerID = geom_data['innerID'], geom_data['outerID']
                        innerEnergy, innerArea = energy[innerID], areas[innerID]
                        outerEnergy, outerArea = energy[outerID], areas[outerID]
                        innerEnergySum, outerEnergySum = np.sum(innerEnergy), np.sum(outerEnergy)
                        innerAreaSum, outerAreaSum = np.sum(innerArea), np.sum(outerArea)
                        totalEnergy = innerEnergySum + outerEnergySum
                        totalArea = innerAreaSum + outerAreaSum
                        innerDensity = innerEnergySum / innerAreaSum if innerAreaSum > 0 else 0
                        outerDensity = outerEnergySum / outerAreaSum if outerAreaSum > 0 else 0
                        energyDelta = (innerEnergySum / totalEnergy * 100) if totalEnergy > 0 else 0
                        totalDensity = totalEnergy / totalArea if totalArea > 0 else 0
                        densityDelta = (innerDensity / totalDensity * 100) if totalDensity > 0 else 0
                        is_damaged = np.zeros(geom_data['num_pts'], dtype=int)
                        if len(innerID) > 0: is_damaged[innerID] = 1
                        bins = np.histogram_bin_edges(energy, bins='sturges')
                        mi = mutual_info_score(np.digitize(energy, bins), is_damaged)
                        if len(innerID) == 0 or len(innerID) == geom_data['num_pts']: sig_k, sig_x0, sig_mse = np.nan, np.nan, np.nan
                        else:
                            e_min, e_max = energy.min(), energy.max()
                            e_range = e_max - e_min if e_max > e_min else 1e-10
                            energy_norm = (energy - e_min) / e_range
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", OptimizeWarning)
                                try:
                                    popt, _ = curve_fit(sigmoid, energy_norm, is_damaged, p0=[10.0, 0.5], maxfev=2000)
                                    sig_x0 = popt[1] * e_range + e_min
                                    sig_k = popt[0] / e_range
                                    sig_mse = np.mean((is_damaged - sigmoid(energy, sig_k, sig_x0))**2)
                                except Exception: sig_k, sig_x0, sig_mse = np.nan, np.nan, np.nan
                        records.append({'Blade': geom_data['Blade'], 'Area': geom_data['Area'], 'Time': geom_data['Time'], 'energyDelta': energyDelta, 'densityDelta': densityDelta, 'mutualInfo': mi, 'sigmoid_k': sig_k, 'sigmoid_x0': sig_x0, 'sigmoid_mse': sig_mse})
                    datasetsDICT[dataset] = pd.DataFrame(records)
                node = levels[name]
                for step in path[:-1]:
                    if step not in node: node[step] = {}
                    node = node[step]
                node[path[-1]] = datasetsDICT
        with open(dstRoot, 'wb') as f: pickle.dump(levels, f)
        print('.... Task2:', colored('Executed ✅', 'green'))
    except Exception as e:
        print('.... Task2:', colored(f'Error: {e} ❌', 'red'))
        raise e
    return