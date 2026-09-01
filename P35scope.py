import pyvista as pv
import numpy as np
import pandas as pd
from pathlib import Path
from Config import Config

# =============================================================================
# 1. FUNZIONE DI RENDERING 3D (PyVista)
# =============================================================================
def plot_3d_cloud(pts, map_data, title="Mappa 3D", cmap="plasma", is_interactive=True):
    """
    Rendering nuvola punti in 3D con PyVista.
    map_data: array di forma (N, 2) contenente i dati delle due telecamere.
    """
    if map_data is None:
        print(f"  -> Dati mancanti per {title}, salto il rendering.")
        return

    # Calcolo della magnitudo (modulo) combinando i dati delle due camere
    map_array = np.array(map_data)
    if map_array.ndim == 2 and map_array.shape[1] == 2:
        mag = np.linalg.norm(map_array, axis=1)
    else:
        mag = map_array.flatten()
    
    # Creazione dell'oggetto PolyData (Mesh di punti)
    cloud = pv.PolyData(pts)
    
    # Taglio degli outlier per un contrasto visivo migliore (1° e 99° percentile)
    p_min, p_max = np.percentile(mag, 1), np.percentile(mag, 99)
    # Se il campo è completamente uniforme (es. tutti zeri), evitiamo errori di divisione
    if p_min == p_max: p_max = p_min + 1e-5 
    
    cloud['Modulo'] = np.clip(mag, p_min, p_max)
    
    # Configurazione del Plotter
    p = pv.Plotter()
    args = dict(
        title=title, 
        title_font_size=24, 
        label_font_size=18, 
        shadow=True, 
        n_labels=5, 
        fmt="%.3e", 
        color="black"
    )
    
    p.set_background("white")
    
    p.add_mesh(
        cloud, 
        scalars='Modulo', 
        cmap=cmap, 
        render_points_as_spheres=True, 
        point_size=8, 
        scalar_bar_args=args, 
        show_scalar_bar=True
    )
    
    p.enable_eye_dome_lighting() 
    
    if is_interactive:
        p.show() 
    else:
        return p

# =============================================================================
# 2. LOGICA DI ESTRAZIONE E VISUALIZZAZIONE
# =============================================================================
def inspect_modal_3d(data_dict, pts, stage='prp', entity='CAV', feature='Grayscale', n_modes=3):
    """
    Estrae i dati modali dal dizionario (incluse le nuove quantità) e lancia il render 3D.
    """
    key = f'{entity}{stage}'
    print(f"\n--- Ispezione 3D Modale: {key} | {feature} ---")
    
    if key not in data_dict:
        print(f"Errore: Chiave {key} non trovata nel dizionario.")
        return
        
    df_results = data_dict[key]
    
    # --- 1. STATISTICHE TEMPORALI ---
    print("\n[1/2] Estrazione Statistiche Temporali e di Potenza...")
    
    timevar_key = f'{feature}_TimeVar'
    if timevar_key in df_results:
        plot_3d_cloud(pts, df_results[timevar_key], title=f"{feature} - Time Variance", cmap="viridis")

    rms_key = f'{feature}_RMS'
    if rms_key in df_results:
        plot_3d_cloud(pts, df_results[rms_key], title=f"{feature} - RMS", cmap="viridis_r")

    powervar_key = f'{feature}_PowerVar'
    if powervar_key in df_results:
        plot_3d_cloud(pts, df_results[powervar_key], title=f"{feature} - Power Variance (Derivata)", cmap="magma")

    powerrms_key = f'{feature}_PowerRMS'
    if powerrms_key in df_results:
        plot_3d_cloud(pts, df_results[powerrms_key], title=f"{feature} - Power RMS (Derivata)", cmap="magma_r")

    # --- 2. ANALISI POD ---
    print("\n[2/2] Estrazione Analisi POD (SVD)...")
    pod_key = f'{feature}_POD'
    if pod_key in df_results:
        pod_data = df_results[pod_key]
        
        if 'mean_field' in pod_data:
            plot_3d_cloud(pts, pod_data['mean_field'], title=f"{feature} - POD Mean Field", cmap="cividis")
            
        if 'spatial_energy_map' in pod_data:
            plot_3d_cloud(pts, pod_data['spatial_energy_map'], title=f"{feature} - POD Spatial Energy", cmap="inferno")
            
        if 'power_map' in pod_data:
            plot_3d_cloud(pts, pod_data['power_map'], title=f"{feature} - POD Power Map", cmap="turbo")
            
        if 'spatial_modes' in pod_data:
            modes = pod_data['spatial_modes']
            for i in range(min(n_modes, len(modes))):
                plot_3d_cloud(pts, modes[i], title=f"{feature} - POD Spatial Mode {i+1}", cmap="plasma")
    else:
        print(f"Analisi POD ({pod_key}) non trovata per questo set di dati.")

# =============================================================================
# 3. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # --- Costruzione dei Path basata su Config ---
    main_root = Path(Config.Paths.mainRooot)
    
    pose_root = (main_root /
        Config.Paths.DataRoots.ResourcesRoot /
        Config.Paths.DataRoots.StreamRoot / 
        Config.Paths.DataRoots.CaseStudyRoot() /
        Config.Packages.Drivers.__name__ /
        Config.Packages.Drivers.Phases.Phase2.__name__ /
        Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ /
        Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.__name__/
        Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.MetaData.OutputExt)
    
    data_root = (main_root /
        Config.Paths.DataRoots.ResourcesRoot /
        Config.Paths.DataRoots.StreamRoot / 
        Config.Paths.DataRoots.CaseStudyRoot() /
        Config.Packages.Drivers.__name__ /
        Config.Packages.Drivers.Phases.Phase3.__name__ /
        Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
        Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task3.__name__/
        Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task3.MetaData.OutputExt)
    
    print("Caricamento nuvola punti e dati modali in corso...")
    pts = pd.read_pickle(pose_root)['pts']
    data = pd.read_pickle(data_root)
    
    # --- INTERFACCIA DI ISPEZIONE ---
    TARGET_STAGE = 'raw'        # 'raw', 'prp' (Blade Locked), 'bld' (Propeller Locked)
    TARGET_ENTITY = 'CAV'       # 'CAV' (Cavity), 'CLD' (Cloud)
    TARGET_FEATURE = 'Entropy'# 'Grayscale', 'Gradient', 'Entropy'
    NUM_MODES = 4               
    
    inspect_modal_3d(data, pts, stage=TARGET_STAGE, entity=TARGET_ENTITY, feature=TARGET_FEATURE, n_modes=NUM_MODES)