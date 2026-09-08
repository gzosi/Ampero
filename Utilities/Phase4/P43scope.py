from Config import Config
from pathlib import Path
import pandas as pd
import numpy as np
import pyvista as pv

# Imposta qui il dataset e la pala che vuoi visualizzare
target_dataset = 'Dataset11' # <-- Sostituisci con il nome reale del tuo dataset
target_blade = 3             # <-- Sostituisci con il numero della pala (es. 1, 2, 3...)


# ---------------------------------------------------------
# CARICAMENTO DATI
# ---------------------------------------------------------
main_root = Path(Config.Paths.mainRooot)

# Puntiamo all'output del nostro Task1 (Phase 4, Module 2)
root = (
    main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ / 
    Config.Packages.Drivers.Phases.Phase4.__name__ / 
    Config.Packages.Drivers.Phases.Phase4.Modules.Module2.__name__ /
    Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.__name__ /
    Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.MetaData.OutputExt
)

data = pd.read_pickle(root)

# ---------------------------------------------------------
# CONTROLLI E VISUALIZZAZIONE
# ---------------------------------------------------------
if target_dataset not in data:
    print(f"Dataset '{target_dataset}' non trovato. Dataset disponibili:\n{list(data.keys())}")
elif target_blade not in data[target_dataset].index:
    print(f"Pala {target_blade} non trovata nel dataset {target_dataset}. Pale disponibili: {list(data[target_dataset].index)}")
else:
    # Estrazione dei dati compatti dalla nuova struttura
    blade_data = data[target_dataset].loc[target_blade]
    
    pose_pts = blade_data['Pose']
    damage_pts = blade_data['Damage']
    area = blade_data['Area']
    
    # Inizializzazione del Plotter di PyVista (finestra singola per maggiore chiarezza)
    pl = pv.Plotter()
    
    # Aggiungiamo un testo informativo in alto
    info_text = f"Dataset: {target_dataset} | Blade: {target_blade}\nTotal Damage Area: {area:.2f} mm^2"
    pl.add_text(info_text, font_size=14, position='upper_edge')
    
    # PLOT 1: POSE (Geometria base dell'elica in posizione 0)
    if len(pose_pts) > 0:
        base_geom = pv.PolyData(np.asarray(pose_pts))
        # Usiamo il colore darkorange per la pala di base
        pl.add_mesh(base_geom, color='darkorange', style='points', point_size=5, render_points_as_spheres=True)
    
    # PLOT 2: DAMAGE (Punti erosi/danneggiati)
    if len(damage_pts) > 0:
        damage_geom = pv.PolyData(np.asarray(damage_pts))
        
        # Visualizziamo i danni direttamente come punti appartenenti alla superficie
        # con una dimensione (point_size) maggiore rispetto alla base per farli risaltare
        pl.add_points(damage_geom, color='red', point_size=7, render_points_as_spheres=True)

    # Mostriamo il risultato
    pl.show(jupyter_backend='trame')