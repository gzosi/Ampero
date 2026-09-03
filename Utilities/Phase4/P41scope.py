import argparse
from Config import Config 
from pathlib import Path
import pickle
import sys
import h5py
import cv2
import numpy as np
import pandas as pd

# Imposta il backend di matplotlib in modalità "headless" (non interattiva)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def padOrigin(img, shape_wh, start_coords_xy):
    """Funzione di padding con unpack sicuro per allineare l'immagine al sensore."""
    numpy_shape = (shape_wh[1], shape_wh[0]) 
    result = np.zeros(numpy_shape, dtype=img.dtype) 
    
    if isinstance(start_coords_xy, (list, tuple, np.ndarray, pd.Series)) and len(start_coords_xy) >= 2:
        if hasattr(start_coords_xy, 'tolist'):
            start_coords_xy = start_coords_xy.tolist()
        x_start, y_start = int(start_coords_xy[0]), int(start_coords_xy[1])
    else:
        x_start, y_start = 0, 0
        
    y_end = y_start + img.shape[0]  
    x_end = x_start + img.shape[1] 
    
    # Prevenzione slice negativi se le coordinate superano le dimensioni
    y_start = max(0, min(y_start, result.shape[0]))
    x_start = max(0, min(x_start, result.shape[1]))
    
    y_end_clamped = max(y_start, min(y_end, result.shape[0]))
    x_end_clamped = max(x_start, min(x_end, result.shape[1]))
    
    src_y_end = y_end_clamped - y_start
    src_x_end = x_end_clamped - x_start
    
    if src_y_end > 0 and src_x_end > 0:
        result[y_start:y_end_clamped, x_start:x_end_clamped] = img[0:src_y_end, 0:src_x_end]
        
    return result

def process_single_frame(dataset_name, key, data, f_h5, origins_data, shape):
    """Processa e salva una singola combinazione di Dataset e Key."""
    print(f"\n🔄 Processando {dataset_name} - Key {key}...")
    
    frame_data = data[dataset_name][key]
    PHASE = frame_data.get('Phase', 'N/A')
    cameras = [k for k in frame_data.keys() if k != 'Phase']
    
    # Inizializzazione PLOT
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    try:
        for idx, camera in enumerate(cameras[:2]):
            print(f"  🔍 Telecamera: {camera}")
            
            # 1. Carica l'immagine originale
            try:
                raw = f_h5[camera]['Database4'][dataset_name][key][:].astype(np.float32)
                if raw.max() > 0:
                    raw = (raw / raw.max()) * 255.0
                raw = raw.astype(np.uint8)
            except KeyError:
                print(f"    ❌ Errore: Key mancante nell'H5 per {camera}. Disegno base vuota.")
                raw = np.zeros(shape[::-1], dtype=np.uint8)

            # 2. Pad dell'immagine originaria 
            if raw.shape[:2] != tuple(shape[::-1]):
                try:
                    start_coords = origins_data[camera]['Database4'][dataset_name]
                    raw_padded = padOrigin(raw, shape, start_coords)
                except KeyError:
                    print(f"    ⚠️ Offset non trovato, l'immagine non è allineata.")
                    raw_padded = raw
            else:
                raw_padded = raw

            # 3. Maschera dal PKL
            mask = frame_data[camera]
            if mask.max() == 1: 
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

            # Verifica shape mismatch
            if raw_padded.shape[:2] != mask.shape[:2]:
                print(f"    ⚠️ ATTENZIONE: Dimensioni sfalsate! Img={raw_padded.shape[:2]}, Mask={mask.shape[:2]}")
                raw_padded = cv2.resize(raw_padded, (mask.shape[1], mask.shape[0]))

            # 4. Disegno Overlay e Contorni 
            display_img = cv2.cvtColor(raw_padded, cv2.COLOR_GRAY2RGB)
            overlay = display_img.copy()
            
            contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
            
            display_img = cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0)
            cv2.drawContours(display_img, contours, -1, (255, 0, 0), thickness=2)

            # 5. Visualizzazione nel plot
            axes[idx].imshow(display_img)
            axes[idx].set_title(f"Telecamera: {camera}")
            axes[idx].axis('off')

    except Exception as e:
        print(f"    ❌ Errore durante il plot di {dataset_name}-{key}: {e}")
        plt.close(fig)
        return

    # Messaggio di compensazione per camera singola
    if len(cameras) < 2:
        axes[1].text(0.5, 0.5, 'Nessuna Seconda Telecamera', horizontalalignment='center', verticalalignment='center', fontsize=12)
        axes[1].axis('off')

    plt.suptitle(f"Dataset: {dataset_name} | Phase: {PHASE} | Key: {key}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    # SALVATAGGIO IN CARTELLA DAMAGES
    damages_dir = Path.cwd() / "Tmp"
    damages_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = damages_dir / f"{dataset_name}_Key{key}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    
    print(f"  ✅ Salvato: {output_filename}")


def main():
    # --- CONFIGURAZIONE ARGPARSE ---
    parser = argparse.ArgumentParser(description="P41Scope - Generatore Overlay Maschere")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['all', 'select'], 
        default='select', 
        help="Scegli 'all' per elaborare tutte le chiavi in tutti i dataset, o 'select' per sceglierne uno specifico."
    )
    parser.add_argument('--dataset', type=str, default='Dataset1', help="Nome del dataset (usato solo se --mode select)")
    parser.add_argument('--key', type=str, default='00030', help="Nome della chiave (usato solo se --mode select)")
    args = parser.parse_args()

    print("🚀 Avvio P41Scope...")
    main_root = Path(Config.Paths.mainRooot)
    
    # --- Path generati ---
    ROOT_PKL = (main_root /
        Config.Paths.DataRoots.ResourcesRoot /
        Config.Paths.DataRoots.StreamRoot / 
        Config.Paths.DataRoots.CaseStudyRoot() /
        Config.Packages.Drivers.__name__ /
        Config.Packages.Drivers.Phases.Phase4.__name__ /
        Config.Packages.Drivers.Phases.Phase4.Modules.Module1.__name__ /
        Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task1.__name__ /
        Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task1.MetaData.OutputExt)
        
    ROOT_H5 = (main_root /
        Config.Paths.DataRoots.ResourcesRoot /
        Config.Paths.DataRoots.StreamRoot / 
        Config.Paths.DataRoots.CaseStudyRoot() /
        Config.Packages.Drivers.__name__ /
        Config.Packages.Drivers.Phases.Phase0.__name__ /
        Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
        Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.__name__ /
        Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.MetaData.OutputName)

    SHAPE_ROOT = (main_root /
        Config.Paths.DataRoots.ResourcesRoot /
        Config.Paths.DataRoots.StreamRoot /
        Config.Paths.DataRoots.CaseStudyRoot() /
        Config.Packages.Drivers.__name__ / 
        Config.Packages.Drivers.Phases.Phase0.__name__ /
        Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
        Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.__name__)

    # --- Apertura PKL ---
    try:
        with open(ROOT_PKL, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Errore durante il caricamento del file PKL: {e}")
        sys.exit(1)

    # --- Caricamento Origins e Shape ---
    try:
        origins_file = SHAPE_ROOT / Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.OriginExt
        origins_data = pd.read_json(origins_file)
        shape = Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.Settings.FullSensorShape
    except Exception as e:
        print(f"❌ Errore nel caricare shape e origins JSON: {e}")
        sys.exit(1)

    # --- Costruzione della lista dei Target da processare ---
    targets = []
    
    if args.mode == 'all':
        print("🌍 Modalità 'ALL' selezionata. Scansione dell'intero file PKL in corso...")
        for ds_name, ds_data in data.items():
            for key_name in ds_data.keys():
                targets.append((ds_name, key_name))
        print(f"Trovate {len(targets)} combinazioni da processare in totale.")
    
    elif args.mode == 'select':
        print(f"🎯 Modalità 'SELECT' selezionata. Cerco Dataset: '{args.dataset}', Key: '{args.key}'...")
        if args.dataset not in data:
            print(f"❌ Errore: Dataset '{args.dataset}' non trovato nel file.")
            sys.exit(1)
        if args.key not in data[args.dataset]:
            print(f"❌ Errore: Key '{args.key}' assente nel dataset '{args.dataset}'.")
            print(f"Scegli tra queste: {sorted(list(data[args.dataset].keys()))}")
            sys.exit(1)
        targets.append((args.dataset, args.key))

    # --- Apertura H5 ed Esecuzione ---
    try:
        # Apriamo il file H5 una volta sola e passiamo l'handle per evitare overhead I/O nel ciclo
        with h5py.File(ROOT_H5, 'r') as f_h5:
            for dataset_name, key in targets:
                process_single_frame(dataset_name, key, data, f_h5, origins_data, shape)
                
    except Exception as e:
        print(f"❌ Errore critico in fase di apertura file H5: {e}")
        sys.exit(1)
        
    print("\n🎉 Operazione completata!")


if __name__ == "__main__":
    main()

# =============================================================================
# LEGENDA E GUIDA ALL'USO (DA TERMINALE / RIGA DI COMANDO)
# =============================================================================
# Questo script si controlla tramite argomenti passati da terminale.
#
# SINTASSI BASE:
#   python p41scope.py [--mode MODALITA] [--dataset NOME_DATASET] [--key CHIAVE]
#
# PARAMETRI DISPONIBILI:
#   --mode    : Sceglie come far funzionare lo script. 
#               Può essere 'select' (di default) o 'all'.
#               - 'select' -> Analizza solo un dataset e una chiave specifici.
#               - 'all'    -> Analizza tutte le chiavi in tutti i dataset presenti 
#                             nel file (ignora i parametri --dataset e --key).
#
#   --dataset : Il nome del dataset che vuoi processare (es: 'Dataset2').
#               (Se non specificato, il valore di default è 'Dataset1').
#
#   --key     : La chiave specifica da processare (es: '00050').
#               (Se non specificato, il valore di default è '00030').
#
# ESEMPI PRATICI DA COPIARE NEL TERMINALE:
# 
# 1. Esecuzione standard (Usa i valori di default: Dataset1 e Key 00030):
#    python p41scope.py
#
# 2. Cercare un'immagine specifica in un dataset specifico:
#    python p41scope.py --mode select --dataset Dataset3 --key 00125
#
# 3. Processare in blocco l'intero contenuto del file PKL:
#    python p41scope.py --mode all
#
# DOVE TROVO I RISULTATI?
# Tutte le immagini generate verranno salvate automaticamente in una cartella
# chiamata "Tmp" nella stessa posizione in cui si trova questo script.
# =============================================================================