import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Forza il backend non interattivo per evitare errori Qt nel terminale
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from Config import Config

# =============================================================================
# 1. CONFIGURAZIONE DELLE VARIABILI
# =============================================================================
# Mappiamo le categorie alle relative variabili di base (senza D0, D1, D2)
CATEGORIES = {
    'dynamic': ['Area', 'Volume'],
    'kinetic': ['AreaC', 'VolumeC', 'DistRel', 'DistNorm'],
    'optics':  ['Whiteness', 'Entropy', 'Gradient']
}

# Parametri propulsore DINAMICI
PPR = Config.Settings.Acquisition.PPR
BLADES_NUMBER = Config.Settings.Acquisition.Blades
DEG_PER_BLADE = 360.0 / BLADES_NUMBER

# Calcolo dinamico dei parametri del grafico
# Genera le linee divisorie (es. 90, 180, 270 per 4 pale)
BLADE_ANGLES = [i * DEG_PER_BLADE for i in range(1, BLADES_NUMBER)]

# Genera il punto centrale per il testo (es. 45, 135, 225, 315 per 4 pale)
BLADE_CENTERS = [(i * DEG_PER_BLADE) + (DEG_PER_BLADE / 2) for i in range(BLADES_NUMBER)]

# Impostazioni grafiche globali stile seaborn
sns.set_theme(style="whitegrid", rc={
    "grid.linestyle": "--", 
    "grid.alpha": 0.7,       # Griglia più visibile (prima era 0.5)
    "grid.linewidth": 1.5,   # Linee della griglia leggermente più spesse
    "axes.facecolor": "#ffffff"
})

# =============================================================================
# 2. FUNZIONI DI SUPPORTO GRAFICO
# =============================================================================
def _add_blade_references(ax):
    """Aggiunge le linee verticali, i box di testo delle pale e le campiture al grafico."""
    
    # Campiture alternate calcolate dinamicamente in base al numero di pale
    blade_regions = [(i * DEG_PER_BLADE, (i + 1) * DEG_PER_BLADE) for i in range(BLADES_NUMBER)]
    
    # Colori alternati (Grigio chiarissimo e bianco) per il numero esatto di pale
    bg_colors = ['#f8f9fa' if i % 2 == 0 else '#ffffff' for i in range(BLADES_NUMBER)]
    
    for (start, end), bg_color in zip(blade_regions, bg_colors):
        ax.axvspan(start, end, facecolor=bg_color, alpha=1.0, zorder=0)

    # Linee di separazione delle pale
    for angle in BLADE_ANGLES:
        ax.axvline(x=angle, color='#6c757d', linestyle=':', linewidth=2, alpha=0.8, zorder=2)
    
    # Box di testo per indicare il nome della pala
    for i, center in enumerate(BLADE_CENTERS):
        ax.text(center, 0.96, f'Blade {i+1}', 
                transform=ax.get_xaxis_transform(), 
                fontsize=28, fontweight='bold', color='#495057', ha='center', va='top', 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.5'),
                zorder=5)

def _format_axes(plt_obj, xlabel, ylabel, title):
    """Formatta assi, etichette e titolo con le dimensioni e l'estetica richieste."""
    ax = plt_obj.gca()
    
    # Formattazione Tick
    plt_obj.xticks(fontsize=24, color='#495057')
    plt_obj.yticks(fontsize=24, color='#495057')
    
    # Formattazione Label Assi
    plt_obj.xlabel(xlabel, fontsize=30, fontweight='bold', color='#343a40', labelpad=15)
    plt_obj.ylabel(ylabel, fontsize=30, fontweight='bold', color='#343a40', labelpad=15)
    
    # Formattazione Titolo
    plt_obj.title(title, fontsize=36, fontweight='bold', color='#212529', pad=20)
    
    # Pulizia dei bordi (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ced4da')
    ax.spines['bottom'].set_color('#ced4da')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt_obj.tight_layout()

# =============================================================================
# 3. MOTORE DI PLOT E GENERAZIONE PDF
# =============================================================================
def generate_pdf_for_variable(df, base_col, pdf_filepath, entity_name, stage):
    """
    Genera un PDF multipagina per una specifica variabile di base.
    Adatta la strategia di plot e l'estetica in base allo stadio ('raw', 'prp', 'bld').
    """
    with PdfPages(pdf_filepath) as pdf:
        
        # ==========================================
        # STILE RAW (Linee connesse per evento)
        # ==========================================
        if stage == 'raw':
            event_column = 'Event'
            unique_events = df[event_column].dropna().unique()
            # Utilizziamo la palette 'husl' di seaborn per colori più gradevoli e distinti
            colors = sns.color_palette("husl", len(unique_events))
            
            # --- PAGINA 1: Base + D0 ---
            fig = plt.figure(figsize=(30, 10))
            ax = plt.gca()
            _add_blade_references(ax)
            
            for (event_name, event_data), color in zip(df.groupby(event_column), colors):
                sorted_data = event_data.sort_values(by='DegP')
                if f'{base_col}D0' in sorted_data.columns:
                    plt.plot(sorted_data['DegP'], sorted_data[f'{base_col}D0'], 
                             marker='o', markersize=8, markeredgecolor='white', markeredgewidth=0.5,
                             linestyle='-', linewidth=3.5, alpha=0.95,
                             color=color, label=f'Event: {event_name} (Smoothed)')
                if base_col in sorted_data.columns:
                    plt.scatter(sorted_data['DegP'], sorted_data[base_col], 
                                s=70, color=color, alpha=0.75, zorder=3, edgecolors='white', linewidths=0.5)
            
            _format_axes(plt, 'Propeller Angle [°]', f'{base_col} Value', f'{entity_name} - {base_col} (Raw vs Smoothed)')
            pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            
            # --- PAGINA 2: Derivata Prima (D1) ---
            if f'{base_col}D1' in df.columns:
                fig = plt.figure(figsize=(30, 10))
                ax = plt.gca()
                _add_blade_references(ax)
                
                for (event_name, event_data), color in zip(df.groupby(event_column), colors):
                    sorted_data = event_data.sort_values(by='DegP')
                    plt.plot(sorted_data['DegP'], sorted_data[f'{base_col}D1'], 
                             marker='s', markersize=7, markeredgecolor='white', markeredgewidth=0.5,
                             linestyle='-', linewidth=3, alpha=0.9,
                             color=color, label=f'Event: {event_name}')
                
                _format_axes(plt, 'Propeller Angle [°]', f'd({base_col})/dt Value', f'{entity_name} - {base_col} First Derivative (D1)')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            # --- PAGINA 3: Derivata Seconda (D2) ---
            if f'{base_col}D2' in df.columns:
                fig = plt.figure(figsize=(30, 10))
                ax = plt.gca()
                _add_blade_references(ax)
                
                for (event_name, event_data), color in zip(df.groupby(event_column), colors):
                    sorted_data = event_data.sort_values(by='DegP')
                    plt.plot(sorted_data['DegP'], sorted_data[f'{base_col}D2'], 
                             marker='^', markersize=7, markeredgecolor='white', markeredgewidth=0.5,
                             linestyle='-', linewidth=3, alpha=0.9,
                             color=color, label=f'Event: {event_name}')
                
                _format_axes(plt, 'Propeller Angle [°]', f'd2({base_col})/dt2 Value', f'{entity_name} - {base_col} Second Derivative (D2)')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # ==========================================
        # STILE PRP (Scatter globale - Blade Locked)
        # ==========================================
        elif stage == 'prp':
            plot_df = df.dropna(subset=['DegP'])
            
            # --- PAGINA 1: Base + D0 ---
            fig = plt.figure(figsize=(30, 10))
            ax = plt.gca()
            _add_blade_references(ax)
            
            if base_col in plot_df.columns:
                plt.scatter(plot_df['DegP'].tolist(), plot_df[base_col].tolist(), 
                            s=70, color='#0dcaf0', alpha=0.75, edgecolors='white', linewidths=0.5, 
                            label=f'{base_col} (Original)', zorder=3)
            if f'{base_col}D0' in plot_df.columns:
                plt.scatter(plot_df['DegP'].tolist(), plot_df[f'{base_col}D0'].tolist(), 
                            s=100, color='#032859', alpha=0.9, edgecolors='white', linewidths=0.8, 
                            label=f'{base_col}D0 (Smoothed)', zorder=4)
            
            _format_axes(plt, 'Propeller Angle [°]', f'{base_col} Value', f'{entity_name} - {base_col} (PRP Blade Locked)')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # --- PAGINA 2: Derivata Prima (D1) ---
            if f'{base_col}D1' in plot_df.columns:
                fig = plt.figure(figsize=(30, 10))
                ax = plt.gca()
                _add_blade_references(ax)
                
                plt.scatter(plot_df['DegP'].tolist(), plot_df[f'{base_col}D1'].tolist(), 
                            s=80, color='#dc3545', alpha=0.85, edgecolors='white', linewidths=0.5, 
                            label=f'{base_col}D1 (First Derivative)', zorder=4)
                
                _format_axes(plt, 'Propeller Angle [°]', f'd({base_col})/dt Value', f'{entity_name} - {base_col} First Derivative (PRP)')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            # --- PAGINA 3: Derivata Seconda (D2) ---
            if f'{base_col}D2' in plot_df.columns:
                fig = plt.figure(figsize=(30, 10))
                ax = plt.gca()
                _add_blade_references(ax)
                
                plt.scatter(plot_df['DegP'].tolist(), plot_df[f'{base_col}D2'].tolist(), 
                            s=80, color='#6f42c1', alpha=0.85, edgecolors='white', linewidths=0.5, 
                            label=f'{base_col}D2 (Second Derivative)', zorder=4)
                
                _format_axes(plt, 'Propeller Angle [°]', f'd2({base_col})/dt2 Value', f'{entity_name} - {base_col} Second Derivative (PRP)')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
        # ==========================================
        # STILE BLD (Scatter globale - Propeller Locked)
        # ==========================================
        elif stage == 'bld':
            # Nel BLD tracciamo l'andamento della pala media
            # Usiamo espressamente 'DegB' come asse X
            plot_df = df.sort_values(by='DegB') if 'DegB' in df.columns else df
            x_vals = plot_df['DegB'].tolist() if 'DegB' in plot_df.columns else plot_df.index.tolist()
            x_label = 'Blade Angle [°]' if 'DegB' in plot_df.columns else 'Time Index'
            
            # --- PAGINA 1: Base + D0 ---
            fig = plt.figure(figsize=(30, 10))
            ax = plt.gca()
            ax.set_facecolor('#f8f9fa')
            
            if base_col in plot_df.columns:
                plt.plot(x_vals, plot_df[base_col].tolist(), 
                         marker='o', markersize=8, linestyle='-', linewidth=2.5,
                         color='#0dcaf0', alpha=0.6, label=f'{base_col} (Original)', zorder=3)
            if f'{base_col}D0' in plot_df.columns:
                plt.plot(x_vals, plot_df[f'{base_col}D0'].tolist(), 
                         marker='s', markersize=8, linestyle='-', linewidth=3.5,
                         color='#032859', alpha=0.9, label=f'{base_col}D0 (Smoothed)', zorder=4)
            
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=24)
            _format_axes(plt, x_label, f'{base_col} Value', f'{entity_name} - {base_col} (BLD Phase Averaged)')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # --- PAGINA 2: Derivata Prima (D1) ---
            if f'{base_col}D1' in plot_df.columns:
                fig = plt.figure(figsize=(30, 10))
                ax = plt.gca()
                ax.set_facecolor('#f8f9fa')
                
                plt.plot(x_vals, plot_df[f'{base_col}D1'].tolist(), 
                         marker='o', markersize=8, linestyle='-', linewidth=3,
                         color='#dc3545', alpha=0.85, label=f'{base_col}D1 (First Derivative)', zorder=4)
                
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=24)
                _format_axes(plt, x_label, f'd({base_col})/dt Value', f'{entity_name} - {base_col} First Derivative (BLD)')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            # --- PAGINA 3: Derivata Seconda (D2) ---
            if f'{base_col}D2' in plot_df.columns:
                fig = plt.figure(figsize=(30, 10))
                ax = plt.gca()
                ax.set_facecolor('#f8f9fa')
                
                plt.plot(x_vals, plot_df[f'{base_col}D2'].tolist(), 
                         marker='^', markersize=8, linestyle='-', linewidth=3,
                         color='#6f42c1', alpha=0.85, label=f'{base_col}D2 (Second Derivative)', zorder=4)
                
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=24)
                _format_axes(plt, x_label, f'd2({base_col})/dt2 Value', f'{entity_name} - {base_col} Second Derivative (BLD)')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

# =============================================================================
# 4. FUNZIONE PRINCIPALE DI COORDINAMENTO
# =============================================================================
def process_data_stage(data_dict, stage_name, base_results_path):
    """
    Processa un livello di elaborazione (es. 'raw', 'prp', 'bld') per CAV e CLD.
    """
    print(f"--- Starting processing stage: {stage_name.upper()} ---")
    
    # Iteriamo sui due "Enti" fisici (Cavity e Cloud)
    for entity in ['CAV', 'CLD']:
        key = f'{entity}{stage_name}'
        
        # Verifichiamo che il DataFrame esista nel dizionario
        if key not in data_dict:
            print(f"Warning: {key} not found in data. Skipping.")
            continue
            
        df = data_dict[key]
        entity_display_name = 'Cavity' if entity == 'CAV' else 'Cloud'
        
        # Iteriamo sulle categorie logiche (dynamic, kinetic, optics)
        for category, variables in CATEGORIES.items():
            # Creazione della cartella result/stage_name/entity/category
            save_dir = Path(base_results_path) / stage_name / entity / category
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Per ogni variabile di base nella categoria, generiamo il suo PDF
            for var in variables:
                if var in df.columns or f'{var}D0' in df.columns:
                    pdf_filename = f"{entity}_{var}_Plots.pdf"
                    pdf_path = save_dir / pdf_filename
                    
                    print(f"Generating plots for {key} -> Category: {category} -> {var}")
                    generate_pdf_for_variable(df, var, pdf_path, entity_display_name, stage_name)

# =============================================================================
# ESECUZIONE (Esempio d'uso con il tuo dizionario 'data')
# =============================================================================
if __name__ == "__main__":
    # Assumiamo che 'data' sia il tuo dizionario precedentemente caricato da pickle
    main_root = Path(Config.Paths.mainRooot)
    root = (main_root /
        Config.Paths.DataRoots.ResourcesRoot /
        Config.Paths.DataRoots.StreamRoot / 
        Config.Paths.DataRoots.CaseStudyRoot() /
        Config.Packages.Drivers.__name__ /
        Config.Packages.Drivers.Phases.Phase3.__name__ /
        Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
        Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task2.__name__/
        Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task2.MetaData.OutputExt)
    data = pd.read_pickle(root)
    
    RESULTS_DIR = "Plot P3"
    
    # Ora processiamo sia 'raw' che 'prp'
    stages_to_process = ['raw', 'prp', 'bld']
    
    
    for stage in stages_to_process:
        # Nota: assicurati di passare il dizionario 'data' reale. 
        process_data_stage(data, stage, RESULTS_DIR)
        # pass
    
    print("Elaborazione e salvataggio PDF completati con successo!")