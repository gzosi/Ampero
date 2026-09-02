import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Forza la modalità senza interfaccia grafica
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from Config import Config
from pathlib import Path

# ==========================================
# 1. FUNZIONI DI SUPPORTO
# ==========================================
def add_blade_regions(ax, blades_list, max_phase=360):
    """
    Divide il grafico in sezioni verticali in base al numero di pale 
    e aggiunge l'etichetta testuale all'interno del plot.
    """
    n_blades = len(blades_list)
    if n_blades <= 0: return
    
    step = max_phase / n_blades
    
    for i in range(0, n_blades + 1):
        ax.axvline(x=i*step, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
        
    for i, blade in enumerate(blades_list):
        start = i * step
        end = (i + 1) * step
        center = start + (step / 2)
        
        # Sfondo grigio alternato per distinguere meglio le pale
        if i % 2 == 0:
            ax.axvspan(start, end, color='gray', alpha=0.1)
            
        # Testo della pala posizionato in alto
        ax.text(center, 0.96, f'Blade {blade}', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=18, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3'))

# ==========================================
# 2. CARICAMENTO DATI REALI
# ==========================================
print("Caricamento dati in corso...")

task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1
main_root = Path(Config.Paths.mainRooot)

root = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ /
    Config.Packages.Drivers.Phases.Phase3.__name__ /
    Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
    task_conf.__name__/
    task_conf.MetaData.OutputExt)
    
data = pd.read_pickle(root)
print(f"Dati caricati con successo! Elementi trovati: {len(data)}")

# Parametri propulsore DINAMICI presi dalla Configurazione
ppr = Config.Settings.Acquisition.PPR
bladesNumber = Config.Settings.Acquisition.Blades
print(f"Configurazione rilevata: PPR={ppr}, Pale={bladesNumber}")

# ==========================================
# 3. PREPARAZIONE ED ESTRAZIONE DATI
# ==========================================
print("Preparazione e formattazione dei dati...")
records = []
points_per_blade = ppr / bladesNumber

for key in list(data.keys()):
    df = data[key]
    key_num = int(key)
    phase = key_num % ppr
    
    # Divide la fase per il blocco di punti appartenenti a ciascuna pala
    blade = int(phase // points_per_blade) + 1
    
    # Identificativo univoco della realizzazione (ogni rotazione completa = 1 realizzazione)
    realization_id = key_num // ppr 
    
    # Calcolo della fase in gradi (0-360)
    phase_deg = phase * (360.0 / ppr)
    
    # Estrazione sicura: somma le aree/volumi se i dati esistono e non sono vuoti
    if not df.empty:
        cavityArea = np.sum([v['Area'] for v in df['Cavity'].values]) if 'Cavity' in df else 0
        cloudArea = np.sum([v['Area'] for v in df['Cloud'].values]) if 'Cloud' in df else 0
        cavityVolume = np.sum([v['Volume'] for v in df['Cavity'].values]) if 'Cavity' in df else 0
        cloudVolume = np.sum([v['Volume'] for v in df['Cloud'].values]) if 'Cloud' in df else 0
    else:
        cavityArea = cloudArea = cavityVolume = cloudVolume = 0

    records.append({
        'key': key_num,
        'Phase': phase,
        'Phase_deg': phase_deg,
        'Blade': blade,
        'Realization': realization_id,
        'Cavity_Area': cavityArea,
        'Cloud_Area': cloudArea,
        'Cavity_Volume': cavityVolume,
        'Cloud_Volume': cloudVolume
    })

# Crea un DataFrame pulito e ordinato per permettere a Seaborn di tracciare le linee
df_plot = pd.DataFrame(records)
df_plot = df_plot.sort_values(by=['Realization', 'key'])

# Creiamo un identificatore univoco per "Realizzazione + Pala" per spezzare le linee nei grafici
df_plot['Realization_Blade'] = df_plot['Realization'].astype(str) + "_" + df_plot['Blade'].astype(str)

unique_blades = sorted(df_plot['Blade'].unique())

# ==========================================
# 4. GENERAZIONE DEI GRAFICI SEPARATI
# ==========================================
print("Generazione dei grafici in corso...")
sns.set_theme(style="whitegrid")

metrics_configs = [
    {'y_col': 'Cavity_Area',   'title': 'Main Cavitation - Area',    'ylabel': 'Area (mm²)'},
    {'y_col': 'Cloud_Area',    'title': 'Cloud Cavitation - Area',   'ylabel': 'Area (mm²)'},
    {'y_col': 'Cavity_Volume', 'title': 'Main Cavitation - Volume',  'ylabel': 'Volume (mm³)'},
    {'y_col': 'Cloud_Volume',  'title': 'Cloud Cavitation - Volume', 'ylabel': 'Volume (mm³)'}
]

color_data = '#00b4d8'   # Azzurro per le realizzazioni originali
color_median = '#03045e' # Blu scuro per la mediana
color_iqr = '#48cae4'    # Azzurro per lo sfondo dell'interquartile

results_dir = Path("Results")
results_dir.mkdir(parents=True, exist_ok=True)

for config in metrics_configs:
    y_col = config['y_col']
    
    # Ignora i punti con valore 0 se vuoi che il grafico non collassi a zero
    df_valid = df_plot[df_plot[y_col] > 0]
    
    if df_valid.empty:
        print(f"Nessun dato valido per {y_col}. Salto la generazione.")
        continue
        
    fig, ax = plt.subplots(figsize=(24, 12))
    fig.suptitle(f"{config['title']} Evolution per Blade", fontsize=36, fontweight='bold', y=0.95)
    
    # 1. Tracciamento delle singole realizzazioni (ora spezzate per pala usando 'Realization_Blade')
    sns.lineplot(data=df_valid, x='Phase_deg', y=y_col,
                 units='Realization_Blade', estimator=None, marker='o', markersize=6, 
                 color=color_data, linewidth=1.5, alpha=0.6, ax=ax, legend=False, zorder=3)
    
    # 2. Calcolo Statistiche (Mediana, Q1, Q3) raggruppate PER PALA e Fase
    stats = df_valid.groupby(['Blade', 'Phase_deg'])[y_col].agg(
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75)
    ).reset_index()
    
    # 3. Tracciamento Mediana e IQR separati per ciascuna pala
    for blade in unique_blades:
        blade_stats = stats[stats['Blade'] == blade].sort_values('Phase_deg')
        if not blade_stats.empty:
            # Area Interquartile (Q1 - Q3)
            ax.fill_between(blade_stats['Phase_deg'], blade_stats['q1'], blade_stats['q3'],
                            color=color_iqr, alpha=0.45, edgecolor='none', zorder=1)
            
            # Linea della Mediana
            ax.plot(blade_stats['Phase_deg'], blade_stats['median'], color=color_median,
                    linewidth=3.5, linestyle='-', zorder=4)

    # Spazio Y extra per evitare sovrapposizioni con i testi delle pale (Blade 1, 2, ecc.)
    y_max = df_valid[y_col].max()
    y_min = df_valid[y_col].min()
    y_range = y_max - y_min if y_max != y_min else 1.0
    ax.set_ylim(y_min - (y_range * 0.05), y_max + (y_range * 0.35))
        
    # Aggiungi i divisori delle pale e le zone grigie
    add_blade_regions(ax, unique_blades, max_phase=360)
    
    # Formattazione estetica
    ax.set_ylabel(config['ylabel'], fontsize=28, labelpad=20)
    ax.set_xlabel('Phase [°]', fontsize=28, labelpad=20)
    ax.set_xlim(-5, 365)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Legenda unificata in basso
    legend_elements = [
        mlines.Line2D([0], [0], color=color_data, lw=1.5, marker='o', markersize=6, alpha=0.6, label='Single Realization'),
        mlines.Line2D([0], [0], color=color_median, lw=3.5, label='Median (per blade)'),
        mpatches.Patch(color=color_iqr, alpha=0.45, label='IQR 25-75% (per blade)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=24)
    
    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(top=0.88, bottom=0.20) # Lascia spazio per il titolo e la legenda in basso
    
    # Salvataggio del singolo grafico
    safe_name = config['title'].replace(' ', '_').replace('-', '')
    file_out = results_dir / f'{safe_name}.pdf'
    fig.savefig(file_out, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f" -> Salvato grafico: {file_out.name}")

print(f"\nCompletato! Tutti i grafici sono stati salvati in: {results_dir.absolute()}")