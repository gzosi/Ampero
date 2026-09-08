import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
from scipy.special import expit
import matplotlib.ticker as ticker
def generate_plots(Config):
    main_root = Path(Config.Paths.mainRooot)
    srcRoot = main_root / Config.Paths.DataRoots.ResourcesRoot / Config.Paths.DataRoots.StreamRoot / Config.Paths.DataRoots.CaseStudyRoot() / Config.Packages.Drivers.__name__ / Config.Packages.Drivers.Phases.Phase4.__name__ / Config.Packages.Drivers.Phases.Phase4.Modules.Module2.__name__ / Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task2.__name__ / Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task2.MetaData.OutputExt
    results_dir = main_root / "results"
    print(f"Loading data from: {srcRoot}")
    try:
        with open(srcRoot, 'rb') as f: levels = pickle.load(f)
    except FileNotFoundError:
        print("Pickle file not found. Please run the data generation task first.")
        return
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fb", "figure.facecolor": "#ffffff", "grid.color": "#e2e5eb", "grid.linestyle": "--", "axes.edgecolor": "#b0b7c3", "axes.linewidth": 1.2})
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 13, 'legend.fontsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11})
    base_palette = ["#7dc4b8", "#44a9a8", "#226a8d", "#114066", "#051630"]
    median_color = "#111111"
    def extract_datasets(node, current_path=[]):
        if all(isinstance(v, pd.DataFrame) for v in node.values()): yield current_path, node
        else:
            for key, child in node.items(): yield from extract_datasets(child, current_path + [key])
    metrics_to_plot = {'energyDelta': 'Energy Delta (%)', 'densityDelta': 'Density Delta (%)', 'mutualInfo': 'Mutual Information', 'sigmoid_mse': 'Brier Score (MSE)', 'sigmoid_x0': 'Energy for 50% Damage Prob ($E_{50}$)'}
    for energy_name in levels.keys():
        energy_node = levels[energy_name]
        for path_keys, datasetsDICT in extract_datasets(energy_node):
            category_name = "_".join(str(k) for k in path_keys)
            dest_folder = results_dir / energy_name / category_name
            os.makedirs(dest_folder, exist_ok=True)
            list_of_dfs = []
            for dataset_name, df_blades in datasetsDICT.items():
                temp_df = df_blades.copy()
                temp_df['Dataset_Name'] = dataset_name
                list_of_dfs.append(temp_df)
            if not list_of_dfs: continue
            df_plot = pd.concat(list_of_dfs, ignore_index=True)
            if 'Time' not in df_plot.columns or 'Blade' not in df_plot.columns: continue
            df_plot = df_plot.sort_values(by=['Blade', 'Time'])
            df_real_blades = df_plot[df_plot['Blade'] != 'Median'].copy()
            df_median = df_plot[df_plot['Blade'] == 'Median'].copy()
            df_real_blades['Blade_Label'] = df_real_blades['Blade'].apply(lambda x: f'Blade {x}' if str(x).isdigit() or isinstance(x, (int, float)) else str(x))
            num_blades = df_real_blades['Blade'].nunique()
            blade_palette = sns.color_palette(base_palette[:num_blades])
            for metric_col, metric_label in metrics_to_plot.items():
                if metric_col not in df_plot.columns: continue
                plt.figure(figsize=(12, 7))
                if not df_real_blades.empty: ax = sns.lineplot(data=df_real_blades, x='Time', y=metric_col, hue='Blade_Label', style='Blade_Label', markers=['o'] * num_blades, dashes=[(4, 3)] * num_blades, palette=blade_palette, linewidth=2.5, markersize=9, alpha=0.9, zorder=2)
                else: ax = plt.gca()
                if not df_median.empty: sns.lineplot(data=df_median, x='Time', y=metric_col, color=median_color, linestyle='-', linewidth=4.0, marker='D', markersize=10, label='Median Parameters', ax=ax, errorbar=None, zorder=10)
                plot_title = f"{energy_name} - {category_name.replace('_', ' ').upper()}"
                plt.title(f'{plot_title}\nEvolution of {metric_label} over Time', fontweight='bold', pad=15, color='#2c3e50')
                plt.xlabel('Time (minutes)', labelpad=12, fontweight='500', color='#34495e')
                plt.ylabel(metric_label, labelpad=12, fontweight='500', color='#34495e')
                handles, labels = ax.get_legend_handles_labels()
                if labels and labels[0] == 'Blade_Label': handles, labels = handles[1:], labels[1:]
                plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=num_blades + 1, frameon=True, facecolor='white', edgecolor='#e2e5eb', framealpha=1)
                plt.tight_layout()
                plt.savefig(dest_folder / f"plot_time_{metric_col}.pdf", dpi=300, bbox_inches='tight')
                plt.close()
            df_sig = df_plot.dropna(subset=['sigmoid_k', 'sigmoid_x0'])
            if not df_sig.empty:
                x0_min, x0_max = df_sig['sigmoid_x0'].min(), df_sig['sigmoid_x0'].max()
                e_min = max(0, x0_min - (x0_max - x0_min) * 1.5)
                e_max = x0_max + (x0_max - x0_min) * 1.5
                if e_min == e_max: e_max = e_min + 1.0 
                E_vals = np.linspace(e_min, e_max, 500)
                
                norm = plt.Normalize(df_sig['Time'].min(), df_sig['Time'].max())
                cmap = plt.get_cmap("mako_r")
                plot_title = f"{energy_name} - {category_name.replace('_', ' ').upper()}"
                
                df_sig_med = df_sig[df_sig['Blade'] == 'Median']
                if not df_sig_med.empty:
                    plt.figure(figsize=(10, 6))
                    for time_val, df_t in df_sig_med.groupby('Time'):
                        color = cmap(norm(time_val))
                        for _, row in df_t.iterrows(): plt.plot(E_vals, expit(row['sigmoid_k'] * (E_vals - row['sigmoid_x0'])), color=color, linestyle='-', alpha=0.9, linewidth=3.0)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=plt.gca())
                    cbar.set_label('Time (minutes)', rotation=270, labelpad=20, fontweight='500', color='#34495e')
                    cbar.outline.set_edgecolor('#b0b7c3')
                    ax_med = plt.gca()
                    ax_med.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                    formatter = ticker.ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((0, 0))
                    ax_med.xaxis.set_major_formatter(formatter)
                    plt.title(f'{plot_title}\nMedian Parameters Sigmoid Evolution', fontweight='bold', pad=15, color='#2c3e50')
                    plt.xlabel('Energy', labelpad=12, fontweight='500', color='#34495e')
                    plt.ylabel('Damage Probability', labelpad=12, fontweight='500', color='#34495e')
                    plt.tight_layout()
                    plt.savefig(dest_folder / "plot_sigmoid_evolution_median.pdf", dpi=300, bbox_inches='tight')
                    plt.close()
                df_sig_blades = df_sig[df_sig['Blade'] != 'Median']
                if not df_sig_blades.empty:
                    blade_ids = df_sig_blades['Blade'].unique()
                    num_real_blades = len(blade_ids)
                    fig, axes = plt.subplots(1, num_real_blades, figsize=(num_real_blades * 5.5, 5.5), sharex=True, sharey=True)
                    axes_flat = [axes] if num_real_blades == 1 else axes
                    sns.despine(fig=fig, top=True, right=True, left=False, bottom=False) 
                    for time_val, df_t in df_sig_blades.groupby('Time'):
                        color = cmap(norm(time_val))
                        for _, row in df_t.iterrows(): axes_flat[np.where(blade_ids == row['Blade'])[0][0]].plot(E_vals, expit(row['sigmoid_k'] * (E_vals - row['sigmoid_x0'])), color=color, alpha=0.75, linewidth=2.5)
                    for idx, b_id in enumerate(blade_ids):
                        ax = axes_flat[idx]
                        ax.text(0.95, 0.05, f'Blade {b_id}' if str(b_id).isdigit() or isinstance(b_id, (int, float)) else str(b_id), transform=ax.transAxes, fontsize=15, fontweight='bold', color='#1d3557', va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#b0b7c3', boxstyle='round,pad=0.5'))
                        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                        formatter = ticker.ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((0, 0)) 
                        ax.xaxis.set_major_formatter(formatter)
                        ax.set_xlabel('Energy', labelpad=12, fontweight='bold', color='#4a5568')
                        ax.tick_params(colors='#4a5568', which='both')
                        if idx == 0: ax.set_ylabel('Damage Probability', labelpad=12, fontweight='bold', color='#4a5568')
                    fig.subplots_adjust(bottom=0.25, wspace=0.1) 
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03]) 
                    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                    cbar.set_label('Time (minutes)', labelpad=10, fontweight='bold', color='#4a5568')
                    cbar.ax.tick_params(colors='#4a5568')
                    cbar.outline.set_edgecolor('#b0b7c3')
                    fig.suptitle(f'{plot_title}\nIndividual Blades Sigmoid Evolution', fontweight='bold', fontsize=17, y=1.08, color='#2c3e50')
                    plt.savefig(dest_folder / "plot_sigmoid_evolution_blades.pdf", dpi=300, bbox_inches='tight')
                    plt.close()
    print("All plots generated successfully! ✅")
from Config import Config
generate_plots(Config)