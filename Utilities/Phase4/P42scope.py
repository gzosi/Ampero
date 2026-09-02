import matplotlib
# Set the 'Agg' backend, which doesn't require a GUI (Qt, Wayland, X11)
# This prevents qt.qpa.plugin errors when saving files in headless environments
matplotlib.use('Agg')

from Config import Config
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# DATA LOADING AND PREPARATION
# ---------------------------------------------------------
main_root = Path(Config.Paths.mainRooot)
root = (
    main_root / 
    Config.Paths.DataRoots.ResourcesRoot / 
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ / 
    Config.Packages.Drivers.Phases.Phase4.__name__ / 
    Config.Packages.Drivers.Phases.Phase4.Modules.Module2.__name__ / 
    Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.__name__ / 
    Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.MetaData.OutputExt)

data = pd.read_pickle(root)

# The new output of Task1 is a dictionary of DataFrames indexed directly by 'Blade'.
# We use the dict keys for the 'Dataset', and keep 'Blade' from the original index.
df_plot = pd.concat(data.values(), keys=data.keys(), names=['Dataset', 'Blade']).reset_index()

# Create explicit labels for the legend (e.g., "Blade 1", "Blade 2")
df_plot['Blade_Label'] = 'Blade ' + df_plot['Blade'].astype(str)

# ---------------------------------------------------------
# VISUALIZATION SETTINGS
# ---------------------------------------------------------
# Professional, clean, and highly readable academic aesthetic
sns.set_theme(style="whitegrid", rc={
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "grid.color": "#eaeaea",
    "grid.linestyle": "--",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.5,
})

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12, 
    'axes.titlesize': 16, 
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

num_blades = df_plot['Blade'].nunique()
# Elegant palette for blades (cool, professional blues/teals)
blade_palette = sns.color_palette(["#8ECAE6", "#219EBC", "#023047", "#126782", "#3F72AF"])[:num_blades]
# Dark charcoal for the median: incredibly professional, creates contrast without clashing
median_color = "#222222" 

# ---------------------------------------------------------
# PLOT 1: Area vs Dataset
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

# Plot data for each blade (slightly transparent so the median stands out)
ax1 = sns.lineplot(
    data=df_plot, 
    x='Dataset', 
    y='Area', 
    hue='Blade_Label',   
    style='Blade_Label', 
    markers=['o'] * num_blades,
    dashes=[(4, 3)] * num_blades,
    palette=blade_palette,
    linewidth=2.0,
    markersize=8,
    alpha=0.7,
    zorder=2
)

# Calculate and plot the median curve (thick, solid, contrast color, on top of everything)
df_median_dataset = df_plot.groupby('Dataset', sort=False)['Area'].median().reset_index()
sns.lineplot(
    data=df_median_dataset, 
    x='Dataset', 
    y='Area', 
    color=median_color, 
    linestyle='-', 
    linewidth=3.5, 
    marker='D', 
    markersize=10,
    label='Median', 
    ax=ax1, 
    errorbar=None,
    zorder=10
)

# Aesthetics and formatting
plt.title('Damage Area Evolution (by Dataset Configuration)', fontweight='bold', pad=15, color='#333333')
plt.xlabel('Dataset Configuration', labelpad=12, color='#555555')
plt.ylabel('Damage Area ($mm^2$)', labelpad=12, color='#555555')
plt.xticks(rotation=30, ha='right')

# Clean up legend and place it at the bottom
handles, labels = ax1.get_legend_handles_labels()
# Remove seaborn's automatic title from the legend if present
if labels and labels[0] == 'Blade_Label':
    handles, labels = handles[1:], labels[1:]

plt.legend(
    handles=handles,
    labels=labels,
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.22), # Placed neatly below the plot
    ncol=num_blades + 1, 
    frameon=False, # Removed box for a cleaner look
    fontsize=12
)

plt.tight_layout()
plt.savefig('damages_dataset.pdf', dpi=300, bbox_inches='tight') 
plt.close()

# ---------------------------------------------------------
# PLOT 2: Area vs Time
# ---------------------------------------------------------
if 'Time' in df_plot.columns:
    plt.figure(figsize=(12, 7))
    
    # Plot data for each blade
    ax2 = sns.lineplot(
        data=df_plot, 
        x='Time', 
        y='Area', 
        hue='Blade_Label',   
        style='Blade_Label', 
        markers=['o'] * num_blades,
        dashes=[(4, 3)] * num_blades,
        palette=blade_palette,
        linewidth=2.0,
        markersize=8,
        alpha=0.7,
        zorder=2
    )
    
    # Calculate and plot the median curve
    df_median_time = df_plot.groupby('Time', sort=False)['Area'].median().reset_index()
    sns.lineplot(
        data=df_median_time, 
        x='Time', 
        y='Area', 
        color=median_color, 
        linestyle='-', 
        linewidth=3.5, 
        marker='D', 
        markersize=10,
        label='Median', 
        ax=ax2, 
        errorbar=None,
        zorder=10
    )
    
    # Aesthetics and formatting
    plt.title('Damage Area Evolution (over Time)', fontweight='bold', pad=15, color='#333333')
    plt.xlabel('Time (minutes)', labelpad=12, color='#555555')
    plt.ylabel('Damage Area ($mm^2$)', labelpad=12, color='#555555')
    
    # Clean up legend and place it at the bottom
    handles, labels = ax2.get_legend_handles_labels()
    if labels and labels[0] == 'Blade_Label':
        handles, labels = handles[1:], labels[1:]
        
    plt.legend(
        handles=handles,
        labels=labels,
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15),
        ncol=num_blades + 1, 
        frameon=False,
        fontsize=12
    )
    
    plt.tight_layout()
    plt.savefig('damages_time.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot generation successful: 'damages_dataset.pdf' and 'damages_time.pdf' saved. ✅")
else:
    print("Warning: 'Time' column not found in data. The time-based plot was skipped. ⚠️")