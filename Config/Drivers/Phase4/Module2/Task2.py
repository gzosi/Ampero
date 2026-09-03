#%% Defining Config Packet
class Task2:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        Categories = [
            # --- Analisi Grayscale ---
            ['Grayscale_TimeVar'],
            ['Grayscale_RMS'],
            ['Grayscale_PowerVar'],
            ['Grayscale_PowerRMS'],
            ['Grayscale_POD', 'mean_field'],
            ['Grayscale_POD', 'spatial_modes', 0], 
            ['Grayscale_POD', 'spatial_energy_map'],
            ['Grayscale_POD', 'power_map'],        
            
            # --- Analisi Gradient ---
            ['Gradient_TimeVar'],
            ['Gradient_RMS'],
            ['Gradient_PowerVar'],
            ['Gradient_PowerRMS'],
            ['Gradient_POD', 'mean_field'],
            ['Gradient_POD', 'spatial_modes', 0], 
            ['Gradient_POD', 'spatial_energy_map'],
            ['Gradient_POD', 'power_map'],

            # --- Analisi Entropy ---
            ['Entropy_TimeVar'],
            ['Entropy_RMS'],
            ['Entropy_PowerVar'],
            ['Entropy_PowerRMS'],
            ['Entropy_POD', 'mean_field'],
            ['Entropy_POD', 'spatial_modes', 0], 
            ['Entropy_POD', 'spatial_energy_map'],
            ['Entropy_POD', 'power_map'],
        ]
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0