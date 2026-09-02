#%% Defining Config Packet
class Task2:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Analysis:
            dt = 0.00015625 
        class Normals:
            CameraPose = (0.0, 0.0, 0.0) 
            Radius = 500
            Max_nn = 500                 
            TileSize = 5   
        class Smoothing:
            Cutoff = 0.1
            Order = 3
            PadPerc = 0.25
            EdgeOrder = 2
            SavgolWindow = 15  # Finestra mobile per la regolarizzazione (dispari)
            SavgolPoly = 2     # Grado del polinomio interpolante locale
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0