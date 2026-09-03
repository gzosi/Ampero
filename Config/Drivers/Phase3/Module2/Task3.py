#%% Defining Config Packet
class Task3:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Calib:
            Dataset = 'Dataset4'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        MaxDist = 20
        class Analysis:
            dt = 0.00015625 
            class POD:
                n_modes = 25
                eig_th = 1e-10 
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0