#%% Defining Config Packet
class Task2:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Normals:
            CameraPose = (0.0, 0.0, 0.0) 
            Radius = 100
            Max_nn = 50                 
            TileSize = 2.5
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0