#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Calib:
            Dataset = 'Dataset4'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        class Carver:
            occupancyLimit = 2
        Resync = [0, 0, 1, 1, 1, 1, 1, 3, 3, 0, 0]
        Time = [5, 10, 15, 30, 45, 60, 90, 120, 150, 180, 210]
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0