#%% Importing Libreries
import cv2 as cv
#%% Defining Config Packet
class Task1:
    class MetaData:
        InputExt = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        OutputName = 'Data.h5'
    class Settings:
        class Rotation:
            Camera1 = None
            Camera2 = None
        class Resync:
            class Database2:
                Dataset2 = 147
            class Database3: 
                class Dataset1:
                    Background = 246
                    Foreground = 40 # to 85
            class Database4:
                Dataset1 = 41
                Dataset2 = 84
                Dataset3 = 39
                Dataset4 = 229
                Dataset5 = 119
                Dataset6 = 7
                Dataset7 = 157
                Dataset8 = 223
                Dataset9 = 216
                Dataset10 = 81
                Dataset11 = 231
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0