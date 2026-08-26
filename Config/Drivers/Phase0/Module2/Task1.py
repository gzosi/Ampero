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
                Dataset1 = 1
            class Database3: 
                class Dataset1:
                    Background = 21
                    Foreground = 8 #8 to 48
            class Database4:
                Dataset1 = 81
                Dataset2 = 168
                Dataset3 = 182
                Dataset4 = 110
                Dataset5 = 239
                Dataset6 = 145
                Dataset7 = 241
                Dataset8 = 117
                Dataset9 = 78
                Dataset10 = 42
                Dataset11 = 214
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0