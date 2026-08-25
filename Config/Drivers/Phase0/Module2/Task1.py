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
            Camera2 = cv.ROTATE_180
            # cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE 
        class Resync:
            class Database2:
                Dataset1 = 195
            class Database3: 
                class Dataset2:
                    Background = 85
                    Foreground = 25
            class Database4:
                Dataset1 = 220
                Dataset2 = 129
                Dataset3 = 210
                Dataset4 = 38
                Dataset5 = 20
                Dataset6 = 221
                Dataset7 = 185
                Dataset8 = 205
                Dataset9 = 11
                Dataset10 = 244
                Dataset11 = 224
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0