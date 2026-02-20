#%% Importing Libreries
import cv2 as cv
#%% Defining Config Packet
class Task1:
    class MetaData:
        InputExt = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        OutputExt = 'Data.h5'
    class Settings:
        class Rotation:
            Camera1 = None
            Camera2 = cv.ROTATE_180
            # cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE 
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0