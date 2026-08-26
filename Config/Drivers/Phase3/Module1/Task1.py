import numpy as np
import cv2 as cv
#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputName = 'Data'
        OutputExt = '.pkl'
    class Settings:
        class Src:
            Database = 'Database3'
            Dataset = 'Dataset1'
            Foreground = 'Foreground'
            Background = 'Background'
        Bounds = [[0, 36]] 
        class DynamicROI:
            Camera1 = {
                0 : np.array([[789, 316], [629, 240], [567, 178], [590, 88], [719, 145], [816, 275]]),
                9 : np.array([[814, 366], [658, 325], [569, 262], [578, 140], [674, 166], [824, 302]]),
                18 : np.array([[830, 388], [695, 386], [590, 335], [574, 211], [686, 218], [833, 329]]),
                27 : np.array([[863, 411], [751, 437], [595, 399], [567, 254], [702, 278], [849, 357]]),
                36 : np.array([[915, 443], [821, 483], [647, 457], [588, 318], [787, 374], [907, 406]]),
            }
            Camera2 = {
                0 : np.array([[865, 511], [701, 445], [590, 329], [586, 249], [724, 308], [859, 447]]),
                9 : np.array([[817, 522], [690, 502], [601, 445], [563, 303], [705, 347], [819, 469]]),
                18 : np.array([[838, 541], [725, 552], [606, 489], [569, 363], [695, 380], [836, 488]]),
                27 : np.array([[865, 575], [742, 608], [633, 569], [576, 421], [698, 433], [849, 527]]),
                36 : np.array([[882, 603], [754, 617], [631, 589], [579, 450], [719, 484], [860, 570]]),
            }
        class Enhancement:
            class Bilateral:
                d = 5  
                sigmaColor = 15     
                sigmaSpace = 15
            class CLAHE:
                clipLimit = 5.0       
                tileGridSize = (8,8)
            class UnsharpMask:
                kernel = (5,3)
                sigma = 0         
                strength = 0.5 
        class Focus:
            class Entropy:
                disk_size = 50
            class HED:
                detect_resolution = 512
            class Weights:
                hed = 1
                entropy = 1.0
                diff = 1.5
            class Morph:
                class Kernels:
                    Dilate = (9, 9)
                    Erode = (9,9)
                class Its:
                    Dilate = 5
                    Erode = 7
                AreaMin = 150
            class EmptyCheck:   
                TopPercent = 0.5
                MeanThresh = 50
        class SmartPrompt:
            MinArea = 150
            GridSize = 25
            NegativeRing = 20
            SafeMargin = 30
            MaxPosPts = 15        
            MaxNegPts = 15
        class Segmenter: 
            Model = 'SamHq'
            Checkpoint = 'sam_hq_vit_h.pth'
            Name = 'vit_h'
        class Group:
            AreaMin = 150             
            Similarity = 0.25     
        class Collapse:
                PercentileThresh = 10.0    
                MinMaxPercentage = 0.1
        class Cloud:
            BlurKernel = [9, 9]     
            Relaxation = 0.85     
            DilateKernel = [9, 9]  
            DilateIter = 1       
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0  