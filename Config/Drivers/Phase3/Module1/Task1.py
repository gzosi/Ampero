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
        Bounds = [[0, 35]] 
        class DynamicROI:
            Camera1 = {
                0 : np.array([
                    [401, 408], [211, 283], [118, 190], 
                    [176, 105], [337, 180], [441, 340]]),
                15 : np.array([
                    [398, 448], [224, 377], [128, 295], 
                    [139, 181], [296, 228], [446, 393]]),
                35 : np.array([
                    [437, 513], [282, 532], [136, 485], 
                    [128, 337], [288, 376], [447, 455]]),
            }
            Camera2 = {
                0 : np.array([
                    [451, 573], [361, 503], [244, 391], 
                    [237, 276], [384, 364], [475, 505]]),
                15 : np.array([
                    [428, 597], [295, 544], [190, 462], 
                    [183, 349], [334, 404], [456, 540]]),
                35 : np.array([
                    [443, 626], [339, 665], [194, 616], 
                    [189, 501], [308, 500], [424, 568]]),
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
                disk_size = 25
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