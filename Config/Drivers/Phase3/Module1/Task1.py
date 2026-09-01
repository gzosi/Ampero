import numpy as np
import cv2 as cv
class Task1:
    class MetaData:
        OutputName = 'Data'
        OutputExt = '.pkl'
    class Settings:
        class Src:
            Database = 'Database3'
            Dataset = 'Dataset2'
            Foreground = 'Foreground'
            Background = 'Background'
        Bounds = [[0, 40]] 
        class DynamicROI:
            Camera1 = {
                0 : np.array([
                    [767, 351], [611, 420], [429, 416], 
                    [342, 271], [541, 245], [767, 318]]),
                10 : np.array([
                    [739, 354], [541, 422], [361, 457], 
                    [292, 303], [487, 263], [735, 324]]),
                20 : np.array([
                    [639, 355], [440, 458], [303, 479], 
                    [198, 318], [480, 262], [630, 316]]),
                30 : np.array([
                    [570, 375], [434, 474], [282, 541], 
                    [292, 357], [406, 295], [542, 313]]),
                40 : np.array([
                    [486, 374], [414, 479], [282, 571], 
                    [302, 368], [389, 308], [427, 306]]),
            }

            Camera2 = {
                0 : np.array([
                    [536, 274], [371, 352], [165, 362], 
                    [82, 206], [315, 167], [526, 225]]),
                10 : np.array([
                    [442, 286], [291, 360], [111, 386], 
                    [52, 210], [273, 185], [436, 246]]),
                20 : np.array([
                    [373, 285], [207, 368], [67, 401], 
                    [18, 261], [216, 190], [371, 244]]),
                30 : np.array([
                    [273, 252], [184, 336], [59, 396], 
                    [53, 281], [156, 202], [233, 195]]),
                40 : np.array([
                    [197, 310], [131, 397], [44, 434], 
                    [29, 328], [94, 250], [154, 236]]),
            }
        class Enhancement:
            class Bilateral:
                d = 5  
                sigmaColor = 15     
                sigmaSpace = 15
            class CLAHE:
                clipLimit = 3.0       
                tileGridSize = (8,8)
            class UnsharpMask:
                kernel = (3,3)
                sigma = 0         
                strength = 0.5 
        class Focus:
            class Entropy:
                disk_size = 5
            class HED:
                detect_resolution = 512
            class Weights:
                hed = 1.5
                entropy = 1.25
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
            NegativeRing = 10
            SafeMargin = 30
            MaxPosPts = 25       
            MaxNegPts = 10
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