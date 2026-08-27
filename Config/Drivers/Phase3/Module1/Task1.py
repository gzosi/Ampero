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
                    [766, 354], [576, 413], [432, 412], 
                    [347, 264], [509, 216], [762, 312]]),
                10 : np.array([
                    [667, 365], [506, 410], [355, 439], 
                    [275, 269], [475, 250], [655, 324]]),
                20 : np.array([
                    [580, 357], [412, 465], [307, 492], 
                    [216, 319], [416, 266], [570, 314]]),
                30 : np.array([
                    [521, 357], [405, 460], [252, 506], 
                    [192, 350], [413, 283], [514, 319]]),
                40 : np.array([
                    [463, 368], [373, 482], [258, 547], 
                    [176, 429], [372, 299], [436, 339]]),
            }

            Camera2 = {
                0 : np.array([
                    [515, 277], [367, 345], [180, 366], 
                    [111, 194], [312, 163], [511, 237]]),
                10 : np.array([
                    [407, 271], [271, 338], [124, 364], 
                    [66, 218], [265, 182], [402, 237]]),
                20 : np.array([
                    [365, 273], [262, 345], [72, 402], 
                    [7, 258], [217, 180], [357, 237]]),
                30 : np.array([
                    [288, 280], [172, 367], [45, 385], 
                    [12, 264], [172, 182], [252, 216]]),
                40 : np.array([
                    [183, 300], [110, 381], [24, 403], 
                    [11, 255], [74, 210], [146, 223]]),
            }

        class Enhancement:
            class Bilateral:
                d = 15  
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
                disk_size = 100
            class HED:
                detect_resolution = 512
            class Weights:
                hed = 1.5
                entropy = 1.5
                diff = 0.0
            class Morph:
                class Kernels:
                    Dilate = (3, 3)
                    Erode = (3,3)
                class Its:
                    Dilate = 9
                    Erode = 9
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