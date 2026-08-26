import numpy as np
import cv2 as cv
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
        Bounds = [[0, 40]] 
        class DynamicROI:
            Camera1 = {
                0 : np.array([[859, 371], [667, 268], [567, 192], [584, 98], [708, 128], [873, 327]]),
                10 : np.array([[827, 386], [662, 332], [566, 268], [574, 132], [720, 214], [836, 344]]),
                20 : np.array([[861, 413], [732, 415], [579, 358], [572, 196], [735, 262], [859, 360]]),
                30 : np.array([[893, 439], [760, 475], [606, 452], [577, 290], [724, 336], [889, 388]]),
                40 : np.array([[942, 462], [810, 512], [637, 494], [612, 366], [760, 419], [915, 412]]),
            }
            Camera2 = {
                0 : np.array([[842, 501], [686, 445], [589, 339], [593, 259], [759, 328], [868, 460]]),
                10 : np.array([[849, 541], [713, 508], [591, 456], [557, 320], [732, 380], [858, 479]]),
                20 : np.array([[859, 567], [732, 567], [616, 531], [574, 368], [712, 395], [863, 507]]),
                30 : np.array([[901, 597], [757, 625], [620, 592], [590, 450], [716, 483], [892, 533]]),
                40 : np.array([[874, 637], [749, 654], [629, 658], [591, 508], [713, 519], [827, 594]]),
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
                disk_size = 25
            class HED:
                detect_resolution = 512
            class Weights:
                hed = 1.0
                entropy = 0.5
                diff = 1.5
            class Morph:
                class Kernels:
                    Dilate = (5, 5)
                    Erode = (5, 5)
                class Its:
                    Dilate = 3
                    Erode = 5
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