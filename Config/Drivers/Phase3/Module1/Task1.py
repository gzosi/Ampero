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
        Bounds = [[0, 45]] 
        class DynamicROI:
            Camera1 = {
                0 : np.array([
                    [735, 309], [586, 216], [507, 151], 
                    [524, 95], [656, 128], [765, 258]]),
                10 : np.array([
                    [761, 356], [606, 298], [481, 205], 
                    [514, 105], [629, 152], [772, 305]]),
                20 : np.array([
                    [755, 384], [610, 337], [490, 273], 
                    [495, 152], [608, 177], [771, 322]]),
                30 : np.array([
                    [782, 406], [619, 382], [499, 328], 
                    [499, 187], [644, 256], [780, 342]]),
                40 : np.array([
                    [815, 427], [702, 437], [536, 397], 
                    [499, 255], [679, 301], [813, 381]]),
                45 : np.array([
                    [835, 436], [713, 473], [548, 451], 
                    [510, 303], [674, 354], [822, 397]]),
            }
            Camera2 = {
                0 : np.array([
                    [773, 493], [652, 417], [567, 352], 
                    [555, 255], [679, 301], [780, 443]]),
                10 : np.array([
                    [756, 499], [621, 458], [520, 368], 
                    [515, 272], [644, 307], [761, 441]]),
                20 : np.array([
                    [730, 515], [617, 490], [518, 438], 
                    [496, 316], [636, 364], [742, 465]]),
                30 : np.array([
                    [752, 549], [629, 528], [519, 481], 
                    [480, 361], [621, 405], [755, 488]]),
                40 : np.array([
                    [778, 563], [671, 582], [527, 539], 
                    [489, 419], [627, 444], [753, 499]]),
                45 : np.array([
                    [788, 580], [672, 593], [531, 549], 
                    [488, 439], [648, 480], [766, 532]]),
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