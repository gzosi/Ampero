import numpy as np
#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Src:
            Database = 'Database4'
        class Ref:
            Database = 'Database3'
            Dataset = 'Dataset1'
            Location = 'Background'
        Bounds = [[10, 20]] 
        class DynamicROI:
            Camera1 = {
                10 : np.array([
                    [722, 291], [628, 283], [582, 247], 
                    [585, 131], [669, 149], [717, 256]]),
                15 : np.array([
                    [704, 303], [642, 316], [581, 296], 
                    [582, 150], [665, 193], [707, 265]]),
                20 : np.array([
                    [712, 334], [660, 347], [601, 326], 
                    [579, 210], [685, 239], [716, 298]]),
            }
            Camera2 = {
                10 : np.array([
                    [725, 439], [688, 449], [604, 417], 
                    [577, 321], [665, 310], [737, 400]]),
                15 : np.array([
                    [738, 474], [674, 481], [602, 435], 
                    [582, 342], [670, 335], [747, 428]]),
                20 : np.array([
                    [739, 502], [664, 492], [615, 471], 
                    [573, 372], [671, 365], [744, 421]]),
            }
        class Zscore:
            zTh = 4.75
            epsilon = 1e-5
            class Morph:
                openKernel = (3,3)
                closeKernel = (5,5)
        class SmartPrompt:
            MinArea = 10             
            TargetPts = 4           
            NumNegativePts = 8      
            NegativeRingBase = 8   
            BboxPaddingBase = 8     
        class Segmenter: 
            Model = 'SamHq'
            Checkpoint = 'sam_hq_vit_h.pth'
            Name = 'vit_h'
        class ContainmentZone:
            Kernel = (25,25)      
            minArea = 10        
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0