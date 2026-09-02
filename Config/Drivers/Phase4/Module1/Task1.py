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
                    [817, 374], [636, 311], [572, 268], 
                    [577, 127], [686, 155], [845, 308]]),
                15 : np.array([
                    [830, 388], [666, 348], [588, 303], 
                    [569, 152], [679, 161], [850, 312]]),
                20 : np.array([
                    [836, 420], [688, 380], [596, 334], 
                    [576, 171], [674, 197], [864, 348]]),
            }

            Camera2 = {
                10 : np.array([
                    [853, 519], [716, 480], [591, 419], 
                    [569, 278], [679, 302], [876, 459]]),
                15 : np.array([
                    [853, 539], [716, 507], [586, 451], 
                    [571, 312], [679, 329], [868, 476]]),
                20 : np.array([
                    [854, 567], [717, 528], [603, 510], 
                    [574, 337], [686, 349], [870, 496]]),
            }   
        class Zscore:
            zTh = 4.5
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