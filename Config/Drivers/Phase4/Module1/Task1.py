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
                    [299, 340], [204, 328], [154, 285], 
                    [180, 215], [260, 179], [341, 270]]),
                15 : np.array([
                    [309, 363], [202, 349], [151, 317], 
                    [195, 222], [274, 205], [324, 296]]),
                20 : np.array([
                    [294, 390], [197, 393], [148, 363], 
                    [161, 265], [258, 250], [321, 331]]),
            }

            Camera2 = {
                10 : np.array([
                    [373, 491], [290, 491], [221, 451], 
                    [222, 360], [318, 349], [396, 407]]),
                15 : np.array([
                    [363, 501], [271, 511], [224, 477], 
                    [236, 396], [315, 378], [404, 448]]),
                20 : np.array([
                    [349, 533], [277, 548], [213, 511], 
                    [224, 419], [288, 399], [359, 480]]),
            }
        class Zscore:
            zTh = 8
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