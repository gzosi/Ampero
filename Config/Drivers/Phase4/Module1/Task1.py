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
        Bounds = [[15, 20, 25]] 
        class DynamicROI:
            Camera1 = {
                15 : np.array([
                    [623, 272], [545, 271], [470, 229], 
                    [512, 112], [581, 125], [654, 219]]),
                20 : np.array([
                    [636, 303], [532, 299], [480, 269], 
                    [505, 132], [595, 153], [669, 241]]),
                25 : np.array([
                    [639, 326], [558, 346], [483, 318], 
                    [497, 166], [580, 171], [668, 253]]),
            }

            Camera2 = {
                15 : np.array([
                    [642, 429], [554, 431], [491, 400], 
                    [471, 278], [583, 291], [664, 356]]),
                20 : np.array([
                    [630, 463], [541, 461], [491, 443], 
                    [473, 308], [570, 311], [653, 395]]),
                25 : np.array([
                    [635, 478], [555, 495], [496, 493], 
                    [486, 340], [575, 332], [645, 404]]),
            }   
        class Zscore:
            zTh = 4.0
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