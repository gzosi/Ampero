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
            Dataset = 'Dataset2'
            Location = 'Background'
        Bounds = [[0, 20]] 
        class DynamicROI:
            Camera1 = {
                0 : np.array([[750, 354], [474, 375], [341, 327],[347, 214], [494, 217], [727, 279]]),
                10 : np.array([[741, 351], [528, 403], [303, 384], [298, 279], [508, 241], [702, 280]]),
                20 : np.array([[679, 348], [508, 418], [307, 438], [281, 340], [439, 257], [645, 287]]),
            }
            Camera2 = {
                0 : np.array([[514, 272], [280, 322], [87, 295], [75, 167], [273, 146], [484, 203]]),
                10 : np.array([[464, 274], [284, 312], [78, 318], [59, 192], [234, 170], [425, 207]]),
                20 : np.array([[392, 274], [243, 323], [77, 346], [55, 216], [231, 166], [358, 218]]),
            }
        class Zscore:
            zTh = 3.5
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