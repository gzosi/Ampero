from Config import Config
import h5py
import matplotlib.pyplot as plt
from pathlib import Path   
import cv2 as cv 

main_root = Path(Config.Paths.mainRooot)
srcRoot = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot /
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ / 
    Config.Packages.Drivers.Phases.Phase0.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.MetaData.OutputName)
with h5py.File(srcRoot, 'r') as f:
    print(f['Camera1']['Database3']['Dataset1'].keys())

    img1 = f['Camera1']['Database3']['Dataset1']['Foregroung']['00008'][:]
    img2 = f['Camera1']['Database3']['Dataset1']['Backgroung']['00021'][:]
    img3 = f['Camera1']['Database2']['Dataset1']['00001'][:]
    img4 = f['Camera1']['Database4']['Dataset1']['00081'][:]
    img5 = f['Camera1']['Database4']['Dataset2']['00168'][:]
    img6 = f['Camera1']['Database4']['Dataset3']['00182'][:]
    img7 = f['Camera1']['Database4']['Dataset4']['00110'][:]
    img8 = f['Camera1']['Database4']['Dataset5']['00239'][:]
    img9 = f['Camera1']['Database4']['Dataset6']['00145'][:]
    img10 = f['Camera1']['Database4']['Dataset7']['00241'][:]
    img11 = f['Camera1']['Database4']['Dataset8']['00117'][:]
    img12 = f['Camera1']['Database4']['Dataset9']['00078'][:]
    img13 = f['Camera1']['Database4']['Dataset10']['00042'][:]
    img14 = f['Camera1']['Database4']['Dataset11']['00214'][:]
img = cv.absdiff(img14, img4)
plt.figure(figsize = (20,20))
plt.imshow(img, cmap = 'gray')
plt.tight_layout()
plt.axis('off')