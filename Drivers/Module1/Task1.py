#%% Importing Libreries
import os
import inspect
import runpy
from pathlib import Path
from termcolor import colored
#%% Defining Subroutines
def driverExplorer(drivers_root):
    ''' This function is used to explore the directory tree of the drivers_root directory and find all Python files (.py) within it.'''
    driver_files = []
    for dirpath, _, filenames in os.walk(drivers_root):
        for f in filenames:
            if f.endswith('.py'):
                full_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(full_path, drivers_root)
                driver_files.append((full_path, rel_path))
    return driver_files
def makerCheck(config_file):
    ''' This function is used to check if the driver file contains a class with a Maker attribute. It uses the runpy module to execute the Python file and inspect its contents.'''
    try:
        mod = runpy.run_path(config_file)
    except Exception as e:
        return None
    classes = [v for v in mod.values() if inspect.isclass(v)]
    if len(classes) != 1:
        return None
    MainClass = classes[0]
    try:
        maker_value = MainClass.General.Maker
        return maker_value
    except AttributeError:
        return None
def rootMaker(relative_path_str, root_str):
    ''' This function is used to create a directory structure based on the relative path of the driver file and a specified root directory. It uses the pathlib module to create the directories.'''
    root = Path(root_str)
    relative_path = Path(relative_path_str)
    if relative_path.suffix:
        relative_path = relative_path.with_suffix("")
    full_path = root / relative_path
    full_path.mkdir(parents=True, exist_ok=True)
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase0.Modules.Module1.Tasks.Task1.General.Activation is True:
        print('.... Task1:', colored( 'Running ℹ️', 'cyan'))
        drivers_root = os.path.join(
            Config.Paths.mainRooot,
            Config.Paths.CodeRoots.DriversRoot)
        config_root = os.path.join(
            Config.Paths.mainRooot,
            Config.Paths.CodeRoots.ConfigRoot)
        archive_root = os.path.join(
            Config.Paths.mainRooot,
            Config.Paths.DataRoots.ResourcesRoot,
            Config.Paths.DataRoots.StreamRoot,
            Config.Paths.DataRoots.CaseStudyRoot())
        driver_files = driverExplorer(drivers_root)
        for _, driver_rel_path in driver_files:
            config_file = os.path.join(
                config_root, 
                Config.Paths.CodeRoots.DriversRoot,
                driver_rel_path)
            maker = makerCheck(config_file)
            if maker is True:
                rootMaker(os.path.join(
                    Config.Paths.CodeRoots.DriversRoot, 
                    driver_rel_path), archive_root)
        print('.... Task1:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase0.Modules.Module1.Tasks.Task1.General.Activation is False:
        print('.... Task1:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Task1 Switch (on/off) ❌')
    return