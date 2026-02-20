#%% Importing Libreries
import os
from pathlib import Path
import inspect
import runpy
import subprocess
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
def destroyerCheck(config_file):
    '''This function is used to check if the driver file contains a class with a Maker attribute. It uses the runpy module to execute the Python file and inspect its contents.'''
    try:
        mod = runpy.run_path(config_file)
    except Exception:
        return None
    classes = [v for v in mod.values() if inspect.isclass(v)]
    if len(classes) != 1:
        return None
    MainClass = classes[0]
    try:
        destroyer_value = MainClass.General.Destroyer
        return destroyer_value
    except AttributeError:
        return None
def destroyerRoot(relative_path_str, root_str):
    ''' This function is used to delete a directory structure based on the relative path of the driver file and a specified root directory. It uses the subprocess module to delete the directories.'''
    full_path = Path(root_str) / Path(relative_path_str).with_suffix("")
    if full_path.exists() and full_path.is_dir():
        subprocess.run(
            f'rmdir /S /Q "{full_path}"',
            shell=True, check=True)
    current = full_path.parent
    archive_root = Path(root_str).resolve()
    while current != archive_root and current.exists():
        try:
            if not any(current.iterdir()):
                subprocess.run(
                f'rmdir /S /Q "{current}"',
                shell=True, check=True)
            current = current.parent
        except Exception as e:
            break
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase0.Modules.Module1.Tasks.Task2.General.Activation is True:
        print('.... Task2:', colored( 'Running ℹ️', 'cyan'))
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
            destroyer = destroyerCheck(config_file)
            if destroyer is True:
                destroyerRoot(
                    os.path.join(Config.Paths.CodeRoots.DriversRoot, driver_rel_path),
                    archive_root)
        print('.... Task2:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase0.Modules.Module1.Tasks.Task2.General.Activation is False:
        print('.... Task2:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Task2 Switch (on/off) ❌')
    return