#!/usr/bin/env python3
"""
HACK le fichier cpbd/compute.py
    cpbd : cannot import name 'imread' from 'scipy.ndimage' --> import a commenter (compute.py)
"""
import os
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec('cpbd')
path_cpbd = spec.submodule_search_locations[0]
nameFile = os.path.join(path_cpbd, 'compute.py')

if not Path(nameFile).is_file():
    raise FileNotFoundError(f"{nameFile}")

with open(nameFile, 'r') as f:
    lines = f.readlines()
with open(nameFile, 'w') as f:
    for line in lines:
        line = line.replace('from scipy.ndimage import imread', '# from scipy.ndimage import imread')
        f.write(line)

print(f"Le fichier {nameFile} a été modifié !")
#EOF