import os
from pathlib import Path
import shutil
from shutil import copyfile



directory = os.fsencode("/research/axns2/mabdelfa/zerowaste/zerowaste-w/seg/test/before")
    
for file in os.listdir(directory):
    
    filename = os.fsdecode(file)
    print(filename)

    if filename.endswith(".PNG"): 
        source = "/research/axns2/mabdelfa/zerowaste/zerowaste-w/seg/test/before/" + filename
        shutil.copy2(source, "/research/axns2/mabdelfa/zerowaste/zerowaste-w/seg/val_test/before")