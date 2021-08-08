import os
from pathlib import Path
import shutil
from shutil import copyfile



directory = os.fsencode("/research/axns2/mabdelfa/zerowaste/zerowaste-w/seg/val/before")
    
count = 0
for file in os.listdir(directory):
    
    filename = os.fsdecode(file)
    print(filename)

    if filename.endswith(".PNG"): 
        source = "/research/axns2/mabdelfa/zerowaste/zerowaste-w/seg/val/before/" + filename
        shutil.move(source, "/research/axns2/mabdelfa/zerowaste/zerowaste-w/seg/test/before")
        count += 1
    
    if count == 601:
        quit()