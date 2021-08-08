#before: /research/axns2/mabdelfa/zerowaste/zerowaste-w/before
#after: /research/axns2/mabdelfa/zerowaste/zerowaste-w/after
#Segemntation maps locations:

### /research/axns2/mabdelfa/zerowaste/zerowaste-f/train/sem_seg
### /research/axns2/mabdelfa/zerowaste/zerowaste-f/val/sem_seg
### /research/axns2/mabdelfa/zerowaste/zerowaste-f/test/sem_seg


import os
from pathlib import Path
import shutil
from shutil import copyfile



directory = os.fsencode("/research/axns2/mabdelfa/zerowaste/zerowaste-w/after")
    
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".PNG"): 
         #print(filename)
         my_file = Path("/research/axns2/mabdelfa/zerowaste/zerowaste-f/train/sem_seg/"+filename)
         if my_file.is_file():
             print("File exists in train, copying....")
             shutil.copy2(my_file, "/research/axns2/mabdelfa/zerowaste/zerowaste-w/segmentation_gt/after")
         else:
             my_file = Path("/research/axns2/mabdelfa/zerowaste/zerowaste-f/val/sem_seg/"+filename)
             if my_file.is_file():
                print("File exists in val, copying....")
                shutil.copy2(my_file, "/research/axns2/mabdelfa/zerowaste/zerowaste-w/segmentation_gt/after")
             else:
                 my_file = Path("/research/axns2/mabdelfa/zerowaste/zerowaste-f/test/sem_seg/"+filename)
                 if my_file.is_file():
                     print("File exists in test, copying....")
                     shutil.copy2(my_file, "/research/axns2/mabdelfa/zerowaste/zerowaste-w/segmentation_gt/after")
         continue
     else:
         continue