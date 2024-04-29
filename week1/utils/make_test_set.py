import os 
import shutil
import random


directory = '/ghome/group04/MCV-C5-G4/MIT_small_train_1/test'
output_dir = '/ghome/group04/MCV-C5-G4/test'
classes = os.listdir(directory)

for class_ in classes:
    class_dir_path = os.path.join(directory, class_)
    files = os.listdir(class_dir_path)
    num_files = int(0.1 * len(files))   
    sample = random.sample(files, num_files)
    for file in sample:
        class_dest_dir = os.path.join(output_dir, class_)
        if not os.path.exists(class_dest_dir):
            os.makedirs(class_dest_dir)
        
        src = os.path.join(class_dir_path, file)
        dest = os.path.join(class_dest_dir, file)

        shutil.move(src, dest)