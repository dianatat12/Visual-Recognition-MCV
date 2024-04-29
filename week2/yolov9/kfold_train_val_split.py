import os 
import shutil
import random

dataset_dir = '/ghome/group04/MCV-C5-G4/week2/optional_task/dataset/train'

dest_dir = '/ghome/group04/MCV-C5-G4/week2/optional_task/dataset/train/splits'

def get_video_nr(file):
    return file.split('_')[0]

def make_images_labels_dirs(directory):
    os.makedirs(os.path.join(directory, 'images'))
    os.makedirs(os.path.join(directory, 'labels'))


imgs_src_dir = os.path.join(dataset_dir, 'images')
labs_src_dir = os.path.join(dataset_dir, 'labels')

image_files = sorted(os.listdir(imgs_src_dir))
label_files = [file.replace('png', 'txt') for file in image_files]

video_nrs = []
for image in image_files:
    video_nrs.append(get_video_nr(image))

video_nrs = list(set(video_nrs))

for k in range(1,5):

    fold_dir = os.path.join(dest_dir, str(k))

    fold_train_dir = os.path.join(fold_dir, 'train')
    make_images_labels_dirs(fold_train_dir)

    fold_test_dir = os.path.join(fold_dir, 'test')
    make_images_labels_dirs(fold_test_dir)

    fold_end_index = 3*k
    fold_start_index = fold_end_index - 3

    val_videos = video_nrs[fold_start_index:fold_end_index]

    for i, image in enumerate(image_files):
        video_nr = get_video_nr(image)

        if video_nr in val_videos:
            img_src = os.path.join(imgs_src_dir, image)
            lab_src = os.path.join(labs_src_dir, label_files[i])

            img_dest = os.path.join(fold_test_dir, 'images', image)
            lab_dest = os.path.join(fold_test_dir, 'labels', label_files[i])

            shutil.copy(img_src, img_dest)
            shutil.copy(lab_src, lab_dest)
        
        if video_nr not in val_videos:
            img_src = os.path.join(imgs_src_dir, image)
            lab_src = os.path.join(labs_src_dir, label_files[i])

            img_dest = os.path.join(fold_train_dir, 'images', image)
            lab_dest = os.path.join(fold_train_dir, 'labels', label_files[i])

            shutil.copy(img_src, img_dest)
            shutil.copy(lab_src, lab_dest)

