import os
import re
import shutil
import numpy as np
from PIL.Image import Image
import skimage
import pandas as pd
from tqdm import tqdm


def convert_tilda_2classes(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    path2defect_less = os.path.join(output_dir, 'defect_free')
    path2defect = os.path.join(output_dir, 'defect')
    if not os.path.exists(path2defect_less):
        os.mkdir(path2defect_less)
    if not os.path.exists(path2defect):
        os.mkdir(path2defect)

    non_defect_regex = re.compile('.*e0.*tif')
    defect_regex = re.compile('.*e[^0].*tif')
    non_defect_images = []
    defect_images = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if non_defect_regex.match(file):
                non_defect_images.append([os.path.join(root, file), file])
    print(len(non_defect_images))

    for root, _, files in os.walk(input_dir):
        for file in files:
            if defect_regex.match(file):
                defect_images.append([os.path.join(root, file), file])
    print(len(defect_images))

    for file in non_defect_images:
        shutil.copyfile(file[0], os.path.join(output_dir, "defect_free", file[1]))
    for file in defect_images:
        shutil.copyfile(file[0], os.path.join(output_dir, "defect", file[1]))


def parse_tilda(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for n in range(1, 8):
        class_label = 'e' + str(n)
        path = os.path.join(output_dir, class_label)
        if not os.path.exists(path):
            os.mkdir(path)

        class_regex = re.compile(f'.*{class_label}.*tif')
        class_images = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if class_regex.match(file):
                    class_images.append([os.path.join(root, file), file])
        print(f"{class_label} - {len(class_images)}.")

        for file in class_images:
            shutil.copyfile(file[0], os.path.join(output_dir, class_label, file[1]))


def get_patches(image, n_rows, n_cols):
    patches = []
    h, w = image.shape[:2]
    h //= n_rows
    w //= n_cols
    for i in range(n_rows):
        for j in range(n_cols):
            patches.append(image[i * h: (i + 1) * h, w * j: (j + 1) * w])
    return patches


def get_severtal_steel_patches(input_data_dir, output_data_dir, n_rows, n_cols):
    if not os.path.exists(f"{output_data_dir}/data"):
        os.makedirs(f"{output_data_dir}/data")
    if not os.path.exists(f"{output_data_dir}/data/0"):
        os.makedirs(f"{output_data_dir}/data/0")

    df = pd.read_csv(f"{input_data_dir}/train.csv")
    normal_dir = f"./data/0/"
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_name = row[0]
        image_path = f"{input_data_dir}/train_images/{image_name}"
        class_id = row[1]
        encoded_pixels = row[2].split(" ")
        pixels = encoded_pixels[::2]
        pixels = list(map(int, pixels))
        steps = encoded_pixels[1::2]
        steps = list(map(int, steps))
        class_dir = f"{output_data_dir}/data/{class_id}"

        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

        img = Image.open(image_path)
        img = np.asarray(img)

        patches = get_patches(img, n_rows, n_cols)

        defect_tensor = np.zeros(img.shape[0] * img.shape[1])
        for pixel, step in zip(pixels, steps):
            defect_tensor[pixel:pixel + step] = 1

        defect_tensor = defect_tensor.reshape(img.shape[0], img.shape[1])
        h = img.shape[0] // n_rows
        w = img.shape[1] // n_cols
        defect_tensor = skimage.measure.block_reduce(defect_tensor, (h, w), np.mean)
        for row_id in range(defect_tensor.shape[0]):
            for col_id in range(defect_tensor.shape[1]):
                patch_id = row_id * 8 + col_id
                patch = patches[patch_id]
                patch = Image.fromarray(patch)

                if defect_tensor[row_id, col_id] > 0.01:
                    patch.save(f"{output_data_dir}/{class_dir}/{image_name.split('.')[0]}_{patch_id}.jpg")
                else:
                    patch.save(f"{output_data_dir}/{normal_dir}/{image_name.split('.')[0]}_{patch_id}.jpg")


def split_folder(root_dir, test_ratio=0.3, max_val=8000):
    folders = os.listdir(root_dir)

    os.makedirs(root_dir + '/train/')
    os.makedirs(root_dir + '/test/')

    for folder in folders:

        source = os.path.join(root_dir, folder)
        file_names = os.listdir(os.path.join(root_dir, folder))
        if len(file_names) > max_val:
            file_names = file_names[:max_val]

        np.random.shuffle(file_names)

        train_files, test_files = np.split(np.array(file_names),
                                           [int(len(file_names) * (1 - test_ratio))])

        train_paths = [source + '/' + name for name in train_files.tolist()]
        test_paths = [source + '/' + name for name in test_files.tolist()]

        os.makedirs(root_dir + '/train/' + folder)
        os.makedirs(root_dir + '/test/' + folder)

        for file_path, file_name in zip(train_paths, train_files):
            shutil.move(file_path, os.path.join(root_dir, 'train', folder, file_name))

        for file_path, file_name in zip(test_paths, test_files):
            shutil.move(file_path, os.path.join(root_dir, 'test', folder, file_name))

        shutil.rmtree(source, ignore_errors=True)
