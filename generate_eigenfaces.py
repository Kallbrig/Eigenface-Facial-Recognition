import PIL
import numpy as np
import cv2
import time
import os
from PIL import Image
import fnmatch
from statistics import median, mode, mean


# used to scale photos so they all end up the same size
def scale(im, nR, nC):
    nR0 = len(im)  # source number of rows
    nC0 = len(im[0])  # source number of columns
    return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)]


def generate_avg_eigenface(name: str):
    # Makes directory to hold the eigenfaces.
    # This method is OS independent. It works on OSX, Windows and Linux
    try:
        os.mkdir(os.path.join(f'training photos', 'eigenfaces', name))
    except:
        pass

    face_list = []
    new_photo = []

    for i in [name for name in
              fnmatch.filter(os.listdir(os.path.join('training photos', 'capture', name)), '*.jpg')]:
        img = cv2.imread(os.path.join('training photos', 'capture', name, i), cv2.IMREAD_GRAYSCALE)

        img = np.array(scale(img, 200, 150))

        face_list.append(img)

    for line in range(200):
        row = []
        for pixel in range(150):

            avg_pix_list = []
            for photo in range(len(face_list) - 1):
                avg_pix_list.append(face_list[photo][line][pixel])

            avg_pix = sum(avg_pix_list) / len(avg_pix_list)

            row.append(round(avg_pix))
        new_photo.append(row)

    new_photo = np.array(new_photo).astype(np.uint8)

    # print(new_photo)

    cv2.imwrite(
        os.path.join('training photos', 'eigenfaces', name, f'{name}_avg_eigenface.jpg'),
        new_photo)
    print('Average Eigenface Generated!')


def generate_mode_eigenface(name: str):
    try:
        os.mkdir(os.path.join(f'training photos', 'eigenfaces', name))
    except:
        pass

    face_list = []
    new_photo = []

    for i in [name for name in
              fnmatch.filter(os.listdir(os.path.join('training photos', 'capture', name)),
                             '*.jpg')]:
        img = cv2.imread(os.path.join('training photos', 'capture', name, i), cv2.IMREAD_GRAYSCALE)

        img = np.array(scale(img, 200, 150))

        face_list.append(img)

    for line in range(200):
        row = []
        for pixel in range(150):

            mode_pix_list = []
            for photo in range(len(face_list) - 1):
                mode_pix_list.append(face_list[photo][line][pixel])

            mode_pix = mode(mode_pix_list)

            row.append(round(mode_pix))
        new_photo.append(row)

    new_photo = np.array(new_photo).astype(np.uint8)

    cv2.imwrite(
        os.path.join('training photos', 'eigenfaces', name, f'{name}_mode_eigenface.jpg'),
        new_photo)
    print('Mode Eigenface Generated!')


def generate_median_eigenface(name: str):
    try:
        os.mkdir(os.path.join(f'training photos', 'eigenfaces', name))
    except:
        pass

    face_list = []
    new_photo = []

    for i in [name for name in fnmatch.filter(os.listdir(os.path.join('training photos', 'capture', name)), '*.jpg')]:
        img = cv2.imread(os.path.join('training photos', 'capture', name, i), cv2.IMREAD_GRAYSCALE)

        img = np.array(scale(img, 200, 150))

        face_list.append(img)

    for line in range(200):
        row = []
        for pixel in range(150):

            median_pixel_list = []
            for photo in range(len(face_list) - 1):
                median_pixel_list.append(np.uint16(face_list[photo][line][pixel]))

            median_pix = median(median_pixel_list)
            # print(median_pix.astype(np.uint8))

            row.append(median_pix.astype(np.uint16))
        new_photo.append(row)

    new_photo = np.array(new_photo).astype(np.uint16)

    # print(new_photo)

    cv2.imwrite(
        os.path.join('training photos', 'eigenfaces', name, f'{name}_median_eigenface.jpg'),
        new_photo)
    print('Median Eigenface Generated!')


def generate_all_eigenfaces(name: str):
    generate_avg_eigenface(name=name)
    generate_mode_eigenface(name=name)
    generate_median_eigenface(name=name)
    print('\nAll Eigenfaces Generated')


if __name__ == '__main__':
    name = input('Please input your name.\n')
    generate_all_eigenfaces(name=name)
