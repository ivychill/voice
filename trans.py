import warnings

import cv2
import librosa
import os
from pathlib import Path
import numpy as np
import torch.utils.data as data
from torchvision import datasets


def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def transform_voice(fname):
    melspectrogram_parameters = {
        "n_mels": 128,
        "fmin": 20,
        "fmax": 16000
    }
    SR = 32000
    img_size = 224
    y, _ = librosa.load(fname,
                        sr=SR,
                        mono=True,
                        res_type="kaiser_fast")

    y = y.astype(np.float32)
    len_y = len(y)
    start = 0
    end = SR * 5
    images = []
    while len_y > start:
        y_batch = y[start:end].astype(np.float32)
        if len(y_batch) != (SR * 5):
            break
        start = end
        end = end + SR * 5

        melspec = librosa.feature.melspectrogram(y_batch, sr=SR, **melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * img_size / height), img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)
        images.append(image)

    return images


# root = Path('input/person/train_b_resampled')


if __name__ == "__main__":
    src_root = 'input/person/train_b_resampled'
    # src_root = '/home/kcadmin/user/fengchen/voice/tianchi/datasets/test'
    dst_root = 'input/person/train_b_npy'

    src_root = os.path.expanduser(src_root)
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    persons = os.listdir(src_root)
    for person in sorted(persons):
        src_dir = os.path.join(src_root, person)
        dst_dir = os.path.join(dst_root, person)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for fname in sorted(os.listdir(src_dir)):
            print(fname)
            src_path = os.path.join(src_dir, fname)
            images = transform_voice(src_path)
            for i, image in enumerate(images):
                dst_path = os.path.join(dst_dir, fname + '_' + str(i) + '.npy')
                # print('image', image.shape, image[0, 0:5, 0:5])
                np.save(dst_path, image)
