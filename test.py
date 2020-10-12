
import cv2
import audioread
import logging
import os
import random
import time
import warnings

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastprogress import progress_bar
from sklearn.metrics import f1_score
from torchvision import models


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


logger = get_logger("output/test.log")
set_seed(1213)

TARGET_SR = 32000
# TEST = Path("./input/birdsong-recognition/test_audio").exists()
# if TEST:
#     DATA_DIR = Path("../input/birdsong-recognition/")
# else:
#     # dataset created by @shonenkov, thanks!
#     DATA_DIR = Path("../input/birdcall-check/")

test_audio_dir = Path("./input/person/testB/")
# N_CLASS = 50
N_CLASS = 52

# test = pd.read_csv(DATA_DIR / "test.csv")
# test_audio = DATA_DIR / "test"
# test.head()
# sub = pd.read_csv("../input/birdsong-recognition/sample_submission.csv")
# sub.to_csv("submission.csv", index=False)  # this will be overwritten if everything goes well


class ResNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(
            pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x).view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = F.sigmoid(x)
        return {
            "logits": x,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }

model_config = {
    # "base_model_name": "resnet50",
    "base_model_name": "resnet50",
    "pretrained": False,
    "num_classes": N_CLASS
}

melspectrogram_parameters = {
    "n_mels": 128,
    "fmin": 20,
    "fmax": 16000
}

weights_path = "output/b_ResNet50/fold0/checkpoints/train.93.pth"


# BIRD_CODE={
#  'cjh_6794': 0,
#  'cl_4732': 1,
#  'cl_4738': 2,
#  'cmk_4613': 3,
#  'cq_7461': 4,
#  'ctj_6713': 5,
#  'cy_3658': 6,
#  'czh_6014': 7,
#  'gll_6412': 8,
#  'gyf_3014': 9,
#  'gyx_3105': 10,
#  'gzx_0348': 11,
#  'hml_3467': 12,
#  'ht_0145': 13,
#  'hxr_3014': 14,
#  'jcx_0316': 15,
#  'jyd_1031': 16,
#  'lh_1034': 17,
#  'lmx_9714': 18,
#  'lqx_9746': 19,
#  'lww_1346': 20,
#  'rjn_0346': 21,
#  'wb_4678': 22,
#  'wc_3014': 23,
#  'whl_1010': 24,
#  'wjx_7974': 25,
#  'wll_3679': 26,
#  'wlq_6714': 27,
#  'ws_0314': 28,
#  'wx_3476': 29,
#  'wyh_1973': 30,
#  'wyj_9746': 31,
#  'wyy_7741': 32,
#  'wzy_3121': 33,
#  'xx_0148': 34,
#  'ylg_1435': 35,
#  'ypp_3746': 36,
#  'yxj_6671': 37,
#  'zfj_6741': 38,
#  'zjp_7843': 39,
#  'znn_3014': 40,
#  'zq_9742': 41,
#  'zwq_4476': 42,
#  'zxd_3476': 43,
#  'zy_3167': 44,
#  'zyl_6677': 45,
#  'zym_0137': 46,
#  'zym_9745': 47,
#  'zzg_6647': 48,
#  'zzw_6479': 49}


BIRD_CODE={
    'abk_1049': 0,
    'af_8572': 1,
    'ah_4729': 2,
    'aq_3792': 3,
    'bfh_8971': 4,
    'bly_1354': 5,
    'bq_8369': 6,
    'bz_3269': 7,
    'cbe_9478': 8,
    'cej_2304': 9,
    'ctf_0172': 10,
    'czf_7398': 11,
    'da_6092': 12,
    'dgl_3846': 13,
    'dq_1348': 14,
    'dsw_3607': 15,
    'eqa_3472': 16,
    'eqg_3205': 17,
    'glp_0974': 18,
    'ha_5413': 19,
    'hcs_4783': 20,
    'hsd_0642': 21,
    'jf_4635': 22,
    'jta_0836': 23,
    'kjh_7219': 24,
    'kp_9837': 25,
    'kpl_2536': 26,
    'kr_6257': 27,
    'kyd_0153': 28,
    'lta_2934': 29,
    'pbs_0752': 30,
    'pcy_6375': 31,
    'pyh_0001': 32,
    'qf_4356': 33,
    'qs_8612': 34,
    'rdk_0724': 35,
    'rq_7342': 36,
    'rs_5932': 37,
    'sj_0269': 38,
    'sl_9052': 39,
    'tr_9178': 40,
    'ts_8039': 41,
    'wk_7215': 42,
    'wrb_1378': 43,
    'wsc_4963': 44,
    'yge_6705': 45,
    'yl_7619': 46,
    'yz_2984': 47,
    'zc_1804': 48,
    'zd_1689': 49,
    'zsq_0001': 50,
    'zy_2371': 51,
}


INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}


def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    """
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
    """
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


class TestDataset(data.Dataset):
    def __init__(self, audio_id: str, clip: np.ndarray,
                 img_size=224, melspectrogram_parameters={}):
        self.audio_id = audio_id
        self.clip = clip
        self.img_size = img_size
        self.melspectrogram_parameters = melspectrogram_parameters

    def __len__(self):
        # return len(self.df)
        return 1

    def __getitem__(self, idx: int):
        SR = 32000
        y = self.clip.astype(np.float32)
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

            melspec = librosa.feature.melspectrogram(y_batch, sr=SR, **self.melspectrogram_parameters)
            melspec = librosa.power_to_db(melspec).astype(np.float32)
            image = mono_to_color(melspec)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
            image = np.moveaxis(image, 2, 0)
            # <class 'numpy.ndarray'> (3, 224, 547)
            image = (image / 255.0).astype(np.float32)
            images.append(image)

        images = np.asarray(images)
        return images, self.audio_id


def get_model(config: dict, weights_path: str):
    model = ResNet(**config)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    return model


def prediction_for_clip(audio_id: str,
                        clip: np.ndarray,
                        model: ResNet,
                        mel_params: dict,
                        threshold=0.5):
    dataset = TestDataset(audio_id,
                          clip=clip,
                          img_size=224,
                          melspectrogram_parameters=mel_params)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    # prediction_dict = {}
    # for image, row_id, site in progress_bar(loader):
    for image, audio_id in progress_bar(loader):
        # to avoid prediction on large batch
        # torch.Size([1, 3, 3, 224, 547])
        # logger.debug(f'image size {image.size()}')
        image = image.squeeze(0)
        batch_size = 16
        whole_size = image.size(0)
        # logger.info(f'==== whole_size {whole_size} ====')
        if whole_size % batch_size == 0:
            n_iter = whole_size // batch_size
        else:
            n_iter = whole_size // batch_size + 1

        proba_total = np.zeros(N_CLASS)
        for batch_i in range(n_iter):
            batch = image[batch_i * batch_size:(batch_i + 1) * batch_size]
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)

            batch = batch.to(device)
            with torch.no_grad():
                prediction = model(batch)
                proba = prediction["multilabel_proba"].detach().cpu().numpy()
                # logger.debug(f'proba {proba.size} {proba}')
                proba_sum = proba.sum(axis=0)
                # logger.debug(f'proba_sum {proba_sum.shape} {proba_sum}')
            proba_total += proba_sum
            # logger.debug(f'proba_total {proba_total.shape} {proba_total}')

        proba_average = proba_total/whole_size
        # logger.debug(f'proba_average {proba_average.shape} {proba_average}')
        index = np.argmax(proba_average)
        max = proba_average[index]
        if max <= threshold:
            prediction_dict = {'AudioID': audio_id, 'IsFaked': '1', 'PersonID': '0'}
            proba_average = np.insert(proba_average, 0, 1)
        else:
            person_id = INV_BIRD_CODE[index]
            prediction_dict = {'AudioID': audio_id, 'IsFaked': '0', 'PersonID': person_id}
            proba_average = np.insert(proba_average, 0, 0)

        prediction_pd = pd.DataFrame(prediction_dict)

    return prediction_pd, (audio_id, proba_average)


def prediction(test_audio_dir: Path,
               model_config: dict,
               mel_params: dict,
               weights_path: str,
               threshold=0.5):
    model = get_model(model_config, weights_path)
    unique_audio_id = os.listdir(test_audio_dir)
    unique_audio_id.sort()

    warnings.filterwarnings("ignore")
    prediction_dfs = []
    audio_ids = []
    proba_averages = []
    for audio_id in unique_audio_id:
        with timer(f"Loading {audio_id}", logger):
            # clip, _ = librosa.load(test_audio / (audio_id + ".mp3"),
            # <class 'numpy.ndarray'> 484454, 32000
            clip, _ = librosa.load(test_audio_dir / audio_id,
                                   sr=TARGET_SR,
                                   mono=True,
                                   res_type="kaiser_fast")

        # test_df_for_audio_id = test_df.query(
        #     f"audio_id == '{audio_id}'").reset_index(drop=True)
        with timer(f"Prediction on {audio_id}", logger):
            prediction_df, prob = prediction_for_clip(audio_id,
                                                  clip=clip,
                                                  model=model,
                                                  mel_params=mel_params,
                                                  threshold=threshold)

            audio_id, proba_average = prob

        prediction_dfs.append(prediction_df)
        audio_ids.append(audio_id)
        proba_averages.append(proba_average)

    prediction = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
    audio_ids_np = np.asarray(audio_ids)
    proba_averages_np = np.asarray(proba_averages)

    return prediction, audio_ids_np, proba_averages_np


os.environ['CUDA_VISIBLE_DEVICES']='2'

threshold = 0.30
logger.info(f'weights_path {weights_path}')
logger.info(f'threshold {threshold}')
prediction, audio_ids_np, proba_averages_np = prediction(test_audio_dir=test_audio_dir,
                        model_config=model_config,
                        mel_params=melspectrogram_parameters,
                        weights_path=weights_path,
                        threshold=threshold)
                        # threshold=0.8)

time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
prediction.to_csv(f'output/result_{time_str}.txt', sep=' ', index=None, header=None)
np.save(f'output/audio_id_{time_str}.npy', audio_ids_np)
np.save(f'output/proba_{time_str}.npy', proba_averages_np)