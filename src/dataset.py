import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os
import torch.utils.data as data

from pathlib import Path

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

# BIRD_CODE = {
#     'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,
#     'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,
#     'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,
#     'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,
#     'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,
#     'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,
#     'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,
#     'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,
#     'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,
#     'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,
#     'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,
#     'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,
#     'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,
#     'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,
#     'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,
#     'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,
#     'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,
#     'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,
#     'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,
#     'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,
#     'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,
#     'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,
#     'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,
#     'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,
#     'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,
#     'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,
#     'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,
#     'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,
#     'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,
#     'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,
#     'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,
#     'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,
#     'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,
#     'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,
#     'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,
#     'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,
#     'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,
#     'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,
#     'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,
#     'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,
#     'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,
#     'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,
#     'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,
#     'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,
#     'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,
#     'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,
#     'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,
#     'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,
#     'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,
#     'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,
#     'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,
#     'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,
#     'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263
# }

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

PERIOD = 5


class NpyDataset(data.Dataset):
    def __init__(self, root):
        # self.root = root
        self.root = 'input/person/train_b_npy'
        self.data = []
        data_root = os.path.expanduser(self.root)
        persons = os.listdir(data_root)
        for person in sorted(persons):
            person_dir = os.path.join(data_root, person)
            fnames = os.listdir(person_dir)
            for fname in sorted(fnames):
                path = os.path.join(person_dir, fname)
                self.data.append((path, person))

    def __len__(self):
        # a: 37944; b: 38592
        return len(self.data)

    def __getitem__(self, idx: int):
        image = np.load(self.data[idx][0])
        labels = np.zeros(len(BIRD_CODE), dtype=int)
        person = self.data[idx][1]
        if person == '0':
            pass
            # print('-------- fake --------')
        else:
            # print('-------- real --------')
            labels[BIRD_CODE[person]] = 1

        return {
            "image": image,
            "targets": labels
        }


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
