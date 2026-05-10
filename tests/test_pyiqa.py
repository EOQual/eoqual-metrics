"""

Install
-------
pip install "setuptools<70 # Erreur sinon : ImportError: cannot import name 'packaging' from 'pkg_resources'
pip install pyiqa

Modify torch.serialization.py (ligne 865)
def load(
    f: FILE_LIKE,
    map_location: MAP_LOCATION = 'cpu',

"""
import warnings
warnings.filterwarnings("ignore")

#FIXME: torch avant numpy, sinon segmentation fault
#       Pourquoi ?
#       Est-ce uniquement sur macos ?

import torch

import numpy as np
from skimage import data
import pyiqa
from torchvision import transforms

from zMetrics.utils import prepare_image


def add_gaussian_noise(image, mean=0, sigma=20):
    """
    Add Gaussian noise to an image of type np.uint8.
    """
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    gaussian_noise = gaussian_noise.reshape(image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


# if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    torch.device('mps')
# else CPU
else:
    torch.device('cpu')

# On force cpu car les modèles sont compatibles cuda et non mps
device = torch.device("cpu")
print(f"Device: {device}")


# Load image
Image = data.camera() # type: ignore
print(f"Image shape: {Image.shape}")
print(f"Image type : {Image.dtype}")

# Add Noise
Image_noisy = add_gaussian_noise(Image)

# Numpy to Torch
ref  = prepare_image(Image).to(device)
dist = prepare_image(Image_noisy).to(device)
# ref = transforms.ToTensor()(Image)
# dist = transforms.ToTensor()(Image_noisy)


# list all available metrics
# print(pyiqa.list_models())

FR_METRIC = ['ahiq', 'cw_ssim', 'mad', 'ms_ssim', 'psnr', 'psnry', 'ssim', 'ssimc', 'vsi']
FR_METRIC3 = ['fsim', 'gmsd', 'nlpd', 'vif']
FR_METRIC_MODEL = ['ckdn', 'dists', 'lpips', 'lpips-vgg', 'pieapp', 'stlpips', 'stlpips-vgg', 'topiq_fr', 'topiq_fr-pipal', 'wadiqam_fr']

NR_METRIC = ['brisque', 'clipiqa', 'clipiqa+', 'clipscore', 'entropy', 'niqe', 'nrqm', 'pi', 'qalign']
NR_METRIC3 = ['ilniqe', ]
NR_METRIC_MODEL = ['clipiqa+_rn50_512', 'clipiqa+_vitL14_512', 'cnniqa', 'dbcnn', 'fid', 'hyperiqa', 'inception_score', 'laion_aes', 'liqe', 'liqe_mix', 
                   'maniqa', 'maniqa-kadid', 'maniqa-koniq', 'maniqa-pipal', 'musiq', 'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 
                   'nima', 'nima-koniq', 'nima-spaq', 'nima-vgg16-ava', 'paq2piq', 'qalign', 'topiq_iaa', 'topiq_iaa_res50', 'topiq_nr', 'topiq_nr-face', 
                   'topiq_nr-flive', 'topiq_nr-spaq', 'tres', 'tres-flive', 'tres-koniq', 'unique', 'uranker', 'wadiqam_nr']

EXCLUDE_METRIC = ['clipscore', 'cnniqa', 'fid', 'inception_score', 'laion_aes', 'musiq', 'musiq-ava', 'musiq-koniq', 
                  'musiq-paq2piq', 'musiq-spaq', 'paq2piq', 'pieapp', 'qalign', 'topiq_nr-face', 'uranker', 'wadiqam_fr', 'wadiqam_nr']
#EXCLUDE_METRIC = []

for metric in pyiqa.list_models():

    if metric in EXCLUDE_METRIC:

        print(f"> {metric}")

        # if metric not in EXCLUDE_METRICS:
        try:

            # create metric with default setting
            iqa_metric = pyiqa.create_metric(metric, device=device)

    #         # Full Reference (FR) ou None Reference (NR)
    #         if iqa_metric.metric_mode == 'FR':
    #             FR_METRIC.append(metric)
    #         elif iqa_metric.metric_mode == 'NR':
    #             NR_METRIC.append(metric)
    #         else:
    #             BAD.append(metric)

            # Compute metric
            score_fr = iqa_metric(ref, dist)
            
            print(f"{metric} - {iqa_metric.metric_mode}: {float(score_fr)} ({'lower' if iqa_metric.lower_better else 'higher'})")

        except Exception as e:
            print(e)
            EXCLUDE_METRIC.append(metric)

print('FR_METRIC)')
print(FR_METRIC)
print('NR_METRIC)')
print(NR_METRIC)
print('EXCLUDE_METRIC)')
print(EXCLUDE_METRIC)