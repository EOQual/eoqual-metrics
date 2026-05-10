# Wrapper for metric method

## Requirements
- FFTW [https://www.fftw.org/] >= 3.3
    + brew install fftw
    + conda install fftw
    + pixi add fftw
- libsvm
    + conda install libsvm
    + pixi add fftw

## Installation
pip install git+https://git.oamcloud.uk/olivier/zMetrics.git \
or \
pip install --index-url https://git.oamcloud.uk/api/packages/olivier/pypi/simple --extra-index-url https://pypi.org/simple zMetrics

And after, \
pip install image-similarity-measures --no-deps \
python postinstallation

## Full-Reference IQA (Image Quality Assessment)
| Métrics                                                    |||||
|------------|------------|------------|------------|------------|
| cw_ssim    | dists      | ergas      | fsim       | gmsd       |
| issm       | lpips_vgg  | mad        | mae        | mdae       |
| mse        | msssim     | ncc        | ndp        | nldp       |
| nmi        | nrmse      | pamse      | psnr       | psnrb      |
| rase       | reco       | rmse       | rmse_sw    | sam        |
| scc        | snr        | sre        | ssim       | uqi        |
| vif        | vsi        | wsnr       |            |            |

Metrics based on structural similarity:
- cw_ssim (Complex Wavelet Structural Similarity Index): An extension of SSIM using complex wavelets.
- fsim (Feature Similarity Index): Measures similarity based on phase congruency and gradient magnitude.
- issm (Information content weighted Structural Similarity): Modified SSIM with information content weighting.
- msssim (Multi-Scale Structural Similarity Index): An extension of SSIM that considers multiple scales.
- ssim (Structural Similarity Index): Measures perceptual similarity based on luminance, contrast, and structure.
- vsi (Visual Saliency-based Index): Measures image quality based on visual saliency.

Metrics based on error or difference:
- dists (Deep Image Structure and Texture Similarity): Uses deep learning features to assess image quality.
- ergas (Erreur Relative Globale Adimensionnelle de Synthèse): Measures the overall error of a processed image.
- lpips_vgg (Learned Perceptual Image Patch Similarity): Uses a pre-trained VGG network to measure perceptual similarity.
- mad (Maximum Absolute Difference): The largest absolute difference between corresponding pixels.
- mae (Mean Absolute Error): The average absolute difference between corresponding pixels.
- mdae (Median Absolute Error): The median absolute difference between corresponding pixels.
- mse (Mean Squared Error): The average squared difference between corresponding pixels.
- ndp (Noise Distortion Power): Measures the power of noise and distortion.
- nldp (Normalized Noise Distortion Power): Normalized version of NDP.
- nrmse (Normalized Root Mean Squared Error): RMSE normalized by the range of pixel values.
- pamse (Penalized Average Mean Squared Error): A variation of MSE with penalties.
- rase (Relative Average Spectral Error): Measures the relative error in spectral bands.
- rmse (Root Mean Squared Error): The square root of MSE.
- rmse_sw (Root Mean Squared Error with Sliding Window): RMSE calculated with a sliding window.

Metrics based on signal-to-noise ratio:
- psnr (Peak Signal-to-Noise Ratio): Measures the ratio between the maximum possible power of a signal and the power of corrupting - noise.
- psnrb (Peak Signal-to-Noise Ratio Block): PSNR calculated on image blocks.
- snr (Signal-to-Noise Ratio): Measures the ratio of signal power to noise power.
- wsnr (Weighted Signal-to-Noise Ratio): SNR with weighted components.

Other metrics:
- ncc (Normalized Cross Correlation): Measures the similarity between two images based on their correlation.
- nmi (Normalized Mutual Information): Measures the statistical dependence between two images.
- reco (Relative global dimensional error): Relative global dimensional error.
- sam (Spectral Angle Mapper): Measures the spectral similarity between two pixels.
- scc (Spatial Correlation Coefficient): Measures the spatial correlation between two images.
- sre (Signal to Reconstruction Error): Signal to Reconstruction Error.
- uqi (Universal Quality Index): Combines luminance, contrast, and correlation comparisons.
- vif (Visual Information Fidelity): Measures the information fidelity between a reference and distorted image.

Not yet : bsn, isnr \
Ne fonctione pas : nqm

## No-Reference IQA or Blind IQA (Blind Image Quality Assessment - BIQA)
- brisque (Blind/Referenceless Image Spatial Quality Evaluator)
- entropy
- niqe (Natural Image Quality Evaluator)
- piqe (Perception based Image Quality Evaluator)
- sharpness

## Reduced-Reference IQA
None

## Credits
- cpbd [https://pypi.org/project/cpbd/]
- image-similarity-measures [https://pypi.org/project/image-similarity-measures/]
- IQA-optimization [https://github.com/dingkeyan93/IQA-optimization]
- IQA_pytorch [https://pypi.org/project/IQA-pytorch/]
- metrikz [https://gitlab.com/gpds-unb/pymetrikz]
- NumPy [https://numpy.org/]
- OpenCV [https://opencv.org/]
- pyrtools [https://pypi.org/project/pyrtools/]
- Scikit-Image [https://scikit-image.org/]
- Scikit-Learn [https://scikit-learn.org/stable/]
- SciPy [https://scipy.org/]
- sewar [https://sewar.readthedocs.io/en/latest/]
- fmeasure [https://fr.mathworks.com/matlabcentral/fileexchange/27314-focus-measure]

- BRISQUE
    - [https://github.com/ocampor/image-quality]
    - [https://github.com/EadCat/NIQA/tree/master/brisque]
    - [https://github.com/buyizhiyou/NRVQA]
    - [https://github.com/rehanguha/brisque]
- CW_SSIM
    - [https://github.com/charparr/tundra-snow/blob/master/continuous_patterns.py]
- MSSSIM
    - [https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py]
- NIQE
    - [https://programtalk.com/vs2/?source=python%2F341%2Fvideo-quality#]
    - [https://github.com/EadCat/NIQA/tree/master/niqe]
- PIQE
    - [https://github.com/EadCat/NIQA/tree/master/piqe]
- RECO
    - [https://programtalk.com/vs2/?source=python%2F341%2Fvideo-quality#]
- SSIM
    - [https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py]
    - [https://github.com/aizvorski/video-quality]
    - [http://isit.u-clermont1.fr/~anvacava/code-ssim-source.html]
- VIF
    - [https://programtalk.com/vs2/?source=python%2F341%2Fvideo-quality#]
    - [https://github.com/abhinaukumar/vif/blob/main/vif_utils.py]