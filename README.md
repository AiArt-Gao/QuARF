# QuARF

[AAAI-25] Quality-Adaptive Receptive Fields for Degraded Image Perception (QuARF)

[![Project](https://img.shields.io/badge/Project-Page-green.svg)](https://github.com/AiArt-Gao/QuARF/)

## Abstract
<p align="justify">
  Advanced Deep Neural Networks (DNNs) perform well for high-quality images, but their performance dramatically decreases for degraded images. Data augmentation is commonly used to alleviate this problem, while using too many perturbed data might seriously decrease the performance for pristine images. To tackle this challenge, we take our cue from the spatial coincidence assumptionabout human visual perception, i.e. multiscale and varying receptive fields are required for understanding pristine and degraded images.Correspondingly, we propose a novel plug-and-play network architecture, dubbed Quality-Adaptive Receptive Fields (QuARF), to automatically select the optimal receptive fields based on the quality of the input image. 
To this end, we first design a multi-kernel convolutional block, which comprises multiscale continuous receptive fields.Afterward, we design a quality-adaptive routing network to predict the significance of each kernel, based on the quality features extracted from the input image. In this way, QuARF automatically selects the optimal inference route for each image. To further boost the efficiency and effectiveness, the input feature map is split into multiple groups, with each group independently learning its quality-adaptive routing parameters.We apply QuARF to a variety of DNNs, and conduct experiments in both discriminative and generation tasks, including semantic segmentation, image translation and restoration. 
Thorough experimental results show that QuARF significantly and robustly improves the performance for degraded images, and outperforms data augmentation in most cases.
</p>

## Pipeline

<p align="center"><img src="assets/QuARF.jpg" width="100%"/></p>


## Sample Results

<p align="center"><img src="assets/res_i2i.jpg" width="100%"/></p>

<p align="center"><img src="assets/res_SAM_AGAN.png" width="100%"/></p>



