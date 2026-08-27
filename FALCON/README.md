# FALCON: Fast Adaptive Lightweight Computation of Intensities and Events for Depth Estimation (CVPR Findings 2026)

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://openaccess.thecvf.com/content/CVPR2026F/html/Venugopal_FALCON_Fast_Adaptive_Lightweight_Computation_of_Intensities_and_Events_for_CVPRF_2026_paper.html)

> **[FALCON: Fast Adaptive Lightweight Computation of Intensities and Events for Depth Estimation](https://openaccess.thecvf.com/content/CVPR2026F/html/Venugopal_FALCON_Fast_Adaptive_Lightweight_Computation_of_Intensities_and_Events_for_CVPRF_2026_paper.html)**  
> [Sankarshana Venugopal](mailto:sankarshana.v@gmail.com), [Mohammad Mostafavi](mailto:mostafavi.isfahani@gmail.com), [Jonghyun Choi](mailto:jonghyunchoi@snu.ac.kr)  
> Seoul National University  

[![FALCON](https://github.com/snumprlab/falcon/blob/master/images/teaser.jpg)](https://youtu.be/iguyu-d_YZQ)

## Abstract
Event-based stereo depth estimation promises high-speed, robust perception under challenging conditions, but fusing events with intensity frames incurs heavy computational costs, hindering real-time deployment. We introduce FALCON: a Fast Adaptive Lightweight Computation framework that co-designs spiking neuron dynamics (via LIF neurons) with 1x1 convolutional event representations and a streamlined stereo backbone for efficient depth estimation. Our per-pixel spiking encoders capture temporal dynamics with over 1000x fewer FLOPs than prior modules, while our sigma-matching heuristic optimizes event grouping to minimize blur. FALCON achieves state-of-the-art accuracy on DSEC (MAE 0.356) and MVSEC (MDE 11.7 cm) at 32.2,FPS and 66.9,FPS, respectively, establishing a new real-time benchmark for neuromorphic vision applications.

## Instructions for Testing our FALCON Model:
This repo provides the inference code for our FALCON model.

The code is based on [SE-CFF](https://github.com/yonseivnl/se-cff), the SoTA work at DSEC benchmark. Thanks to this work, we were able to improve our work, more effectively and efficiently.


## Downloading the MVSEC Dataset
Please download the MVSEC dataset from the HuggingFace link here [link](https://huggingface.co/datasets/limsanky/falcon_mvsec/resolve/main/indoor_flying_mvsec.tar.gz).

You can use wget to download and extract the MVSEC dataset. You can download it via the following command:
```bash
wget <dataset link above>
```


After downloading the “indoor_flying_mvsec.tar.gz” file into your preferred directory, extract it using “tar -xvf indoor_flying_mvsec.tar.gz”.

This will create a folder named “indoor_flying2” that contains the “indoor_flying” subset of the MVSEC dataset.

## Download the Repo

Download the git repo to your preferred destination.

## Building Docker Image
On the terminal, enter the folder where you saved the Git repo via the 'cd' command.

NOTE: Make sure that this is the FALCON repository folder, ie., it should contain the Dockerfile.

```bash
cd /Path/to/GitRepository/Containing/the/FALCON/Dockerfile
docker build -t falcon_image .
```

This creates a docker image named "falcon_image".

## Create Docker Container for FALCON
```bash
docker run \
    -v /Path/to/MVSEC/Folder/:/root/data/ \
    -v /Path/To/Git/Repo:/root/code/ \
    -itd --gpus=all --ipc=host --name=falcon_container \
    falcon_image
```
This creates a docker container named "falcon_container".

## Execute the FALCON Container
Execute the container using the following command:
```bash
docker exec -it falcon_container bash
```

## Build Deformable Convolution
```bash
cd /root/code/scripts/models/deform_conv/ && bash build.sh
```
This builds the deformable convolution layers.

## Run inference with the FALCON model
```bash
cd /root/code/scripts/launch && . inference_cvpr26.sh
```

## Citation
If you find our work useful for your research, please consider citing our paper:

```
@InProceedings{Venugopal_2026_CVPR,
    author    = {Venugopal, Sankarshana and Mostafavi, Mohammad and Choi, Jonghyun},
    title     = {FALCON: Fast Adaptive Lightweight Computation of Intensities and Events for Depth Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {5024-5033}
}
```

## Acknowledgements

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
