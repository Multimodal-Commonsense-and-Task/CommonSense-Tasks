# DBMSolver: A Training-free Diffusion Bridge Sampler for High-Quality Image-to-Image Translation (CVPR 2026)

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2605.05889-B31B1B.svg)](https://arxiv.org/abs/2605.05889)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://openaccess.thecvf.com/content/CVPR2026/html/Venugopal_DBMSolver_A_Training-free_Diffusion_Bridge_Sampler_for_High-Quality_Image-to-Image_Translation_CVPR_2026_paper.html)

> **[DBMSolver: A Training-free Diffusion Bridge Sampler for High-Quality Image-to-Image Translation](https://openaccess.thecvf.com/content/CVPR2026/html/Venugopal_DBMSolver_A_Training-free_Diffusion_Bridge_Sampler_for_High-Quality_Image-to-Image_Translation_CVPR_2026_paper.html)**  
> [Sankarshana Venugopal](mailto:sankarshana.v@gmail.com), [Mohammad Mostafavi](mailto:mostafavi.isfahani@gmail.com), [Jonghyun Choi](mailto:jonghyunchoi@snu.ac.kr)  
> Seoul National University  


## Abstract
Diffusion-based image-to-image (I2I) translation excels in high-fidelity generation but suffers from slow sampling in state-of-the-art Diffusion Bridge Models (DBMs), often requiring dozens of function evaluations (NFEs). We introduce **DBMSolver**, a training-free sampler that exploits the semi-linear structure of DBM's underlying SDE and ODE via exponential integrators, yielding highly-efficient 1st- and 2nd-order solutions. This reduces NFEs by up to 5x while boosting quality (e.g., FID drops 53% on DIODE at 20 NFEs vs. 2nd-order baseline). Experiments on inpainting, stylization, and semantics-to-image tasks across resolutions up to 256x256 show DBMSolver sets new SOTA efficiency-quality tradeoffs, enabling real-world applicability.

## Instructions: Sampling with DBMSolver

This repo provides instructions on running the inference code for our training-free sampler, DBMSolver, on the [DIODE](https://diode-dataset.org/) (256x256) dataset.

The code is based on [DDBM](https://arxiv.org/abs/2309.16948) and [DBIM](https://arxiv.org/abs/2405.15885v6), the current state-of-the-art training-free samplers for Diffusion Bridge Models (DBMs).

### Download the Repo

Download the git repo to your preferred destination.

### Building Docker Image
On the terminal, enter the folder where you saved the Git repo via the 'cd' command. The following command will build the Docker image with the required packages.

NOTE: Make sure that you are "inside" the repository folder, ie., the folder should contain the Dockerfile.

```bash
cd /Path/to/GitRepository/Containing/the/DBMSolver/Dockerfile
docker build -t dbmsolver_image .
```

This creates a docker image named `dbmsolver_image`, which we will use to run our inference (and/or training) experiments.

### Creating the Docker Container
```bash
docker run -v /Path/To/Git/Repo:/root/code/ \
    -itd --gpus=all --ipc=host --name=dbmsolver_container \
    dbmsolver_image
```
This creates a docker container named "dbmsolver_container".

### Executing the DBMSolver Container
Execute the container using the following command:
```bash
docker exec -it dbmsolver_container bash
```

### Downloading the DIODE Dataset

Important: Please create `/root/data` folder by running the following code:
```bash
mkdir /root/data
```
This is the directory in which all the data will be saved to!

DBIM provides automatic downloading scripts, which can be used like this:
```
cd /root/data/
bash /root/code/assets/datasets/download_extract_DIODE.sh
```

After downloading, the DIODE dataset requires preprocessing by running `python preprocess_depth.py`.

### Pre-trained DBM Checkpoints for Evaluation

Please put the downloaded checkpoints under `assets/ckpts/`.

We directly utilize the pretrained checkpoints from [DDBM](https://github.com/alexzhou907/DDBM):

- DIODE: [diode_ema_0.9999_440000.pt](https://huggingface.co/alexzhou907/DDBM/resolve/main/diode_ema_0.9999_440000.pt)

We remove the dependency on external packages such as `flash_attn` in this codebase, which is already supported natively by PyTorch. After downloading the two checkpoints above, please run `python preprocess_ckpt.py` to complete the conversion.

### Sampling DIODE images

```
bash scripts/sample.sh $DATASET_NAME $NFE $SAMPLER ($AUX)
```

- `$DATASET_NAME` is `diode`.
- `$NFE` is the *Number of Function Evaluations*, which is proportional to the sampling time.
- `$SAMPLER` can be chosen from `ground_truth`/`heun`/`dbim`/`dbim_high_order`/`dbmsolver`. 
  - `heun` is the vanilla sampler of DDBM, which simulates the SDE/ODE step alternatively. In this case, `$AUX` is not required.
  - `dbim` is DBIM's proposed 1st-order sampler. When using `dbim`, `$AUX` corresponds to $\eta$ which controls the stochasticity level (floating-point value in $[0,1]$). 
  - `dbim_high_order` is DBIM's proposed higher-order (2nd and 3rd) samplers. In this case, `$AUX` corresponds to the order (2 or 3).
  - `dbmsolver` is our proposed SOTA 2nd-order sampler. In this case, `$AUX` corresponds to the order of the Bridge SDE solution. In our paper, we empirically show that using `$AUX` = 1 is enough, as `$AUX` = 2 requires one extra NFE but only provides negligible gains (ie, higher computational cost with little gain).
  - `ground_truth` just returns the ground truth images. This can be used for generating reference statistics (explained below). In this case, `$AUX` is not required.

The samples will be saved to `workdir/`.

### Evaluations

Before evaluating the image translation results, generate (or download) the ground truth reference statistics for DIODE and place it `assets/stats/`:
- Recommended: The true reference statistics can be generated by running ```bash scripts/sample.sh diode 1 ground_truth```. Please make sure that the generated '.npz' file is stored in the `assets/stats/` folder!
- Not Recommended: It can be downloaded from the following link: [diode_ref_256_data.npz](https://huggingface.co/alexzhou907/DDBM/resolve/main/diode_ref_256_data.npz). However, we found that the LPIPS and MSE metrics obtained by evaluating with the downloaded statistics are faulty. The FID and IS metrics are fine, though.

The evaluation can automatically proceed by specifying the same dataset and sampler arguments as sampling:

```
bash scripts/evaluate.sh $DATASET_NAME $NFE $SAMPLER ($AUX)
```

## Citation
If you find our work useful for your research, please consider citing our paper:

```
@InProceedings{venugopal2026dbmsolver,
    author    = {Venugopal, Sankarshana and Mostafavi, Mohammad and Choi, Jonghyun},
    title     = {DBMSolver: A Training-free Diffusion Bridge Sampler for High-Quality Image-to-Image Translation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {36062-36071},
    url       = {https://arxiv.org/abs/2605.05889}
}
```

## Acknowledgements

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
