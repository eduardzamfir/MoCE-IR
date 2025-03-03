# MoCE-IR - CVPR 2025



### Complexity Experts are Task-Discriminative Learners for Any Image Restoration

#### [Eduard Zamfir<sup>1</sup>](https://eduardzamfir.github.io), [Zongwei Wu<sup>1</sup>](https://sites.google.com/view/zwwu/accueil), [Nancy Mehta<sup>1</sup>](https://scholar.google.com/citations?user=WwdYdlUAAAAJ&hl=en&oi=ao), [Yuedong Tan<sup>1</sup>](https://scholar.google.com/citations?user=p7fOQkUAAAAJ&hl=en), 
#### [Danda Pani Paudel<sup>2</sup>](https://insait.ai/dr-danda-paudel/),  [Yulun Zhang<sup>3</sup>](http://yulunzhang.com/) and [Radu Timofte<sup>1</sup>](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en&oi=sra)

#### **<sup>1</sup> University of Würzburg, Germany - <sup>2</sup> INSAIT Sofia University, Bulgaria - <sup>3</sup> Shanghai Jiao Tong University, China**

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2411.18466)
[![project](https://img.shields.io/badge/project-page-brightgreen)](https://eduardzamfir.github.io/moceir)


## Latest
- `02/27/2025`: Our work got accepted at CVPR 2025! Stay tuned for full code & model release. 🎉🥳
- `11/28/2024`: Repository is created.

## Method
<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Recent advancements in all-in-one image restoration models have revolutionized the ability to address diverse degradations through a unified framework. However, parameters tied to specific tasks often remain inactive for other tasks, making mixture-of-experts (MoE) architectures a natural extension. Despite this, MoEs often show inconsistent behavior, with some experts unexpectedly generalizing across tasks while others struggle within their intended scope. This hinders leveraging MoEs' computational benefits by bypassing irrelevant experts during inference.
We attribute this undesired behavior to the uniform and rigid architecture of traditional MoEs. To address this, we introduce ``complexity experts" -- flexible expert blocks with varying computational complexity and receptive fields. A key challenge is assigning tasks to each expert, as degradation complexity is unknown in advance. Thus, we execute tasks with a simple bias toward lower complexity.
To our surprise, this preference effectively drives task-specific allocation, assigning tasks to experts with the appropriate complexity. 
Extensive experiments validate our approach, demonstrating the ability to bypass irrelevant experts during inference while maintaining superior performance. The proposed MoCE-IR model outperforms state-of-the-art methods, affirming its efficiency and practical applicability.
</details>

![](assets/method.png)

## Results
<br>
<details>
  <summary>
  <font size="+1">All-in-One Restoration: Haze, Rain, Noise</font>
  </summary>
  <p align="center">
  <img src = "assets/aio_3_table.png">
  </p>
</details>

<br>
<details>
  <summary>
  <font size="+1">All-in-One Restoration: Haze, Rain, Noise, Blur, Low Light</font>
  </summary>
  <p align="center">
  <img src = "assets/aio_5_table.png">
  </p>
</details>


## Install
Download this repository
````
git clone https://github.com/eduardzamfir/MoCE-IR.git
cd MoCE-IR
````
Create a conda enviroment:
````
ENV_NAME="moceir"
conda create -n $ENV_NAME python=3.10
conda activate $ENV_NAME
````
Run following script to install the dependencies:
````
bash install.sh
````

## Usage
Pre-trained checkpoints and visual results can be downloaded [here](https://drive.google.com/drive/folders/1pQBceb8cCPdIzbqbNNGqV5qNXzzqL4uK?usp=share_link). Place the checkpoints in `checkpoints/`.

In `options` you can find the corresponding config files for reproducing our experiments.

### **Evaluation**
For testing the pre-trained checkpoints please use following commands. Replace `[MODEL]_` with desired model configuration. Argument `--benchmarks` accepts also a list of `str` and will iterate over defined testsets.

#### All-in-One: 3 Degradations
**1. Rain100L**
`````
python src/test.py --benchmarks derain --checkpoint_id [MODEL]_AIO3 --de_type denoise_15 denoise_25 denoise_50 dehaze derain
`````
**2. SOTS**
`````
python src/test.py --benchmarks dehaze --checkpoint_id [MODEL]_AIO3 --de_type denoise_15 denoise_25 denoise_50 dehaze derain
`````
**3. CBSD68**
`````
python src/test.py --benchmarks denoise_15 denoise_25 denoise_50 --checkpoint_id [MODEL]_AIO3 --de_type denoise_15 denoise_25 denoise_50 dehaze derain
`````

#### All-in-One: 5 Degradations
**1. Rain100L**
`````
python src/test.py --benchmarks derain --checkpoint_id [MODEL]_AIO3 --de_type denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie
`````
**2. SOTS**
`````
python src/test.py --benchmarks dehaze --checkpoint_id [MODEL]_AIO3 --de_type denoise_15 denoise_25 denoise_50 dehaze derain
`````
**3. CBSD68**
`````
python src/test.py --benchmarks denoise_25 --checkpoint_id [MODEL]_AIO3 --de_type denoise_15 denoise_25 denoise_50 dehaze derain
`````
**4. GoPro**
`````
python src/test.py --benchmarks gopro --checkpoint_id [MODEL]_AIO3 --de_type denoise_15 denoise_25 denoise_50 dehaze derain
`````
**5. LoLv1**
`````
python src/test.py --benchmarks lolv1 --checkpoint_id [MODEL]_AIO3 --de_type denoise_15 denoise_25 denoise_50 dehaze derain
`````

### **Training**

Use following commands to train the *lightweight* `MoCE-IR-S` or *heavy* `MoCE-IR` version either on three or five degradations. You can specify with `--gpus` whether you want to train on a single (`1`) or multiple gpus (`>1`). However, `--batch_size` defines the batch size per gpu. We trained our networks on 4x NVIDIA 4090 cards.
 
**1. All-in-One: 3 Degradations**
`````
python src/train.py --model [MoCE_IR_S/MoCE_IR] --batch_size 8 --de_type derain dehaze denoise_15 denoise_25 denoise_50 --num_gpus 4 --loss_type FFT --balance_loss_weight 0.01
`````

**2. All-in-One: 5 Degradations**
`````
python src/train.py --model [MoCE_IR_S/MoCE_IR] --batch_size 8 --de_type derain dehaze denoise_15 denoise_25 denoise_50 deblur synllie --trainset standard --num_gpus 4 --loss_type FFT --balance_loss_weight 0.01
`````

**3. CDD11: Composited Degradations**

You can also train our models from scratch on the composited degradations dataset [CDD11](https://github.com/gy65896/OneRestore) with four different setups running following commands:

+ `--trainset CDD_single`: Low light (L), Haze (H), Rain (R) and Snow (S)
+ `--trainset CDD_double`: L+H, L+R, L+S, H+R, H+S
+ `--trainset CDD_triple`: L+H+R, L+H+S
+ `--trainset CDD_all`: CDD_single + CDD_double + CDD_triple at the same time

`````
python src/train.py --model [MoCE_IR_S/MoCE_IR] --batch_size 8 --de_type derain dehaze denoise_15 denoise_25 denoise_50 deblur synllie --trainset [CDD11_single/CDD11_double/CDD11_triple/CDD11_all] --num_gpus 4 --loss_type FFT --balance_loss_weight 0.01
`````


## Citation

If you find our work helpful, please consider citing the following paper and/or ⭐ the repo.
```
@misc{zamfir2024complexityexperts,
      title={Complexity Experts are Task-Discriminative Learners for Any Image Restoration}, 
      author={Eduard Zamfir and Zongwei Wu and Nancy Mehta and Yuedong Tan and Danda Pani Paudel and Yulun Zhang and Radu Timofte},
      year={2024},
      eprint={2411.18466},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```

## Acknowledgements

This code is built on [PromptIR](https://github.com/va1shn9v/PromptIR) and [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet).
