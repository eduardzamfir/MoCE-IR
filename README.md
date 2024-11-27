# MoCE-IR



### Complexity Experts are Task-Discriminative Learners for Any Image Restoration

#### [Eduard Zamfir<sup>1</sup>](https://eduardzamfir.github.io), [Zongwei Wu<sup>1</sup>](https://sites.google.com/view/zwwu/accueil), [Nancy Mehta<sup>1</sup>](https://scholar.google.com/citations?user=WwdYdlUAAAAJ&hl=en&oi=ao), [Yuedong Tan<sup>1</sup>](https://scholar.google.com/citations?user=p7fOQkUAAAAJ&hl=en), [Danda Pani Paudel<sup>2,3</sup>](https://people.ee.ethz.ch/~paudeld/),  [Yulun Zhang<sup>4</sup>](http://yulunzhang.com/) and [Radu Timofte<sup>1</sup>](https://www.informatik.uni-wuerzburg.de/computervision/)

#### **<sup>1</sup> University of Würzburg, Germany - <sup>2</sup> INSAIT Sofia University, Bulgaria - <sup>3</sup> ETH Zürich, Switzerland - <sup>4</sup> Shanghai Jiao Tong University, China**

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2405.15475)
[![project](https://img.shields.io/badge/project-page-brightgreen)](https://eduardzamfir.github.io/daair/)


## Latest
- `11/27/2024`: Repository is created.

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


## Citation

If you find our work helpful, please consider citing the following paper and/or ⭐ the repo.
```
@misc{zamfir2024complexity,
      title={Complexity Experts are Task-Discriminative Learners for Any Image Restoration}, 
      author={Eduard Zamfir and Zongwei Wu and Nancy Mehta and Yuedong Tan and Danda Pani Paudel and Yulun Zhang and Radu Timofte},
      year={2024},
      eprint={2405.15475},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

This code is built on [PromptIR](https://github.com/va1shn9v/PromptIR) and [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet).
