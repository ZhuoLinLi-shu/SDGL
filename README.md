# SDGL- Official PyTorch Implementation
The official implementation of Dynamic graph structure learning for multivariate time series forecasting ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323001243)).  

The initial version was Dynamic Graph Learning-Neural Network for Multivariate Time Series Modeling ([arxiv](https://arxiv.org/abs/2112.03273)).

![](https://github.com/ZhuoLinLi-shu/SDGL/blob/main/figs/model.png)

## Requirements
- python 3
- see `requirements.txt`

## Data Preparation
Raw Dataset

Traffic data  https://github.com/LeiBAI/AGCRN

Time series data https://github.com/laiguokun/multivariate-time-series-data



## Train Commands

python train_pems.py --gcn_bool --addaptadj  --dataset


## Citing

If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@article{DBLP:journals/pr/LiZYX23,
  author       = {Zhuo Lin Li and
                  Gao Wei Zhang and
                  Jie Yu and
                  Lingyu Xu},
  title        = {Dynamic graph structure learning for multivariate time series forecasting},
  journal      = {Pattern Recognit.},
  volume       = {138},
  pages        = {109423},
  year         = {2023},
  url          = {https://doi.org/10.1016/j.patcog.2023.109423},
  doi          = {10.1016/j.patcog.2023.109423},
  timestamp    = {Fri, 23 Jun 2023 22:30:47 +0200},
  biburl       = {https://dblp.org/rec/journals/pr/LiZYX23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
