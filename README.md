# Boosting Pseudo Labeling with Curriculum Self-Reflexion for Attributed Graph Clustering
PyTorch implementation of the paper "[Boosting Pseudo Labeling with Curriculum Self-Reflexion for Attributed Graph Clustering](https://ieeexplore.ieee.org/abstract/document/9777842)".


# Get started
```Shell
python main.py --sta_high=0.65  --edp=0.2 --w_pos=1 --w_pq=0  --tt=0.5   --sta_low=0.3 --w_nega=0.1 
python main.py --sta_high=0.65  --fdp=0.2 --w_pos=1 --w_pq=0  --tt=0.5  --sta_low=0.35 --w_nega=0.4 
```

# Citation

```BibTeX
@article{plcsr,
  author       = {Pengfei Zhu, Jialu Li, Yu Wang, Bin Xiao, Jinglin Zhang, Wanyu Lin, Qinghua Hu},
  title        = {Boosting Pseudo Labeling with Curriculum Self-Reflexion for Attributed Graph Clustering},
  journal      = {{IEEE} Trans. Neural Networks Learn. Syst.},
  volume       = {1},
  number       = {1},
  pages        = {1--14},
  year         = {2024},
}
```
