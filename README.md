# Boosting Pseudo Labeling with Curriculum Self-Reflexion for Attributed Graph Clustering
PyTorch implementation of the paper "[Boosting Pseudo Labeling with Curriculum Self-Reflexion for Attributed Graph Clustering](https://ieeexplore.ieee.org/abstract/document/9777842)".


# Get started
```Shell
cd ARVGA
python cgcn_mt.py --score=0.65 --de=1 --edp=0.2 --w_mt=1 --w_pq=0  --tt=0.5 --ls=0 --nl=1 --nt=0.3 --w_n=0.1 
python cgcn_mt.py --score=0.65 --fea=1 --fdp=0.2 --w_mt=1 --w_pq=0  --tt=0.5 --ls=0 --nl=1 --nt=0.35 --w_n=0.4 
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
