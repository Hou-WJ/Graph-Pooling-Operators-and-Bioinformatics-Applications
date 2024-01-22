# Graph Pooling Operators and Bioinformatics Applications 

# A Survey of Graph Pooling Operators for Graph Neural Networks and Their Applications in Bioinformatics

## Paper [**[Paper Link]**](#table-of-contents)
## Cite
If this repository is useful for your research, please consider citing our paper.
```
@article{}
```
## News

## Table of Contents
* [Overview](#Overview) 
* [Global Pooling](#global-pooling) 
* [Hierarchical Pooling](#hierarchical-pooling) 
  - [Clustering Pooling](#clustering-pooling)
  - [Selection Pooling](#selection-pooling)
  - [Edge Pooling](#edge-pooling)
  - [Hybrid Pooling](#hybrid-pooling)
  - [Graph Unpooling](#graph-unpooling)
* [Benchmark Datasets](#benchmark-datasets) 
* [Applications in Bioinformatics](#applications-in-bioinformatics)
  - [Biological Networks from Medical Images](#biological-networks-from-medical-images)
  - [Molecular Structure](#molecular-structure)
  - [Others](#others)
* [Update](#update) 

## Overview
Graph pooling is an essential component of GNNs for graph-level representations. The goal of graph pooling is to learn a graph representation that captures topology, node features, and other relational characteristics in the graph, which can be used as input to downstream machine learning tasks. Typically, there are two types of graph pooling: global pooling or readout to condense the input graph into a single vector, and hierarchical pooling to condense the input graph as a smaller-sized graph.

## Global Pooling
1. **[Neural graph fingerprints]** Duvenaud D, Maclaurin D, Aguilera-Iparraguirre J, et al (2015) Convolutional Networks on Graphs for Learning Molecular Fingerprints. In: Advances in Neural Information Processing Systems [**[Paper]**](https://proceedings.neurips.cc/paper_files/paper/2015/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html) [**[Code]**](https://github.com/HIPS/neural-fingerprint) 

2. **[DCNN]** Atwood J, Towsley D (2016) Diffusion-Convolutional Neural Networks. In: Advances in Neural Information Processing Systems [**[Paper]**](https://proceedings.neurips.cc/paper_files/paper/2016/hash/390e982518a50e280d8e2b535462ec1f-Abstract.html) [**[Code]**](https://github.com/jcatw/dcnn) 

3. **[Set2set]** Vinyals O, Bengio S, Kudlur M (2016) Order Matters: Sequence to sequence for sets. In: International Conference on Learning Representations [**[Paper]**](https://arxiv.org/abs/1511.06391v4) [**[Code]**](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/aggr/set2set.py) 

4. **[SortPooling]** Zhang M, Cui Z, Neumann M, Chen Y (2018) An End-to-End Deep Learning Architecture for Graph Classification. In: Proceedings of the AAAI Conference on Artificial Intelligence [**[Paper]**](https://doi.org/10.1609/aaai.v32i1.11782) [**[Code]**](https://github.com/muhanzhang/DGCNN) 

5. **[k-GNN]** Morris C, Ritzert M, Fey M, et al (2019) Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks. In: Proceedings of the AAAI Conference on Artificial Intelligence. pp 4602–4609 [**[Paper]**](https://doi.org/10.1609/aaai.v33i01.33014602) [**[Code]**](https://github.com/chrsmrrs/k-gnn) 

6. **[GIN]** Xu K, Hu W, Leskovec J, Jegelka S (2019) How Powerful are Graph Neural Networks? In: International Conference on Learning Representations [**[Paper]**](https://openreview.net/forum?id=ryGs6iA5Km&noteId=rkl2Q1Qi6X&noteId=rkl2Q1Qi6X) [**[Code]**](https://github.com/weihua916/powerful-gnns) 

7. **[DEMO-Net]** Wu J, He J, Xu J (2019) Demo-Net: Degree-specific graph neural networks for node and graph classification. In: Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp 406–415 [**[Paper]**](https://doi.org/10.1145/3292500.3330950) [**[Code]**](https://github.com/jwu4sml/DEMO-Net) 

8. **[GFN/GLN]** Chen T, Bian S, Sun Y (2019) Are Powerful Graph Neural Nets Necessary? A Dissection on Graph Classification. arXiv preprint arXiv:190504579 [**[Paper]**](https://arxiv.org/abs/1905.04579) [**[Code]**](https://github.com/chentingpc/gfn) 

9. **[DAGCN]** Chen F, Pan S, Jiang J, et al (2019) DAGCN: Dual Attention Graph Convolutional Networks. In: 2019 International Joint Conference on Neural Networks (IJCNN) [**[Paper]**](https://doi.org/10.1109/IJCNN.2019.8851698) [**[Code]**](https://github.com/dawenzi123/DAGCN) 

10. **[PiNet]** Meltzer, P., Mallea, M. D. G., & Bentley, P. J. (2019). PiNet: Attention Pooling for Graph Classification, NeurIPS 2019 Graph Representation Learning Workshop [**[Paper]**](https://grlearning.github.io/papers/70.pdf) [**[Code1]**](https://github.com/meltzerpete/PiNet) [**[Code2]**](https://github.com/meltzerpete/pinet2) 

11. **[SAGE]** Li J, Meng H, Rong Y, et al (2019) Semi-supervised graph classification: A hierarchical graph perspective. In: The Web Conference 2019 - Proceedings of the World Wide Web Conference, WWW 2019. pp 972–982 [**[Paper]**](https://doi.org/10.1145/3308558.3313461) [**[Code]**](https://github.com/benedekrozemberczki/SEAL-CI) 

12. **[DeepSets]** Navarin N, Tran D Van, Sperduti A (2019) Universal Readout for Graph Convolutional Neural Networks. In: 2019 International Joint Conference on Neural Networks (IJCNN) [**[Paper]**](https://doi.org/10.1109/IJCNN.2019.8852103) 


13. **[UGRAPHEMB]** Bai Y, Ding H, Qiao Y, et al (2019) Unsupervised Inductive Graph-Level Representation Learning via Graph-Graph Proximity. In: Proceedings of the 28th International Joint Conference on Artificial Intelligence. pp 1988–1994 [**[Paper]**](https://dl.acm.org/doi/abs/10.5555/3367243.3367315) [**[Code]**](https://github.com/yunshengb/UGraphEmb) 

14. **[RP]** Murphy RL, Srinivasan B, Rao V, Ribeiro B (2019) Relational Pooling for Graph Representations. In: Proceedings of the 36th International Conference on Machine Learning. pp 4663–4673 [**[Paper]**](http://proceedings.mlr.press/v97/murphy19a.html) [**[Code]**](https://github.com/PurdueMINDS/RelationalPooling) 

15. **[LPR]** Chen Z, Chen L, Villar S, Bruna J (2020) Can Graph Neural Networks Count Substructures? In: Advances in Neural Information Processing Systems. pp 10383–10395 [**[Paper]**](https://proceedings.neurips.cc/paper/2020/hash/75877cb75154206c4e65e76b88a12712-Abstract.html) [**[Code]**](https://github.com/leichen2018/GNN-Substructure-Counting) 

16. **[Structured Self-attention Architecture]** Fan X, Gong M, Xie Y, et al (2020) Structured self-attention architecture for graph-level representation learning. Pattern Recognit 100:107084. [**[Paper]**](https://doi.org/10.1016/j.patcog.2019.107084) 

17. **[SOPool]** Wang Z, Ji S (2020) Second-Order Pooling for Graph Neural Networks. IEEE Trans Pattern Anal Mach Intell 45:1–1.  [**[Paper]**](https://doi.org/10.1109/TPAMI.2020.2999032) [**[Code]**](https://github.com/divelab/sopool) 

18. **[DropGNN]** Papp PA, Martinkus K, Faber L, Wattenhofer R (2021) DropGNN: Random Dropouts Increase the Expressiveness of Graph Neural Networks. In: Advances in Neural Information Processing Systems. pp 21997–22009 [**[Paper]**](https://proceedings.neurips.cc/paper/2021/hash/b8b2926bd27d4307569ad119b6025f94-Abstract.html) [**[Code]**](https://github.com/KarolisMart/DropGNN) 

19. **[UFGPool]** Zheng X, Zhou B, Gao J, et al (2021) How Framelets Enhance Graph Neural Networks. In: Proceedings of the 38th International Conference on Machine Learning. pp 12761–12771 [**[Paper]**](http://proceedings.mlr.press/v139/zheng21c.html) [**[Code]**](https://github.com/YuGuangWang/UFG) 

20. **[SSRead]** Lee D, Kim S, Lee S, et al (2021) Learnable Structural Semantic Readout for Graph Classification. In: 2021 IEEE International Conference on Data Mining (ICDM). IEEE, pp 1180–1185 [**[Paper]**](https://doi.org/10.1109/ICDM51629.2021.00142) [**[Code]**](https://github.com/donalee/SSRead) 

21. **[De‑correlation pooling]** Li X, Wu H (2021) Toward graph classification on structure property using adaptive motif based on graph convolutional network. J Supercomput 77:8767–8786.  [**[Paper]**](https://doi.org/10.1007/s11227-021-03628-4) 

22. **[DiP-Readout]** Roy KK, Roy A, Mahbubur Rahman AKM, et al (2021) Structure-Aware Hierarchical Graph Pooling using Information Bottleneck. In: 2021 International Joint Conference on Neural Networks (IJCNN) [**[Paper]**](https://doi.org/10.1109/IJCNN52387.2021.9533778) [**[Code]**](https://github.com/forkkr/HIBPool) 

23. **[GMT]** Baek J, Kang M, Hwang SJ (2021) Accurate Learning of Graph Representations with Graph Multiset Pooling. In: International Conference on Learning Representations [**[Paper]**](https://openreview.net/forum?id=JHcqXGaqiGn) [**[Code]**](https://github.com/JinheonBaek/GMT) 

24. **[GraphTrans]** Wu Z, Jain P, Wright M, et al (2021) Representing Long-Range Context for Graph Neural Networks with Global Attention. In: Advances in Neural Information Processing Systems. pp 13266–13279 [**[Paper]**](https://proceedings.neurips.cc/paper/2021/hash/6e67691b60ed3e4a55935261314dd534-Abstract.html) [**[Code]**](https://github.com/ucbrise/graphtrans) 

25. **[QSGCNN]** Bai L, Jiao Y, Cui L, et al (2021) Learning Graph Convolutional Networks based on Quantum Vertex Information Propagation. IEEE Trans Knowl Data Eng 1747–1760.  [**[Paper]**](https://doi.org/10.1109/TKDE.2021.3106804) 

26. **[LRP]** Chen L, Chen Z, Bruna J (2021) Learning the Relevant Substructures for Tasks on Graph Data. In: ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, pp 8528–8532 [**[Paper]**](https://doi.org/10.1109/ICASSP39728.2021.9414377)

27. **[DKEPool]** Chen K, Song J, Liu S, et al (2022) Distribution Knowledge Embedding for Graph Pooling. IEEE Trans Knowl Data Eng 7898–7908.  [**[Paper]**](https://doi.org/10.1109/TKDE.2022.3208063) [**[Code]**](https://github.com/chenchkx/dkepool) 

28. **[MLAP]** Itoh TD, Kubo T, Ikeda K (2022) Multi-level attention pooling for graph neural networks: Unifying graph representations with multiple localities. Neural Networks 145:356–373. [**[Paper]**](https://doi.org/10.1016/j.neunet.2021.11.001)

29. **[Adaptive Readouts]** Buterez D, Janet JP, Kiddle SJ, et al (2022) Graph Neural Networks with Adaptive Readouts. In: Advances in Neural Information Processing Systems. pp 19746–19758 [**[Paper]**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7caf9d251b546bc78078b35b4a6f3b7e-Abstract-Conference.html) [**[Code]**](https://github.com/davidbuterez/gnn-neural-readouts) 


## Hierarchical Pooling
### Clustering Pooling
0. **[CapsGNN]** Xinyi Z, Chen L (2018) Capsule graph neural network. In: International Conference on Learning Representations  [**[Paper]**](https://openreview.net/forum?id=Byl8BnRcYm) [**[Code]**](https://github.com/benedekrozemberczki/CapsGNN) 

0. **[DiffPool]** Ying Z, You J, Morris C, et al (2018) Hierarchical Graph Representation Learning with Differentiable Pooling. In: Advances in Neural Information Processing Systems [**[Paper]**](https://proceedings.neurips.cc/paper_files/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html) [**[Code]**](https://github.com/RexYing/diffpool) 

0. **[GRAHIES]** Yu L, Zhang Q, DIllenberger D, et al (2019) GRAHIES: Multi-scale graph representation learning with latent hierarchical structure. In: Proceedings - 2019 IEEE 1st International Conference on Cognitive Machine Intelligence, CogMI 2019. pp 8–15 [**[Paper]**](https://doi.org/10.1109/CogMI48466.2019.00011)

0. **[LaPool]** Noutahi E, Beaini D, Horwood J, et al (2019) Towards Interpretable Sparse Graph Representation Learning with Laplacian Pooling. arXiv preprint arXiv:190511577 [**[Paper]**](https://arxiv.org/abs/1905.11577v4)

0. **[H-GCN]** Hu F, Zhu Y, Wu S, et al (2019) Hierarchical Graph Convolutional Networks for Semi-supervised Node Classification. In: Proceedings of the 28th International Joint Conference on Artificial Intelligence. pp 4532–4539 [**[Paper]**](https://dl.acm.org/doi/abs/10.5555/3367471.3367673) [**[Code]**](https://github.com/CRIPAC-DIG/H-GCN) 

0. **[NMFPool]** Bacciu D, Di Sotto L (2019) A Non-negative Factorization Approach to Node Pooling in Graph Convolutional Neural Networks. In: International Conference of the Italian Association for Artificial Intelligence. pp 294–306 [**[Paper]**](https://link.springer.com/chapter/10.1007/978-3-030-35166-3_21) 

0. **[EigenPooling]** Ma Y, Wang S, Aggarwal CC, Tang J (2019) Graph Convolutional Networks with EigenPooling. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. pp 723–731 [**[Paper]**](https://dl.acm.org/doi/abs/10.1145/3292500.3330982) [**[Code]**](https://github.com/alge24/eigenpooling) 

0. **[Clique Pooling]** Luzhnica E, Day B, Lio’ P (2019) Clique pooling for graph classification. arXiv preprint arXiv:190400374 [**[Paper]**](https://arxiv.org/abs/1904.00374v2) 

0. **[HaarPooling]** Wang YG, Li M, Ma Z, et al (2020) Haar Graph Pooling. In: Proceedings of the 37th International Conference on Machine Learning. pp 9952–9962 [**[Paper]**](https://proceedings.mlr.press/v119/wang20m.html) [**[Code]**](https://github.com/YuGuangWang/HaarPool) 

0. **[GMN]** Khasahmadi AH, Hassani K, Moradi P, et al (2020) Memory-Based Graph Networks. In: International Conference on Learning Representations [**[Paper]**](https://openreview.net/forum?id=r1laNeBYPB) [**[Code]**](https://github.com/amirkhas/GraphMemoryNet) 

0. **[MuchGNN]** Zhou K, Song Q, Huang X, et al (2020) Multi-Channel Graph Neural Networks. In: Proceedings of the 29th International Joint Conference on Artificial Intelligence, IJCAI 2020. pp 1352–1358 [**[Paper]**](https://dl.acm.org/doi/abs/10.5555/3491440.3491628)

0. **[MxPool]** Liang Y, Zhang Y, Gao D, Xu Q (2020) MxPool: Multiplex Pooling for Hierarchical Graph Representation Learning. arXiv preprint arXiv:200406846 [**[Paper]**](https://arxiv.org/abs/2004.06846) [**[Code]**](https://github.com/JucatL/MxPool/) 

0. **[SOPool]** Wang Z, Ji S (2020) Second-Order Pooling for Graph Neural Networks. IEEE Trans Pattern Anal Mach Intell 45:1–1.  [**[Paper]**](https://doi.org/10.1109/TPAMI.2020.2999032) [**[Code]**](https://github.com/divelab/sopool)

0. **[GraPHmax]** Bandyopadhyay S, Aggarwal M, Murty MN (2020a) Self-supervised Hierarchical Graph Neural Network for Graph Representation. In: 2020 IEEE International Conference on Big Data (Big Data). IEEE, pp 603–608 [**[Paper]**](https://doi.org/10.1109/BigData50022.2020.9377860) [**[Code]**](https://github.com/manasviaggarwal/GraPHmax) 

0. **[MinCutPool]** Maria Bianchi F, Grattarola D, Alippi C (2020) Spectral Clustering with Graph Neural Networks for Graph Pooling. In: Proceedings of the 37th International Conference on Machine Learning. pp 874–883 [**[Paper]**](https://proceedings.mlr.press/v119/bianchi20a.html) [**[Code]**](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling) 

0. **[StructPool]** Yuan H, Ji S (2020) StructPool: Structured Graph Pooling via Conditional Random Fields. In: International Conference on Learning Representations [**[Paper]**](https://openreview.net/forum?id=BJxg_hVtwH) [**[Code]**](https://github.com/Nate1874/StructPool) 

0. **[ASAP]** Ranjan E, Sanyal S, Talukdar P (2020) ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations. In: Proceedings of the AAAI Conference on Artificial Intelligence. pp 5470–5477 [**[Paper]**](https://doi.org/10.1609/aaai.v34i04.5997) [**[Code]**](https://github.com/malllabiisc/ASAP) 

0. **[MLC-GCN]** Xie Y, Yao C, Gong M, et al (2020) Graph convolutional networks with multi-level coarsening for graph classification. Knowl Based Syst 194:105578.  [**[Paper]**](https://doi.org/10.1016/j.knosys.2020.105578) 

0. **[SubGattPool]** Bandyopadhyay S, Aggarwal M, Murty MN (2020) Robust Hierarchical Graph Classification with Subgraph Attention. arXiv preprint arXiv:200710908 [**[Paper]**](https://arxiv.org/abs/2007.10908) [**[Code]**]() 

0. **[SUGAR]** Sun Q, Li J, Peng H, et al (2021) SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism. In: Proceedings of the Web Conference 2021. pp 2081–2091 [**[Paper]**](https://doi.org/10.1145/3442381.3449822) [**[Code]**](https://github.com/RingBDStack/SUGAR) 

0. **[LCP]** Su Z, Hu Z, Li Y (2021) Hierarchical Graph Representation Learning with Local Capsule Pooling. In: ACM Multimedia Asia. pp 1–7 [**[Paper]**](https://doi.org/10.1145/3469877.3495645) [**[Code]**](https://github.com/imSeaton/LocalCapsulePoolingNetwork) 

0. **[GSAPool]** Yu H, Yuan J, Cheng H, et al (2021) GSAPool: Gated Structure Aware Pooling for Graph Representation Learning. In: 2021 International Joint Conference on Neural Networks (IJCNN). IEEE, pp 1–8 [**[Paper]**](https://doi.org/10.1109/IJCNN52387.2021.9534320) 

0. **[FPool]** Pham H Van, Thanh DH, Moore P (2021) Hierarchical Pooling in Graph Neural Networks to Enhance Classification Performance in Large Datasets. Sensors 21:6070.  [**[Paper]**](https://doi.org/10.3390/s21186070) 

0. **[HGCN]** Yang J, Zhao P, Rong Y, et al (2021a) Hierarchical Graph Capsule Network. Proceedings of the AAAI Conference on Artificial Intelligence 35:10603–10611.  [**[Paper]**](https://doi.org/10.1609/aaai.v35i12.17268) [**[Code]**](https://github.com/uta-smile/HGCN) 

0. **[HAP]** Liu N, Jian S, Li D, et al (2021) Hierarchical Adaptive Pooling by Capturing High-order Dependency for Graph Representation Learning. IEEE Trans Knowl Data Eng 3952-3965. [**[Paper]**](https://doi.org/10.1109/TKDE.2021.3133646) [**[Code]**]() 
0. **[DMP/MPR]** Bodnar C, Cangea C, Liò P (2021) Deep Graph Mapper: Seeing Graphs Through the Neural Lens. Front Big Data 4:680535.  [**[Paper]**](https://doi.org/10.3389/fdata.2021.680535) [**[Code]**](https://github.com/crisbodnar/dgm) 

0. **[HIBPool]** Roy KK, Roy A, Mahbubur Rahman AKM, et al (2021) Structure-Aware Hierarchical Graph Pooling using Information Bottleneck. In: 2021 International Joint Conference on Neural Networks (IJCNN). IEEE, pp 1–8 [**[Paper]**](https://doi.org/10.1109/IJCNN52387.2021.9533778) [**[Code]**](https://github.com/forkkr/HIBPool) 

0. **[KPlexPool]** Bacciu D, Conte A, Grossi R, et al (2021) K-plex cover pooling for graph neural networks. Data Min Knowl Discov 35:2200–2220. [**[Paper]**](https://doi.org/10.1007/s10618-021-00779-z) [**[Code]**](https://github.com/flandolfi/kplex-pool/) 

0. **[CommPOOL]** Tang H, Ma G, He L, et al (2021) CommPOOL: An interpretable graph pooling framework for hierarchical graph representation learning. Neural Networks 143:669–677.  [**[Paper]**](https://doi.org/10.1016/j.neunet.2021.07.028) [**[Code]**]() 

0. **[HGP-SACA]** Li ZP, Su HL, Zhu XB, et al (2022) Hierarchical Graph Pooling With Self-Adaptive Cluster Aggregation. IEEE Trans Cogn Dev Syst 14:1198–1207.  [**[Paper]**](https://doi.org/10.1109/TCDS.2021.3100883)

0. **[SEP]** Wu J, Chen X, Xu K, Li S (2022) Structural Entropy Guided Graph Hierarchical Pooling. In: Proceedings of the 39th International Conference on Machine Learning. PMLR, pp 24017–24030 [**[Paper]**](https://proceedings.mlr.press/v162/wu22b.html) [**[Code]**](https://github.com/Wu-Junran/SEP) 

0. **[SMIP]** Liu N, Jian S, Li D, Xu H (2022) Unsupervised Hierarchical Graph Pooling via Substructure-Sensitive Mutual Information Maximization. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management. pp 1299–1308 [**[Paper]**](https://doi.org/10.1145/3511808.3557485)

0. **[HoscPool]** Duval A, Malliaros F (2022) Higher-Order Clustering and Pooling for Graph Neural Networks. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management. pp 426–435 [**[Paper]**](https://doi.org/10.1145/3511808.3557353) [**[Code]**](https://github.com/AlexDuvalinho/HoscPool) 

0. **[AdamGNN]** Zhong Z, Li C-T, Pang J (2022) Multi-grained Semantics-aware Graph Neural Networks. IEEE Trans Knowl Data Eng 7251-7262.  [**[Paper]**](https://doi.org/10.1109/TKDE.2022.3195004) [**[Code]**](https://github.com/zhiqiangzhongddu/AdamGNN) 

0. **[MATHNET]** Zheng X, Zhou B, Li M, et al (2020) MathNet: Haar-Like Wavelet Multiresolution-Analysis for Graph Representation and Learning. arXiv preprint arXiv:200711202 [**[Paper]**](https://doi.org/10.1016/j.knosys.2023.110609) 

### Selection Pooling
0. **[AttPool]** Huang J, Li Z, Li N, et al (2019) Attpool: Towards hierarchical feature representation in graph convolutional networks via attention mechanism. In: Proceedings of the IEEE International Conference on Computer Vision. pp 6479–6488 [**[Paper]**](https://openaccess.thecvf.com/content_ICCV_2019/html/Huang_AttPool_Towards_Hierarchical_Feature_Representation_in_Graph_Convolutional_Networks_via_ICCV_2019_paper.html) [**[Code]**](https://github.com/hjjpku/Attention_in_Graph) 

0. **[gPool]** Gao H, Ji S (2019) Graph U-Nets. In: Proceedings of the 36th International Conference on Machine Learning. pp 2083--2092 [**[Paper]**](https://proceedings.mlr.press/v97/gao19a.html) [**[Code]**](https://github.com/HongyangGao/Graph-U-Nets) 

0. **[HGP-SL]** Zhang Z, Bu J, Ester M, et al (2019) Hierarchical Graph Pooling with Structure Learning. arXiv preprint arXiv:191105954 [**[Paper]**](https://arxiv.org/abs/1911.05954) [**[Code]**](https://github.com/cszhangzhen/HGP-SL) 

0. **[SAGPool]** Lee J, Lee I, Kang J (2019) Self-Attention Graph Pooling. In: Proceedings of the 36th International Conference on Machine Learning. pp 3734–3743 [**[Paper]**](https://proceedings.mlr.press/v97/lee19c.html) [**[Code]**](https://github.com/inyeoplee77/SAGPool) 

0. **[CovPooling]** Jiang J, Lei F, Dai Q, Li Z (2020) Graph pooling in graph neural networks with node feature correlation. In: Proceedings of the 3rd International Conference on Data Science and Information Technology. pp 105–110 [**[Paper]**](https://doi.org/10.1145/3414274.3414490) 

0. **[UGPool]** Qin J, Liu L, Shen H, Hu D (2020) Uniform Pooling for Graph Networks. Applied Sciences 10:6287.  [**[Paper]**](https://doi.org/10.3390/app10186287) [**[Code]**](https://github.com/Qin-J/Uniform-graph-pooling) 

0. **[GSAPool]** Zhang L, Wang X, Li H, et al (2020) Structure-Feature based Graph Self-adaptive Pooling. In: Proceedings of The Web Conference 2020. pp 3098–3104 [**[Paper]**](https://doi.org/10.1145/3366423.3380083) [**[Code]**](https://github.com/psp3dcg/gsapool) 

0. **[PANPool]** Ma Z, Xuan J, Wang YG, et al (2020) Path Integral Based Convolution and Pooling for Graph Neural Networks. In: Advances in Neural Information Processing Systems. pp 16421–16433 [**[Paper]**](https://proceedings.neurips.cc/paper/2020/hash/be53d253d6bc3258a8160556dda3e9b2-Abstract.html) [**[Code]**](https://github.com/YuGuangWang/PAN) 

0. **[VIPool]** Li M, Chen S, Zhang Y, Tsang IW (2020b) Graph Cross Networks with Vertex Infomax Pooling. In: Advances in Neural Information Processing Systems [**[Paper]**](https://proceedings.neurips.cc/paper/2020/hash/a26398dca6f47b49876cbaffbc9954f9-Abstract.html) [**[Code]**](https://github.com/limaosen0/GXN) 

0. **[RepPool]** Li J, Ma Y, Wang Y, et al (2020a) Graph pooling with representativeness. In: Proceedings - IEEE International Conference on Data Mining, ICDM. pp 302–311 [**[Paper]**](https://doi.org/10.1109/ICDM50108.2020.00039) [**[Code]**](https://github.com/Juanhui28/RepPool/tree/master/RepPool) 

0. **[CGIPool]** Pang Y, Zhao Y, Li D (2021) Graph Pooling via Coarsened Graph Infomax. In: SIGIR 2021 - Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp 2177–2181 [**[Paper]**](https://doi.org/10.1145/3404835.3463074) [**[Code]**](https://github.com/PangYunsheng8/CGIPool) 

0. **[MVPool]** Zhang Z, Bu J, Ester M, et al (2021) Hierarchical Multi-View Graph Pooling with Structure Learning. IEEE Trans Knowl Data Eng 545-559. [**[Paper]**](https://doi.org/10.1109/TKDE.2021.3090664) [**[Code]**](https://github.com/cszhangzhen/MVPool) 

0. **[HTAP]** Bi L, Sun X, Zhou F, Dong J (2021) Hierarchical Triplet Attention Pooling for Graph Classification. In: 2021 IEEE 33rd International Conference on Tools with Artificial Intelligence (ICTAI). IEEE, pp 624–631 [**[Paper]**](https://doi.org/10.1109/ICTAI52525.2021.00100) 

0. **[R2POOL]** Aggarwal M, Murty MN (2021) Region and Relations Based Multi Attention Network for Graph Classification. In: 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, pp 8101–8108 [**[Paper]**](https://doi.org/10.1109/ICPR48806.2021.9413216) 

0. **[SMG]** Yang M, Shen Y, Qi H, Yin B (2021b) Soft-mask: Adaptive Substructure Extractions for Graph Neural Networks. In: Proceedings of the Web Conference 2021. pp 2058–2068 [**[Paper]**](https://doi.org/10.1145/3442381.3449929) [**[Code]**](https://github.com/qslim/soft-mask-gnn) 

0. **[TAP]** Gao H, Liu Y, Ji S (2021a) Topology-Aware Graph Pooling Networks. IEEE Trans Pattern Anal Mach Intell 43:4512–4518.  [**[Paper]**](https://doi.org/10.1109/TPAMI.2021.3062794)

0. **[OTCoarsening]** Ma T, Chen J (2021) Unsupervised Learning of Graph Hierarchical Abstractions with Differentiable Coarsening and Optimal Transport. Proceedings of the AAAI Conference on Artificial Intelligence 35:8856–8864.  [**[Paper]**](https://doi.org/10.1609/aaai.v35i10.17072) [**[Code]**](https://github.com/matenure/OTCoarsening) 

0. **[MEWISPool]** Nouranizadeh A, Matinkia M, Rahmati M, Safabakhsh R (2021) Maximum Entropy Weighted Independent Set Pooling for Graph Neural Networks. arXiv preprint arXiv:210701410 [**[Paper]**](https://arxiv.org/abs/2107.01410) [**[Code]**](https://github.com/mewispool/mewispool) 

0. **[LiftPool]** Xu M, Dai W, Li C, et al (2022a) LiftPool: Lifting-based Graph Pooling for Hierarchical Graph Representation Learning. arXiv preprint arXiv:220412881 [**[Paper]**](https://arxiv.org/abs/2204.12881)

0. **[MIVSPool]** Stanovic Stevan and Gaüzère B and BL (2022) Maximal Independent Vertex Set Applied to Graph Pooling. In: Structural, Syntactic, and Statistical Pattern Recognition. pp 11–21 [**[Paper]**](https://link.springer.com/chapter/10.1007/978-3-031-23028-8_2) 

0. **[NCPool]** Wang Y, Chang D, Fu Z, Zhao Y (2022) Seeing All From a Few: Nodes Selection Using Graph Pooling for Graph Clustering. IEEE Trans Neural Netw Learn Syst 1–7.  [**[Paper]**](https://doi.org/10.1109/TNNLS.2022.3210370) 

0. **[RCGNN]** Duan Y, Wang J, Ma H, Sun Y (2022) Residual convolutional graph neural network with subgraph attention pooling. Tsinghua Sci Technol 27:653–663.  [**[Paper]**](https://doi.org/10.26599/TST.2021.9010058)

0. **[MSAPool]** Xu Y, Wang J, Guang M, et al (2022b) Multistructure Graph Classification Method With Attention-Based Pooling. IEEE Trans Comput Soc Syst 602-613.  [**[Paper]**](https://doi.org/10.1109/TCSS.2022.3169219) [**[Code]**](https://github.com/xyhappy/MAC) 

0. **[attnPool]** Gao H, Ji S (2022) Graph U-Nets. IEEE Trans Pattern Anal Mach Intell 44:4948–4960.  [**[Paper]**](https://doi.org/10.1109/TPAMI.2021.3081010)

0. **[iPool]** Gao X, Dai W, Li C, et al (2022) iPool—Information-Based Pooling in Hierarchical Graph Neural Networks. IEEE Trans Neural Netw Learn Syst 33:5032–5044.  [**[Paper]**](https://doi.org/10.1109/TNNLS.2021.3067441)

0. **[NDP]** Bianchi FM, Grattarola D, Livi L, Alippi C (2022) Hierarchical Representation Learning in Graph Neural Networks With Node Decimation Pooling. IEEE Trans Neural Netw Learn Syst 33:2195–2207.  [**[Paper]**](https://doi.org/10.1109/TNNLS.2020.3044146) [**[Code]**](https://github.com/danielegrattarola/decimation-pooling) 

0. **[knnPool]** Chen C, Li K, Wei W, et al (2022a) Hierarchical Graph Neural Networks for Few-Shot Learning. IEEE Transactions on Circuits and Systems for Video Technology 32:240–252.  [**[Paper]**](https://doi.org/10.1109/TCSVT.2021.3058098) [**[Code]**](https://github.com/smartprobe/HGNN) 

0. **[MID]** Liu C, Zhan Y, Yu B, et al (2023) On exploring node-feature and graph-structure diversities for node drop graph pooling. Neural Networks 167:559–571. [**[Paper]**](https://doi.org/https://doi.org/10.1016/j.neunet.2023.08.046) [**[Code]**](https://github.com/whuchuang/mid) 

### Edge Pooling
1. **[EdgePool]** Diehl F, Brunner T, Le MT, Knoll A (2019) Towards Graph Pooling by Edge Contraction. In: ICML 2019 Workshop on Learning and Reasoning with Graph-Structured Data [**[Paper1]**](https://arxiv.org/abs/1905.10990) [**[Paper2]**](https://graphreason.github.io/papers/17.pdf)

2. **[MeshPooling]** Yuan YJ, Lai YK, Yang J, et al (2020) Mesh variational autoencoders with edge contraction pooling. In: IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops. pp 1105–1112 [**[Paper]**](https://openaccess.thecvf.com/content_CVPRW_2020/html/w17/Yuan_Mesh_Variational_Autoencoders_With_Edge_Contraction_Pooling_CVPRW_2020_paper.html) [**[Code]**](https://github.com/IGLICT/MeshPooling) 

3. **[DHT]** Jo J, Baek J, Lee S, et al (2021) Edge Representation Learning with Hypergraphs. In: Advances in Neural Information Processing Systems [**[Paper]**](https://proceedings.neurips.cc/paper_files/paper/2021/hash/3def184ad8f4755ff269862ea77393dd-Abstract.html) [**[Code]**](https://github.com/harryjo97/EHGNN) 

4. **[H2MN]** Zhang Z, Bu J, Ester M, et al (2021b) H2MN: Graph Similarity Learning with Hierarchical Hypergraph Matching Networks. In: Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp 2274–2284 [**[Paper]**](https://doi.org/10.1145/3447548.3467328) [**[Code]**](https://github.com/cszhangzhen/H2MN) 

### Hybrid Pooling
1. **[LookHops]** Gao Z, Lin H, Li StanZ (2020) LookHops: light multi-order convolution and pooling for graph classification. arXiv preprint arXiv:201215741 [**[Paper]**](https://arxiv.org/abs/2012.15741)
2. **[ProxPool]** Gao X, Dai W, Li C, et al (2021b) Multiscale Representation Learning of Graph Data With Node Affinity. IEEE Trans Signal Inf Process Netw 7:30–44.  [**[Paper]**](https://doi.org/10.1109/TSIPN.2020.3044913)

3. **[ASPool]** Yu H, Yuan J, Yao Y, Wang C (2022) Not all edges are peers: Accurate structure-aware graph pooling networks. Neural Networks 156:58–66. [**[Paper]**](https://doi.org/10.1016/j.neunet.2022.09.004)

4. **[Co-Pooling]** Zhou Xiaowei and Yin J and TIW (2023) Edge but not Least: Cross-View Graph Pooling. In: Machine Learning and Knowledge Discovery in Databases. pp 344–359 [**[Paper]**](https://doi.org/10.1007/978-3-031-26390-3_21)
### Graph Unpooling
1. **[UL]** Guo Y, Zou D, Lerman G (2023) An Unpooling Layer for Graph Generation. In: Proceedings of The 26th International Conference on Artificial Intelligence and Statistics. PMLR, pp 3179–3209 [**[Paper]**](https://proceedings.mlr.press/v206/guo23a.html) [**[Code]**](https://github.com/guo00413/graph_unpooling) 

## Benchmark Datasets
<table border="1">
    <tr>
        <th align="center">Datasets</th>
        <th align="center">Name</th> 
        <th align="center">Description</th>
        <th align="center">URL</th>
    </tr>
    <tr>
        <td rowspan="15">TUDataset</td>
        <td>FRANKENSTEIN</td>
        <td rowspan="6">Small molecules</td>
        <td rowspan="15"> www.graphlearning.io <br> https://chrsmrrs.github.io/datasets/ </td>
    </tr>
    <tr>
        <td>Mutagenicity</td>
    </tr>
    <tr>
        <td>MUTAG</td>
    </tr>
    <tr>
        <td>NCI1</td>
    </tr>
    <tr>
        <td>NCI109</td>
    </tr>
    <tr>
        <td>PTC_MR</td>
    </tr>
    <tr>
        <td>DD</td>
        <td rowspan="3">Bioinformatics</td>
    </tr>
    <tr>
        <td>ENZYMES</td>
    </tr>
    <tr>
        <td>PROTEINS</td>
    </tr>
    <tr>
        <td>COLLAB</td>
        <td rowspan="6">Social networks</td>
    </tr>
    <tr>
        <td>IMDB-BINARY</td>
    </tr>
    <tr>
        <td>IMDB-MULTI</td>
    </tr>
    <tr>
        <td>REDDIT-BINARY</td>
    </tr>
    <tr>
        <td>REDDIT-MULTI-5K</td>
    </tr>
    <tr>
        <td>REDDIT-MULTI-12K</td>
    </tr>
    <tr>
    <td rowspan="5">Open Graph Benchmark (OGB)</td>
        <td>Ogbg-molhiv (HIV)</td>
        <td rowspan="4">Small scale, Molecular property prediction</td>
        <td rowspan="5"> https://ogb.stanford.edu </td>
    </tr>
    <tr>
        <td>Ogbg-molbbbp (BBBP)</td>
    </tr>
    <tr>
        <td>Ogbg-moltox21 (Tox21)</td>
    </tr>
     <tr>
        <td>Ogbg-moltoxcast (ToxCast)</td>
    </tr>
    <tr>
        <td>Ogbg-ppa (PPA)</td>
        <td>Medium scale, Protein classification</td>
    </tr>
</table>

## Evaluation, Review and Analysis
Mesquita, D., Souza, A., & Kaski, S. (2020). Rethinking pooling in graph neural networks. Advances in Neural Information Processing Systems, 33, 2220-2231.
1. Mesquita, D., Souza, A., & Kaski, S. (2020). Rethinking pooling in graph neural networks. Advances in Neural Information Processing Systems, 33, 2220-2231. [**[Paper]**](https://proceedings.neurips.cc/paper/2020/hash/1764183ef03fc7324eb58c3842bd9a57-Abstract.html) [**[Code]**](https://github.com/AaltoPML/Rethinking-pooling-in-GNNs) 

1. Grattarola, D., Zambon, D., Bianchi, F. M., & Alippi, C. (2022). Understanding pooling in graph neural networks. IEEE Transactions on Neural Networks and Learning Systems. [**[Paper]**](https://doi.org/10.1109/TNNLS.2022.3190922) [**[Code]**](https://github.com/danielegrattarola/SRC) 

1. Bianchi, F. M., & Lachi, V. (2023). The expressive power of pooling in graph neural networks. arXiv preprint arXiv:2304.01575. [**[Paper]**](https://arxiv.org/abs/2304.01575) [**[Code]**](https://github.com/FilippoMB/The-expressive-power-of-pooling-in-GNNs) 

1. Liu, C., Zhan, Y., Wu, J., Li, C., Du, B., Hu, W., ... & Tao, D. (2022). Graph pooling for graph neural networks: Progress, challenges, and opportunities. arXiv preprint arXiv:2204.07321.  [**[Paper]**]()

1. Cheung, M., Shi, J., Jiang, L., Wright, O., & Moura, J. M. (2019, November). Pooling in graph convolutional neural networks. In 2019 53rd Asilomar Conference on Signals, Systems, and Computers (pp. 462-466). IEEE. [**[Paper]**](https://doi.org/10.1109/IEEECONF44664.2019.9048796) 

1. Errica, F., Podda, M., Bacciu, D., & Micheli, A. (2019, September). A Fair Comparison of Graph Neural Networks for Graph Classification. In International Conference on Learning Representations. [**[Paper]**](https://openreview.net/forum?id=HygDF6NFPB) [**[Code]**](https://github.com/diningphil/gnn-comparison) 

## Applications in Bioinformatics
### Biological Networks from Medical Images
1. **[BrainGNN]** Li, X., Zhou, Y., Dvornek, N., Zhang, M., Gao, S., Zhuang, J., ... & Duncan, J. S. (2021). Braingnn: Interpretable brain graph neural network for fmri analysis. Medical Image Analysis, 74, 102233. [**[Paper]**](https://doi.org/10.1016/j.media.2021.102233) [**[Code]**](https://github.com/xxlya/BrainGNN_Pytorch) 
0. **[DH-SAGPool]** Zhang, S., Wang, J., Yu, S., Wang, R., Han, J., Zhao, S., ... & Lv, J. (2023). An explainable deep learning framework for characterizing and interpreting human brain states. Medical Image Analysis, 83, 102665. [**[Paper]**](https://doi.org/10.1016/j.media.2022.102665)
0. **[GAT-LI]** Hu, J., Cao, L., Li, T., Dong, S., & Li, P. (2021). GAT-LI: a graph attention network based learning and interpreting method for functional brain network classification. BMC bioinformatics, 22(1), 1-20. [**[Paper]**](https://doi.org/10.1186/s12859-021-04295-1) [**[Code]**](https://github.com/largeapp/gat-li) 
0. **[HSGPL]** Tang, H., Ma, G., Guo, L., Fu, X., Huang, H., & Zhan, L. (2022). Contrastive brain network learning via hierarchical signed graph pooling model. IEEE Transactions on Neural Networks and Learning Systems. [**[Paper]**](https://doi.org/10.1109/TNNLS.2022.3220220)
0. **[GC+P]** Gopinath, K., Desrosiers, C., & Lombaert, H. (2020). Learnable pooling in graph convolutional networks for brain surface analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(2), 864-876.  [**[Paper]**](https://doi.org/10.1109/TPAMI.2020.3028391) [**[Code]**](https://github.com/kharitz/learnpool.git) 
0. **[Multi-Channel Pooling]** Song, X., Zhou, F., Frangi, A. F., Cao, J., Xiao, X., Lei, Y., ... & Lei, B. (2022). Multicenter and Multichannel Pooling GCN for Early AD Diagnosis Based on Dual-Modality Fused Brain Network. IEEE Transactions on Medical Imaging, 42(2), 354-367. [**[Paper]**](https://doi.org/10.1109/TMI.2022.3187141) [**[Code]**](https://github.com/Xuegang-S) 
0. **[MM-GNN]** Sebenius, I., Campbell, A., Morgan, S. E., Bullmore, E. T., & Liò, P. (2021, October). Multimodal graph coarsening for interpretable, MRI-based brain graph neural network. In 2021 IEEE 31st International Workshop on Machine Learning for Signal Processing (MLSP) (pp. 1-6). IEEE. [**[Paper]**](https://doi.org/10.1109/MLSP52302.2021.9690626) 
0. **[PR-GNN]** Li, X., Zhou, Y., Dvornek, N. C., Zhang, M., Zhuang, J., Ventola, P., & Duncan, J. S. (2020). Pooling regularized graph neural network for fmri biomarker analysis. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part VII 23 (pp. 625-635). Springer International Publishing. [**[Paper]**](https://doi.org/10.1007/978-3-030-59728-3_61)
0. **[SA-GCN]** Zhao, F., Li, N., Pan, H., Chen, X., Li, Y., Zhang, H., ... & Cheng, D. (2022). Multi-view feature enhancement based on self-attention mechanism graph convolutional network for autism spectrum disorder diagnosis. Frontiers in human neuroscience, 16, 918969. [**[Paper]**](https://doi.org/10.3389/fnhum.2022.918969) 
0. **[TMGCN]** Gao, Y., Tang, Y., Zhang, H., Yang, Y., Dong, T., & Jia, Q. (2022). Sex differences of cerebellum and cerebrum: evidence from graph convolutional network. Interdisciplinary Sciences: Computational Life Sciences, 14(2), 532-544. [**[Paper]**](https://doi.org/10.1007/s12539-021-00498-5) 
0. **[CNN-GCN]** Gao, Z., Lu, Z., Wang, J., Ying, S., & Shi, J. (2022). A convolutional neural network and graph convolutional network based framework for classification of breast histopathological images. IEEE Journal of Biomedical and Health Informatics, 26(7), 3163-3173. [**[Paper]**](https://doi.org/10.1109/JBHI.2022.3153671) 
0. **[b-HGFN]** Di, D., Zhang, J., Lei, F., Tian, Q., & Gao, Y. (2022). Big-hypergraph factorization neural network for survival prediction from whole slide image. IEEE Transactions on Image Processing, 31, 1149-1160. [**[Paper]**](https://doi.org/10.1109/TIP.2021.3139229) 
0. **[HACT]** Pati, P., Jaume, G., Foncubierta-Rodriguez, A., Feroce, F., Anniciello, A. M., Scognamiglio, G., ... & Gabrani, M. (2022). Hierarchical graph representations in digital pathology. Medical image analysis, 75, 102264. [**[Paper]**](https://doi.org/10.1016/j.media.2021.102264) [**[Code]**](https://github.com/histocartography/hact-net) 
0. **[MULTIPLAI]** Martin-Gonzalez, P., Crispin-Ortuzar, M., & Markowetz, F. (2021). Predictive modelling of highly multiplexed tumour tissue images by graph neural networks. In Interpretability of Machine Intelligence in Medical Image Computing, and Topological Data Analysis and Its Applications for Medical Data: 4th International Workshop, iMIMIC 2021, and 1st International Workshop, TDA4MedicalData 2021, Held in Conjunction with MICCAI 2021, Strasbourg, France, September 27, 2021, Proceedings 4 (pp. 98-107). Springer International Publishing. [**[Paper]**](https://doi.org/10.1007/978-3-030-87444-5_10) [**[Code]**]( https://github.com/markowetzlab/MULTIPLAI) 
0. **[GNN-MIL]** Adnan, M., Kalra, S., & Tizhoosh, H. R. (2020). Representation learning of histopathology images using graph neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 988-989). [**[Paper]**](https://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Adnan_Representation_Learning_of_Histopathology_Images_Using_Graph_Neural_Networks_CVPRW_2020_paper.html)
### Molecular Structure
1. **[SIGN]** Li, S., Zhou, J., Xu, T., Huang, L., Wang, F., Xiong, H., ... & Xiong, H. (2021, August). Structure-aware interactive graph neural networks for the prediction of protein-ligand binding affinity. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (pp. 975-985). [**[Paper]**](https://doi.org/10.1145/3447548.3467311) [**[Code]**](https://github.com/agave233/SIGN) 

0. **[APMNet]** Shen, H., Zhang, Y., Zheng, C., Wang, B., & Chen, P. (2021). A Cascade graph convolutional network for predicting protein–ligand binding affinity. International journal of molecular sciences, 22(8), 4023. [**[Paper]**](https://doi.org/10.3390/ijms22084023)

0. **[Affinity-by-GNN]** Nikolaienko, T., Gurbych, O., & Druchok, M. (2022). Complex machine learning model needs complex testing: Examining predictability of molecular binding affinity by a graph neural network. Journal of Computational Chemistry, 43(10), 728-739.  [**[Paper]**](https://doi.org/10.1002/jcc.26831) [**[Code]**](https://github.com/SoftServeInc/affinity-by-GNN) 

0. **[GINet]** Réau, M., Renaud, N., Xue, L. C., & Bonvin, A. M. (2023). DeepRank-GNN: a graph neural network framework to learn patterns in protein–protein interfaces. Bioinformatics, 39(1), btac759. [**[Paper]**](https://doi.org/10.1093/bioinformatics/btac759) [**[Code]**](https://github.com/DeepRank/DeepRankGNN) 

0. **[GraphBAR]** Son, J., & Kim, D. (2021). Development of a graph convolutional neural network model for efficient prediction of protein-ligand binding affinities. PloS one, 16(4), e0249404. [**[Paper]**](https://doi.org/10.1371/journal.pone.0249404) [**[Code]**](http://github.com/jtson82/graphbar) 

0. **[EGNA]** Xia, C., Feng, S. H., Xia, Y., Pan, X., & Shen, H. B. (2023). Leveraging scaffold information to predict protein–ligand binding affinity with an empirical graph neural network. Briefings in Bioinformatics, 24(1), bbac603. [**[Paper]**](https://doi.org/10.1093/bib/bbac603) [**[Code]**]( https://github.com/chunqiux/EGNA) 

0. **[GIGN]** Yang, Z., Zhong, W., Lv, Q., Dong, T., & Yu-Chian Chen, C. (2023). Geometric Interaction Graph Neural Network for Predicting Protein–Ligand Binding Affinities from 3D Structures (GIGN). The Journal of Physical Chemistry Letters, 14(8), 2020-2033. [**[Paper]**](https://doi.org/10.1021/acs.jpclett.2c03906) [**[Code]**](https://github.com/guaguabujianle/GIGN) 

0. **[GraphSite]**  [**[Paper]**](https://doi.org/10.3390/biom12081053) [**[Code]**](https://github.com/shiwentao00/Graphsiteclassifier) 

0. **[IGN]** Jiang, D., Hsieh, C. Y., Wu, Z., Kang, Y., Wang, J., Wang, E., ... & Hou, T. (2021). InteractionGraphNet: a novel and efficient deep graph representation learning framework for accurate protein–ligand interaction predictions. Journal of medicinal chemistry, 64(24), 18209-18232. [**[Paper]**](https://doi.org/10.1021/acs.jmedchem.1c01830) [**[Code]**](https://github.com/zjujdj/InteractionGraphNet/tree/master) 

0. **[InteractionNet]** Cho, H., Lee, E. K., & Choi, I. S. (2020). Layer-wise relevance propagation of InteractionNet explains protein–ligand interactions at the atom level. Scientific reports, 10(1), 21155. [**[Paper]**](https://doi.org/10.1038/s41598-020-78169-6) [**[Code]**](https://github.com/blackmints/InteractionNet) 

0. **[MP-GNN]** Li, X. S., Liu, X., Lu, L., Hua, X. S., Chi, Y., & Xia, K. (2022). Multiphysical graph neural network (MP-GNN) for COVID-19 drug design. Briefings in Bioinformatics, 23(4), bbac231. [**[Paper]**](https://doi.org/10.1093/bib/bbac231) [**[Code]**]( https://github.com/Alibaba-DAMO-DrugAI/MGNN) 

0. **[PSG-BAR]** Pandey, M., Radaeva, M., Mslati, H., Garland, O., Fernandez, M., Ester, M., & Cherkasov, A. (2022). Ligand binding prediction using protein structure graphs and residual graph attention networks. Molecules, 27(16), 5114. [**[Paper]**](https://doi.org/10.3390/molecules27165114) [**[Code]**]( https://github.com/diamondspark/PSG-BAR) 

0. **[QATEN]** Zhang, P., Xia, C., & Shen, H. B. (2023). High-accuracy protein model quality assessment using attention graph neural networks. Briefings in Bioinformatics, 24(2), bbac614. [**[Paper]**](https://doi.org/10.1093/bib/bbac614) [**[Server]**]( http://www.csbio.sjtu.edu.cn/bioinf/QATEN/) [**[Code]**](https://github.com/CQ-zhang-2016/QATEN) 

0. **[SGPPI]** Huang, Y., Wuchty, S., Zhou, Y., & Zhang, Z. (2023). SGPPI: structure-aware prediction of protein–protein interactions in rigorous conditions with graph convolutional network. Briefings in Bioinformatics, 24(2), bbad020. [**[Paper]**](https://doi.org/10.1093/bib/bbad020) [**[Code]**]( https://github.com/emerson106/SGPPI) 

### Others
1. **[HGNN]** Hou, W., Wang, Y., Zhao, Z., Cong, Y., Pang, W., & Tian, Y. (2023). Hierarchical graph neural network with subgraph perturbations for key gene cluster discovery in cancer staging. Complex & Intelligent Systems, 1-18. [**[Paper]**](https://doi.org/10.1007/s40747-023-01068-6)

0. **[PathGNN]** Liang, B., Gong, H., Lu, L., & Xu, J. (2022). Risk stratification and pathway analysis based on graph neural network and interpretable algorithm. BMC bioinformatics, 23(1), 394. [**[Paper]**](https://doi.org/10.1186/s12859-022-04950-1) [**[Code]**](https://github.com/BioAI-kits/PathGNN) 

0.  Zhuo, L., Chen, Y., Song, B., Liu, Y., & Su, Y. (2022). A model for predicting ncRNA–protein interactions based on graph neural networks and community detection. Methods, 207, 74-80. [**[Paper]**](https://doi.org/10.1016/j.ymeth.2022.09.001) 

0.  Rhee, S., Seo, S., & Kim, S. (2018, July). Hybrid approach of relation network and localized graph convolutional filtering for breast cancer subtype classification. In Proceedings of the 27th International Joint Conference on Artificial Intelligence (pp. 3527-3534). [**[Paper]**](https://doi.org/10.24963/ijcai.2018/490)

0.  Li, B., & Nabavi, S. (2024). A multimodal graph neural network framework for cancer molecular subtype classification. BMC bioinformatics, 25(1), 27. [**[Paper]**](https://doi.org/10.1186/s12859-023-05622-4)

## Update
