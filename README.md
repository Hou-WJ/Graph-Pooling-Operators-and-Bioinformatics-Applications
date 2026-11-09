# Graph Pooling Operators and Bioinformatics Applications 

# Graph Pooling in Graph Neural Networks: Methods and Their Applications in Omics

## Paper [**[Paper]**](https://doi.org/10.1007/s10462-024-10918-9) [**[TOC]**](#table-of-contents)
## Cite
If this repository is useful for your research, please consider citing our paper.
```
@article{wangGraphPoolingGraph2024,
  title = {Graph Pooling in Graph Neural Networks: Methods and Their Applications in Omics Studies},
  author = {Wang, Yan and Hou, Wenju and Sheng, Nan and Zhao, Ziqi and Liu, Jialin and Huang, Lan and Wang, Juexin},
  year = {2024},
  month = sep,
  journal = {Artificial Intelligence Review},
  volume = {57},
  number = {11},
  pages = {294},
  issn = {1573-7462},
  doi = {10.1007/s10462-024-10918-9},
  urldate = {2024-09-18},
  abstract = {Graph neural networks (GNNs) process the graph-structured data using neural networks and have proven successful in various graph processing tasks. Currently, graph pooling operators have emerged as crucial components that bridge the gap between node representation learning and diverse graph-level tasks by transforming node representations into graph representations. Given the rapid growth and widespread adoption of graph pooling, this review aims to summarize the existing graph pooling operators for GNNs and their representative applications in omics. Specifically, we first present a comprehensive taxonomy of existing graph pooling algorithms, expanding the categorization for both global and hierarchical pooling operators, and for the first time reviewing the inverse operation of graph pooling, named unpooling. Next, we describe the general evaluation framework for graph pooling operators, encompassing three fundamental aspects: experimental setup, ablation analysis, and model interpretation. We also discuss open issues that significantly influence the design of graph pooling operators, including complexity, connectivity, adaptability, additional loss, and attention mechanisms. Finally, we summarize bioinformatics applications of graph pooling operators in omics, including graphs of gene interaction, medical images, and protein structures for drug discovery and disease diagnosis. Furthermore, we showcase the impact of graph pooling operators on research in specific real-world domains, with a focus on prediction performance and model interpretability. This review provides methodological insights in machine learning based graph modeling and related omics research, as well as an ongoing resource by gathering related papers and code in a dedicated GitHub repository (https://github.com/Hou-WJ/Graph-Pooling-Operators-and-Bioinformatics-Applications).}
}

```
## News

- **[2026-11]** Our new paper **“Rebuilding to remember: reconstruction-driven regularization for information-preserving in clustering graph pooling”** is now available. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0031320326020984) [**[Code]**](https://github.com/Hou-WJ/REGPool) 

<details>
<summary>BibTeX</summary>

```
@article{houRebuildingRememberReconstructiondriven2027,
  title = {Rebuilding to Remember: Reconstruction-Driven Regularization for Information-Preserving in Clustering Graph Pooling},
  shorttitle = {Rebuilding to Remember},
  author = {Hou, Wenju and Hong, Jie and Liu, Jialin and Sun, Qixing and Wang, Xinfei and Zhao, Ziqi and Xie, Xuping and Sheng, Nan and Zuo, Chunman and Wang, Yan},
  year = 2027,
  month = mar,
  journal = {Pattern Recognition},
  volume = {183},
  pages = {115135},
  doi = {10.1016/j.patcog.2026.115135}
}
```

</details>

## Table of Contents
* [Overview](#Overview) 
* [Global Pooling](#global-pooling) 
* [Hierarchical Pooling](#hierarchical-pooling) 
  - [Clustering Pooling](#clustering-pooling)
  - [Selection Pooling](#selection-pooling)
  - [Edge Pooling](#edge-pooling)
  - [Hybrid Pooling](#hybrid-pooling)
* [Graph Unpooling](#graph-unpooling)
* [Graph Pooling for Complex Graphs](#graph-pooling-for-complex-graphs)
  - [Hypergraph](#hypergraph)
  - [Heterophilic Graph](#heterophilic-graph)
  - [Dynamic Graph](#dynamic-graph)
  - [Heterogeneous Graph](#heterogeneous-graph)
* [Benchmark Datasets](#benchmark-datasets)
* [Libraries](#libraries) 
* [Evaluation, Review and Analysis](#evaluation-review-and-analysis)
* [Applications in Multi-omics Data](#applications-in-bioinformatics)
  - [Genomics](#genomics)
  - [Radiomics](#radiomics)
  - [Proteomics](#proteomics)
  - [Metabolomics](#metabolomics)
  - [Multi-omics](#multi-omics)
* [Applications in Biological Computing and Molecular Modeling](#applications-in-biological-computing-and-molecular-modeling)
  - [Molecular Modeling](#molecular-modeling)
  - [Biological Computing](#biological-computing)
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

30. **[SSVP]** Liu, Y., & Du, B. (2024). Frequency Domain-Oriented Complex Graph Neural Networks for Graph Classification. IEEE Transactions on Neural Networks and Learning Systems. [**[Paper]**](https://doi.org/10.1109/TNNLS.2024.3351762) [**[Code]**](https://github.com/Liuyouf/ComplexGNN) 

31. **[G-Prompt]** Duan, Y., Liu, J., Chen, S., Chen, L., & Wu, J. (2024). G-Prompt: Graphon-based Prompt Tuning for graph classification. Information Processing & Management, 61(3), 103639. [**[Paper]**](10.1016/j.ipm.2023.103639) 

32. **[HybridPooling]** Dong, Z., Hu, Q., Zhang, Z., Guo, Y., Cordy, M., Papadakis, M., ... & Zhao, J. (2024). On the effectiveness of hybrid pooling in mixup-based graph learning for language processing. Journal of Systems and Software, 216, 112139. [**[Paper]**](https://doi.org/10.1016/j.jss.2024.112139) [**[Code]**](https://github.com/zemingd/HybridPool4Mixup) 

33. **[AAGR]** Zhong, W., Mu, X., Zhang, J., & Li, D. (2025). AAGR: Graph-level Anomaly Detection with Anomaly-aware Graph Readout. Expert Systems with Applications, 128672. [**[Paper]**](https://doi.org/10.1016/j.eswa.2025.128672) [**[Code]**](https://github.com/zemingd/HybridPool4Mixup) 

33. Binkowski, J., Sawczyn, A., Janiak, D., Bielak, P., & Kajdanowicz, T. (2023, June). Graph-level representations using ensemble-based readout functions. In International Conference on Computational Science (pp. 393-405). Cham: Springer Nature Switzerland. [**[Paper]**](10.1007/978-3-031-35995-8_28) 

33. **[nmrGCL]** Sun, J., Chen, R., Li, J., Ding, Y., Wu, C., Liu, Z., & Yan, J. (2025). Understanding and mitigating dimensional collapse of graph contrastive learning: A non-maximum removal approach. Neural Networks, 181, 106652. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0893608024005768) 

33. **[Atten-Mixer]** Zhang, P., Guo, J., Li, C., Xie, Y., Kim, J. B., Zhang, Y., ... & Kim, S. (2023, February). Efficiently leveraging multi-level user intent for session-based recommendation via atten-mixer network. In Proceedings of the sixteenth ACM international conference on web search and data mining (pp. 168-176). [**[Paper]**](https://doi.org/10.1016/j.eswa.2025.128672) [**[Code]**](https://github.com/Peiyance/Atten-Mixer-torch) 

33. Duan, Y., Liu, X., Lynden, S., Matono, A., & Ma, Q. (2025, September). How Useful Is Graph Pooling for Node-Level Tasks?. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 92-107). Cham: Springer Nature Switzerland. [**[Paper]**](https://link.springer.com/10.1007/978-3-032-06066-2_6)

33. **[HISTOGRAPH]** Galron, Y., Sinai, H., Maron, H., & Eliasof, M. (2026). Learning from Historical Activations in Graph Neural Networks. arXiv preprint arXiv:2601.01123. [**[Paper]**](https://openreview.net/forum?id=8SnAGYf2wM) 

33. **[HDGNet]** Bu, L., Wang, Y., Ma, Q., Hou, Z., Wang, R., & Bu, F. (2025). Deep hierarchical learning on point clouds in feature space. Neurocomputing, 630, 129647. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0925231225003194)

33. **[PAPooling]** Wang, J., Xu, T., Song, L., Ding, L., Li, H., Jiang, P., ... & Li, J. (2025). PAPooling: Graph-based Position Adaptive Aggregation of Local Geometry in Point Clouds. ACM Transactions on Multimedia Computing, Communications and Applications, 21(4), 1-18. [**[Paper]**](https://dl.acm.org/doi/10.1145/3718742) [**[Code]**](https://github.com/Roywangj/PAPooling/) 

33. **[BAPool]** Sun, W. X., & Xue, H. (2021). Learning graph-level representation from local-structural distribution with Graph Neural Networks. Knowledge-Based Systems, 230, 107383. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0950705121006456)

33. **[MATPOOL]** Zhu, Z., & Xu, Y. Matrix-pattern-oriented Pooling for Graph-level Data Representation and Prediction. Available at SSRN 7113638. [**[Paper]**](https://papers.ssrn.com/abstract=7113638)

33. **[BotPool]** Xue, X., Zhang, R., Liu, D., Liu, B., & Cao, J. (2026). Behavior-Driven Detection of Social Bot Groups Through Coordination Patterns and Stance Consistency. IEEE Transactions on Computational Social Systems. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11568560)

33. **[MABLE]** Shulman, Y., Akbarpour, S., & Muir, J. B. (2026). MABLE: Masked Autoencoding with Bi-Lipschitz Decoding for Embeddings and Graph Metric Learning. arXiv preprint arXiv:2607.02990. [**[Paper]**](http://arxiv.org/abs/2607.02990)



## Hierarchical Pooling
### Clustering Pooling
1. **[CapsGNN]** Xinyi Z, Chen L (2018) Capsule graph neural network. In: International Conference on Learning Representations  [**[Paper]**](https://openreview.net/forum?id=Byl8BnRcYm) [**[Code]**](https://github.com/benedekrozemberczki/CapsGNN) 

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

0. **[SubGattPool]** Bandyopadhyay S, Aggarwal M, Murty MN (2020) Robust Hierarchical Graph Classification with Subgraph Attention. arXiv preprint arXiv:200710908 [**[Paper]**](https://arxiv.org/abs/2007.10908)

0. **[SUGAR]** Sun Q, Li J, Peng H, et al (2021) SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism. In: Proceedings of the Web Conference 2021. pp 2081–2091 [**[Paper]**](https://doi.org/10.1145/3442381.3449822) [**[Code]**](https://github.com/RingBDStack/SUGAR) 

0. **[LCP]** Su Z, Hu Z, Li Y (2021) Hierarchical Graph Representation Learning with Local Capsule Pooling. In: ACM Multimedia Asia. pp 1–7 [**[Paper]**](https://doi.org/10.1145/3469877.3495645) [**[Code]**](https://github.com/imSeaton/LocalCapsulePoolingNetwork) 

0. **[GSAPool]** Yu H, Yuan J, Cheng H, et al (2021) GSAPool: Gated Structure Aware Pooling for Graph Representation Learning. In: 2021 International Joint Conference on Neural Networks (IJCNN). IEEE, pp 1–8 [**[Paper]**](https://doi.org/10.1109/IJCNN52387.2021.9534320) 

0. **[FPool]** Pham H Van, Thanh DH, Moore P (2021) Hierarchical Pooling in Graph Neural Networks to Enhance Classification Performance in Large Datasets. Sensors 21:6070.  [**[Paper]**](https://doi.org/10.3390/s21186070) 

0. **[HGCN]** Yang J, Zhao P, Rong Y, et al (2021a) Hierarchical Graph Capsule Network. Proceedings of the AAAI Conference on Artificial Intelligence 35:10603–10611.  [**[Paper]**](https://doi.org/10.1609/aaai.v35i12.17268) [**[Code]**](https://github.com/uta-smile/HGCN) 

0. **[HAP]** Liu N, Jian S, Li D, et al (2021) Hierarchical Adaptive Pooling by Capturing High-order Dependency for Graph Representation Learning. IEEE Trans Knowl Data Eng 3952-3965. [**[Paper]**](https://doi.org/10.1109/TKDE.2021.3133646)
0. **[DMP/MPR]** Bodnar C, Cangea C, Liò P (2021) Deep Graph Mapper: Seeing Graphs Through the Neural Lens. Front Big Data 4:680535.  [**[Paper]**](https://doi.org/10.3389/fdata.2021.680535) [**[Code]**](https://github.com/crisbodnar/dgm) 

0. **[HIBPool]** Roy KK, Roy A, Mahbubur Rahman AKM, et al (2021) Structure-Aware Hierarchical Graph Pooling using Information Bottleneck. In: 2021 International Joint Conference on Neural Networks (IJCNN). IEEE, pp 1–8 [**[Paper]**](https://doi.org/10.1109/IJCNN52387.2021.9533778) [**[Code]**](https://github.com/forkkr/HIBPool) 

0. **[KPlexPool]** Bacciu D, Conte A, Grossi R, et al (2021) K-plex cover pooling for graph neural networks. Data Min Knowl Discov 35:2200–2220. [**[Paper]**](https://doi.org/10.1007/s10618-021-00779-z) [**[Code]**](https://github.com/flandolfi/kplex-pool/) 

0. **[CommPOOL]** Tang H, Ma G, He L, et al (2021) CommPOOL: An interpretable graph pooling framework for hierarchical graph representation learning. Neural Networks 143:669–677.  [**[Paper]**](https://doi.org/10.1016/j.neunet.2021.07.028) 

0. **[HGP-SACA]** Li ZP, Su HL, Zhu XB, et al (2022) Hierarchical Graph Pooling With Self-Adaptive Cluster Aggregation. IEEE Trans Cogn Dev Syst 14:1198–1207.  [**[Paper]**](https://doi.org/10.1109/TCDS.2021.3100883)

0. **[SEP]** Wu J, Chen X, Xu K, Li S (2022) Structural Entropy Guided Graph Hierarchical Pooling. In: Proceedings of the 39th International Conference on Machine Learning. PMLR, pp 24017–24030 [**[Paper]**](https://proceedings.mlr.press/v162/wu22b.html) [**[Code]**](https://github.com/Wu-Junran/SEP) 

0. **[SMIP]** Liu N, Jian S, Li D, Xu H (2022) Unsupervised Hierarchical Graph Pooling via Substructure-Sensitive Mutual Information Maximization. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management. pp 1299–1308 [**[Paper]**](https://doi.org/10.1145/3511808.3557485)

0. **[HoscPool]** Duval A, Malliaros F (2022) Higher-Order Clustering and Pooling for Graph Neural Networks. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management. pp 426–435 [**[Paper]**](https://doi.org/10.1145/3511808.3557353) [**[Code]**](https://github.com/AlexDuvalinho/HoscPool) 

0. **[AdamGNN]** Zhong Z, Li C-T, Pang J (2022) Multi-grained Semantics-aware Graph Neural Networks. IEEE Trans Knowl Data Eng 7251-7262.  [**[Paper]**](https://doi.org/10.1109/TKDE.2022.3195004) [**[Code]**](https://github.com/zhiqiangzhongddu/AdamGNN) 

0. **[DMoN]** Tsitsulin, A., Palowitch, J., Perozzi, B., & Müller, E. (2023). Graph clustering with graph neural networks. Journal of Machine Learning Research, 24(127), 1-21. [**[Paper]**](https://www.jmlr.org/papers/v24/20-998.html) [**[Code]**](https://github.com/google-research/google-research/tree/master/graph_embedding/dmon)

0. **[MATHNET]** Zheng X, Zhou B, Li M, et al (2020) MathNet: Haar-Like Wavelet Multiresolution-Analysis for Graph Representation and Learning. arXiv preprint arXiv:200711202 [**[Paper]**](https://doi.org/10.1016/j.knosys.2023.110609) 

0. **[GTV]** Hansen, J. B., & Bianchi, F. M. (2023, July). Total variation graph neural networks. In International Conference on Machine Learning (pp. 12445-12468). PMLR. [**[Paper]**](https://proceedings.mlr.press/v202/hansen23a.html) [**[Code]**](https://github.com/FilippoMB/Total-variation-graph-neural-networks)

0. **[k-MIS]** Bacciu, D., Conte, A., & Landolfi, F. (2023, June). Generalizing downsampling from regular data to graphs. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 6718-6727). [**[Paper]**](https://doi.org/10.1609/aaai.v37i6.25824) 

0. **[Topology-Invariant Pooling]** Ying, C., Zhao, X., & Yu, T. (2024). Boosting Graph Pooling with Persistent Homology. arXiv preprint arXiv:2402.16346. [**[Paper]**](https://doi.org/10.48550/arXiv.2402.16346) 

0. **[GPN]** Song, Y., Huang, S., Wang, X., Zhou, C., & Lin, Z. (2023, October). Graph Parsing Networks. In The Twelfth International Conference on Learning Representations. [**[Paper]**](https://openreview.net/forum?id=hv3SklibkL) [**[Code]**](https://github.com/LUMIA-Group/GraphParsingNetworks)

0. **[SSHPool]** Xu, Z., Cui, L., Wang, Y., Du, H., Bai, L., & Hancock, E. R. (2024). SSHPool: The Separated Subgraph-based Hierarchical Pooling. arXiv preprint arXiv:2403.16133. [**[Paper]**](https://doi.org/10.48550/arXiv.2403.16133) 

0. **[MGPool]** Deng, S., Yang, G., Yang, Y., Gong, Z., Chen, C., Chen, X., & Hao, Z. (2024). Module-based graph pooling for graph classification. Pattern Recognition, 154, 110606. [**[Paper]**](https://doi.org/10.1016/j.patcog.2024.110606) [**[Code]**](https://github.com/SubaiDeng/MGPool)

0. **[HC-GAE]** Bai, L., Xu, Z., Cui, L., Li, M., Wang, Y., & Hancock, E. (2024). HC-GAE: The Hierarchical Cluster-based Graph Auto-Encoder for Graph Representation Learning. Advances in Neural Information Processing Systems, 37, 127968-127986. [**[Paper]**](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e6e706454d72c18582b9c1ff70b11f7d-Abstract-Conference.html) [**[Code]**](https://github.com/JonathanGXu/HC-GAE)

0. **[SAGoG]** Wang, S., Zhang, Y., Lin, X., Hu, Y., Huang, Q., & Yin, B. (2025). SAGoG: Similarity-Aware Graph of Graphs Neural Networks for Multivariate Time Series Classification. IEEE Transactions on Knowledge and Data Engineering. [**[Paper]**](10.1109/TKDE.2025.3572216)

0. **[GSpect]** Zhang, X., Yang, W., Feng, J., Dai, B., Bu, T., & Lu, X. (2024). GSpect: Spectral filtering for cross-scale graph classification. IEEE Transactions on Network Science and Engineering. [**[Paper]**](10.1109/TNSE.2024.3513456) [**[Code]**](https://github.com/XiaoyuZhang001/GSpect)

0. **[RTPool]** Zhang, Y., Li, L., & Xia, K. Rhomboid Tiling for Geometric Graph Deep Learning. In Forty-second International Conference on Machine Learning. [**[Paper]**](https://openreview.net/forum?id=lTZEAqL1R1)

0. **[HireGC]** Zhao, S., Xu, C., Du, Z., Zhang, Y., Duan, Z., & Chen, J. (2025). HireGC: Hierarchical inductive network representation learning via graph coarsening. Knowledge-Based Systems, 311, 113058. [**[Paper]**](https://doi.org/10.1016/j.knosys.2025.113058)

0. **[SSGNN]** Zhang, Z., & Zhao, L. (2024). Self-similar graph neural network for hierarchical graph learning. In Proceedings of the 2024 SIAM International Conference on Data Mining (SDM) (pp. 28-36). Society for Industrial and Applied Mathematics. [**[Paper]**](https://doi.org/10.1137/1.9781611978032.4) [**[Code]**](https://github.com/rollingstonezz/SDM24_self_similar_graph)

0. **[Quasi-cliquepool]** Ali, W., Vascon, S., Stadelmann, T., & Pelillo, M. (2023, March). Quasi-cliquepool: Hierarchical graph pooling for graph classification. In Proceedings of the 38th ACM/SIGAPP Symposium on Applied Computing (pp. 544-552). [**[Paper]**](https://doi.org/10.1145/3555776.3578600)

0. **[DHTAGK]** Qin, X., Bai, L., Cui, L., Li, M., Lyu, Z., Du, H., & Hancock, E. R. (2025, August). DHTAGK: Deep Hierarchical Transitive-Aligned Graph Kernels for Graph Classification. In IJCAI (pp. 3254-3262). [**[Paper]**](https://www.ijcai.org/proceedings/2025/0362.pdf) [**[Code]**](https://github.com/Xiaoqin0421/DHTAGK)

0. **[IMPO]** Ragno, A., & Capobianco, R. (2025, March). Impo: Interpretable memory-based prototypical pooling. In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining (pp. 625-632). [**[Paper]**](https://dl.acm.org/doi/10.1145/3701551.3703543) [**[Code]**](https://github.com/KRLGroup/IMPO)

0. **[MDL-Pool]** von Pichowski, J., Blöcker, C., & Scholtes, I. (2024). MDL-Pool: Adaptive Multilevel Graph Pooling Based on Minimum Description Length. arXiv preprint arXiv:2409.10263. [**[Paper]**](https://openreview.net/forum?id=xUM038Z7US) [**[Code]**](https://github.com/rollingstonezz/SDM24_self_similar_graph)

0. **[LACORE]** Li, R., Li, Q., & Li, Y. LaCore: Laplacian Cohesive Subgraphs for Graph Representation Learning. [**[Paper]**](https://openreview.net/forum?id=AH3yWSAAfA) [**[Code]**](https://github.com/divelab/DIG)

0. **[BiFormer]** Zhang, Q., Sun, Y., Wang, S., Gao, J., & Yin, B. (2025). BiFormer: A Bipartite-stream Information Fusion framework for large-scale graph representation learning. Neural Networks, 191, 107792. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0893608025006720) [**[Code]**](https://github.com/QiZhang3709/BiFormer)

0. **[GPPool]** Zhang, Q., Sun, Y., Wang, S., Gao, J., Hu, Y., & Yin, B. (2024). Training large-scale graph neural networks via graph partial pooling. IEEE Transactions on Big Data, 11(1), 221-233. [**[Paper]**](https://ieeexplore.ieee.org/document/10535224) [**[Code]**](https://github.com/TracReggie/GPPool)

0. **[CMRGP]** Wang, Z., Yang, L., Chen, T., & Long, J. (2025). Clustering matrix regularization guided hierarchical graph pooling. Knowledge-Based Systems, 311, 113108. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0950705125001558) 

0. **[HGBG-Pool]** Ni, J., Chen, L., Tang, G., Yu, N., Tang, W., Yi, X., & Xia, S. (2026). Hierarchical granular-ball graph pooling via feature–structure coupling. Information Sciences, 123501. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0020025526004329) [**[Code]**](https://anonymous.4open.science/r/Hgbg-pool-2ED4/)

0. **[DHTAGK]** Qin, X., Bai, L., Cui, L., Li, M., Lyu, Z., Du, H., & Hancock, E. R. (2025, August). DHTAGK: Deep Hierarchical Transitive-Aligned Graph Kernels for Graph Classification. In IJCAI (pp. 3254-3262). [**[Paper]**](https://www.ijcai.org/proceedings/2025/362) [**[Code]**](https://github.com/Xiaoqin0421/DHTAGK)

0. **[DGLU]** Su, Y., Dai, C., Sun, Y., Yang, R., Yu, A., & Zhong, Y. (2026). Beyond static graphs: a scale-variant dynamic graph-based framework integrating spatial and hierarchical relationships for urban land-use mapping. GIScience & Remote Sensing, 63(1), 2705683. [**[Paper]**](https://doi.org/10.1080/15481603.2026.2705683) 

0. **[GMPOOL]** Ko, S. M., Cho, S., Jeong, D. W., Han, S., Lee, M., & Lee, H. (2023, June). Grouping matrix based graph pooling with adaptive number of clusters. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 7, pp. 8334-8342). [**[Paper]**](https://ojs.aaai.org/index.php/AAAI/article/view/26005) 

0. **[SparsePool]** Li, Z., Zhai, X., Hu, H., Liang, J., & Hu, G. (2026). SparsePool: A Graph Pooling Framework via Sparse Representation for Graph Classification. Sensors, 26(9), 2627. [**[Paper]**](https://www.mdpi.com/1424-8220/26/9/2627) 

0. **[ABDPool]** Liu, Y., Cui, L., Wang, Y., & Bai, L. (2022, August). Abdpool: Attention-based differentiable pooling. In 2022 26th International Conference on Pattern Recognition (ICPR) (pp. 3021-3026). IEEE. [**[Paper]**](https://ieeexplore.ieee.org/document/9956378/authors) 

0. **[ICEPool]** Yang, M. (2025). ICEPool: Enhancing Graph Pooling Networks with Inter-cluster Connectivity. arXiv preprint arXiv:2510.03987. [**[Paper]**](http://arxiv.org/abs/2510.03987) 

0. Azad, P., Heidari, A., Akcora, C. G., Khonsari, A., & Rastegar, S. H. (2026). A unified graph neural network-based approach for few-shot learning with task nodes and DiffPool abstraction. Neurocomputing, 133003. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0925231226004005) 

0. **[SumCL]** Lei, M., Zhang, C., Jia, H., & Ji, J. (2026). Subgraph Neural Networks With Multi-Scale Contrastive Learning for Graph Classification. IEEE Transactions on Emerging Topics in Computational Intelligence. [**[Paper]**](https://ieeexplore.ieee.org/document/9956378/authors) 

0. **[GEGNN]** Qiu, L., Xu, Y., Li, H., Zhang, C. J., Zhou, A., Cheng, P., ... & Li, Q. (2026). Towards A Generalizable and Expressive Graph Neural Network for Graph-Level Tasks with Theoretical Guarantees. The VLDB Journal, 35(4), 30. [**[Paper]**](https://doi.org/10.1007/s00778-026-00978-4) 

0. **[DCPGCN]** Zheng, L., Wu, F., Li, C., Estupinan, E., & Qin, Y. (2026). Dynamic curvature pooling graph convolutional network to fuse multi-sensor signals for remaining useful life prediction. Advanced Engineering Informatics, 74, 104693. [**[Paper]**](https://linkinghub.elsevier.com/retrieve/pii/S147403462600385X) 

0. **[REGPool]** Hou, W., Hong, J., Liu, J., Sun, Q., Wang, X., Zhao, Z., ... & Wang, Y. (2026). Rebuilding to remember: Reconstruction-driven regularization for information-preserving in clustering graph pooling. Pattern Recognition, 115135. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0031320326020984) [**[Code]**](https://github.com/Hou-WJ/REGPool)


### Selection Pooling
1. **[AttPool]** Huang J, Li Z, Li N, et al (2019) Attpool: Towards hierarchical feature representation in graph convolutional networks via attention mechanism. In: Proceedings of the IEEE International Conference on Computer Vision. pp 6479–6488 [**[Paper]**](https://openaccess.thecvf.com/content_ICCV_2019/html/Huang_AttPool_Towards_Hierarchical_Feature_Representation_in_Graph_Convolutional_Networks_via_ICCV_2019_paper.html) [**[Code]**](https://github.com/hjjpku/Attention_in_Graph) 

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

0. **[MID]** Liu C, Zhan Y, Yu B, et al (2023) On exploring node-feature and graph-structure diversities for node drop graph pooling. Neural Networks 167:559–571. [**[Paper]**](https://doi.org/10.1016/j.neunet.2023.08.046) [**[Code]**](https://github.com/whuchuang/mid) 

0. **[Element Ranking Pooling]** Guo, J., Guo, S., Zhou, Q., Liu, Z., Lu, X., & Huo, F. (2023, June). Graph knows unknowns: Reformulate zero-shot learning as sample-level graph recognition. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 7775-7783). [**[Paper]**](https://doi.org/10.1609/aaai.v37i6.25942) 

1. **[pqPool]** Amouzad, A., Dehghanian, Z., Saravani, S., Amirmazlaghani, M., & Roshanfekr, B. (2024). Graph isomorphism u-net. Expert Systems with Applications, 236, 121280. [**[Paper]**](https://doi.org/10.1016/j.eswa.2023.121280)

0. **[NIAPool]** Sun, C., Huang, F., & Peng, J. (2022, May). Node information awareness pooling for graph representation learning. In Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp. 182-193). Cham: Springer International Publishing. [**[Paper]**](10.1007/978-3-031-05933-9_15) 

0. **[MVPPool]** Huang, J., & Wang, Y. Y. (2025). Multi-view parallel graph pooling. International Journal of Data Science and Analytics, 19(2), 257-267. [**[Paper]**](10.1007/s41060-023-00476-8)

0. **[HGLA-Pool]** Ali, W., Vascon, S., Stadelmann, T., & Pelillo, M. (2024). Hierarchical glocal attention pooling for graph classification. Pattern Recognition Letters, 186, 71-77. [**[Paper]**](https://doi.org/10.1016/j.patrec.2024.09.009)

0. **[MGR]** Huang, W., Chen, Y., Jiang, X., Gao, C., Zhang, T., Chen, Q., & Wang, Y. (2025, April). Mitigating Pervasive Modality Absence Through Multimodal Generalization and Refinement. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 25, pp. 26796-26804). [**[Paper]**](10.1609/aaai.v39i25.34883)

0. **[GrePool]** Liu, C., Yu, W., Gao, K., Ma, X., Zhan, Y., Wu, J., ... & Du, B. (2025). Graph explicit pooling for graph-level representation learning. Neural Networks, 181, 106790. [**[Paper]**](https://doi.org/10.1016/j.neunet.2024.106790) [**[Code]**](https://github.com/LiuChuang0059/GrePool) 

0. **[Fair Graph Pooling]** Wang, Z., Chu, Z., Doan, T. V., Wang, S., Wu, Y., Palade, V., & Zhang, W. (2025, April). Fair graph u-net: A fair graph learning framework integrating group and individual awareness. In proceedings of the AAAI conference on artificial intelligence (Vol. 39, No. 27, pp. 28485-28493). [**[Paper]**](10.1609/aaai.v39i27.35071)

0. **[GDMGP]** Wang, Z., & Fan, H. (2024). Multi-view graph pooling with coarsened graph disentanglement. Neural Networks, 174, 106221. [**[Paper]**](10.1016/j.neunet.2024.106221) 

0. **[MSSGC]** Liu, W., & Guo, Q. (2025). Graph Classification via Hierarchical Structure Learning with Multi-Scale Convolution and Pooling. Engineering Letters, 33(5).

0. **[GraphADT]** Ma, X., Fu, X., Wang, T., Zhuo, L., & Zou, Q. (2024). GraphADT: empowering interpretable predictions of acute dermal toxicity with multi-view graph pooling and structure remapping. Bioinformatics, 40(7), btae438. [**[Paper]**](https://doi.org/10.1093/bioinformatics/btae438) [**[Code]**](https://github.com/mxqmxqmxq/GraphADT.git) 

0. **[Wit-TopoPool]** Chen, Y., & Gel, Y. R. (2023, June). Topological pooling on graphs. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 7096-7103). [**[Paper]**](10.1609/aaai.v37i6.25866) [**[Code]**](https://github.com/topologicalpooling/TopologicalPool.git) 

0. **[DOTIN]** Zhong, C., Zhang, S., Zhu, F., Zhao, R., Yang, X., & Yan, J. (2025, April). Graph Pooling via Dropping Task-Irrelevant Nodes. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE. [**[Paper]**](10.1109/ICASSP49660.2025.10889769) [**[Code]**](https://github.com/Sherrylone/DOTIN) 

0. **[DMIPool]** Zhao, Q., Zhang, H., He, M., Li, W., Kang, C., & Han, M. (2023). Graph pooling via dual-view multi-level infomax. Knowledge-Based Systems, 260, 110089. [**[Paper]**](https://doi.org/10.1016/j.knosys.2022.110089)

0. **[GPIB]** Long, J., Wang, Z., Chen, T., & Yang, L. (2025). Improving hierarchical graph pooling with information bottleneck. Journal of King Saud University Computer and Information Sciences, 37(3), 36. [**[Paper]**](https://doi.org/10.1007/s44443-025-00046-x)

0. **[SSPool]** Li, Z. P., Su, H. L., Zhu, X. B., Gribova, V., Filaretov, V. F., & Huang, D. S. (2023). SSPool: a simple siamese framework for graph infomax pooling. IEEE Transactions on Network Science and Engineering, 11(1), 463-470. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/10207695)

0. Stanovic, S., Gaüzère, B., & Brun, L. (2025). Graph neural networks with maximal independent set-based pooling: Mitigating over-smoothing and over-squashing. Pattern Recognition Letters, 187, 14-20. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0167865524003106)

0. **[MIESPool]** Stanovic, S., Gaüzère, B., & Brun, L. (2023, August). Maximal independent sets for pooling in graph neural networks. In International Workshop on Graph-Based Representations in Pattern Recognition (pp. 113-124). Cham: Springer Nature Switzerland. [**[Paper]**](10.1007/978-3-031-42795-4_11) [**[Code]**](https://scm.univ-tours.fr/projetspublics/lifat/codegnn) 

0. **[CentralityPool]** Ghanbari, M., Bagloee, S. A., Qi, J., & Sarvi, M. (2025). CentralityPool: Centrality-Aware Hierarchical Graph Pooling. IEEE Transactions on Network Science and Engineering, 13, 314-326. [**[Paper]**](https://ieeexplore.ieee.org/document/11051044)

0. **[FRGCN]** Sun, L., Cai, Y., Ding, W., & Xu, J. (2025). Feature selection-based rough graph classification via graph convolutional neural networks. Engineering Applications of Artificial Intelligence, 158, 111502. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0952197625015040)

0. **[GTPool]** Li, G., Chen, J., & He, K. (2025). GTPool: Graph Transformer Pooling With Diverse Sampling. IEEE Transactions on Big Data, 11(6), 3255-3267. [**[Paper]**](https://ieeexplore.ieee.org/document/11077426)

0. **[HMTPool]** Li, Z. P., Su, H. L., Zhang, Q. H., Yuan, C. A., Gribova, V., Filaretov, V. F., & Huang, D. S. (2023). Hierarchical multiview top-k pooling with deep-Q-networks. IEEE Transactions on Artificial Intelligence, 5(6), 2985-2996. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/10324365)

0. **[EnsNet]** Guang, M., Li, Z., Zhang, R., Wang, J., & Cheng, D. (2026). Ensemble Graph Neural Networks with Individual Decision Feedback for Graph Classification. IEEE Transactions on Knowledge and Data Engineering. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11511857) [**[Code]**](https://github.com/guangmingjian/EnsNet) 

0. **[GMCAP]** Xu, Y., Wang, J., Guang, M., & Jiang, C. (2024). Graph multi-convolution and attention pooling for graph classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12), 10546-10557. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/10636839) [**[Code]**](https://github.com/xyhappy/GMCAP) 

0. **[CentralityPool]** Ghanbari, M., Bagloee, S. A., Qi, J., & Sarvi, M. (2025). CentralityPool: Centrality-Aware Hierarchical Graph Pooling. IEEE Transactions on Network Science and Engineering, 13, 314-326. [**[Paper]**](https://ieeexplore.ieee.org/document/11051044) 

0. **[EGAT-Pool]** Makieu, P., Dartue, E. D., Yeboah-Amankwah, K. O., Mulatu, O. H., & Tamiru, S. Y. (2026). EGAT-Pool: A Hierarchical Graph Attention Network with Edge-aware SAG Pooling for Robust Environmental Toxicity Prediction. Asian Journal of Research in Computer Science, 19(5), 10-24. [**[Paper]**](https://doi.org/10.9734/ajrcos/2026/v19i5858)

0. **[GP-Pool]** Liu, G., Lei, Y., Sun, M., Zhang, T., Wang, X., Zhou, C., ... & Cui, Z. (2025). Graph Probabilistic Pooling: From Bernoulli to Poisson Distribution. IEEE Transactions on Neural Networks and Learning Systems. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11267785) [**[Code]**](https://github.com/GuangbuLiu/Graph-Probabilistic-Pooling) 

0. **[DA-MAGPool]** Booryaee, R., Haddadian, P., & Kamandi, A. (2025, October). Dual-Mode Density-Aware Attention-based Hierarchical Graph Pooling. In 2025 15th International Conference on Computer and Knowledge Engineering (ICCKE) (pp. 1-6). IEEE. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11273849)

0. **[HPoolGCL]** Cen, F., Liu, S., Feng, Q., He, T., & Xu, J. (2026). HPoolGCL: Augmentation‐Free Cross‐Granularity Graph Contrastive Learning With Hierarchical Pooling. CAAI Transactions on Intelligence Technology, 11(2), 610-632. [**[Paper]**](https://onlinelibrary.wiley.com/doi/abs/10.1049/cit2.70096) [**[Code]**](https://github.com/Heycen/HPoolGCL)

0. **[SCHPool]** Zhao, Z., Bai, L., Li, M., Cui, L., Du, H., Wang, Y., & Hancock, E. R. (2025, August). An End-to-End Simple Clustering Hierarchical Pooling Operation for Graph Learning Based on Top-K Node Selection. In IJCAI (pp. 3670-3679). [**[Paper]**](https://doi.org/10.24963/ijcai.2025/408)  [**[Code]**](https://github.com/zhzhaoo/SCHPool) 

0. **[PRPOOL]** Huang, S., Wang, W., Li, Z., Wang, X., Xie, C., & Wang, Q. (2026). Towards Structure-captive Differentiable Graph Pooling via PageRank. IEEE Internet of Things Journal. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11595287) 

0. **[GPPTrans]** Li, Z. P., Wang, S. G., Zhang, Q. H., Liu, W. J., Gribova, V., Filaretov, V. F., ... & Huang, D. S. (2026). Graph pyramid pooling transformer for graph-level representation learning. IEEE Transactions on Big Data. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11499436)  [**[Code]**](https://github.com/zhzhaoo/SCHPool) 


### Edge Pooling
1. **[EdgePool]** Diehl F, Brunner T, Le MT, Knoll A (2019) Towards Graph Pooling by Edge Contraction. In: ICML 2019 Workshop on Learning and Reasoning with Graph-Structured Data [**[Paper1]**](https://arxiv.org/abs/1905.10990) [**[Paper2]**](https://graphreason.github.io/papers/17.pdf)

2. **[MeshPooling]** Yuan YJ, Lai YK, Yang J, et al (2020) Mesh variational autoencoders with edge contraction pooling. In: IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops. pp 1105–1112 [**[Paper]**](https://openaccess.thecvf.com/content_CVPRW_2020/html/w17/Yuan_Mesh_Variational_Autoencoders_With_Edge_Contraction_Pooling_CVPRW_2020_paper.html) [**[Code]**](https://github.com/IGLICT/MeshPooling) 

3. **[DHT]** Jo J, Baek J, Lee S, et al (2021) Edge Representation Learning with Hypergraphs. In: Advances in Neural Information Processing Systems [**[Paper]**](https://proceedings.neurips.cc/paper_files/paper/2021/hash/3def184ad8f4755ff269862ea77393dd-Abstract.html) [**[Code]**](https://github.com/harryjo97/EHGNN) 

4. **[H2MN]** Zhang Z, Bu J, Ester M, et al (2021b) H2MN: Graph Similarity Learning with Hierarchical Hypergraph Matching Networks. In: Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp 2274–2284 [**[Paper]**](https://doi.org/10.1145/3447548.3467328) [**[Code]**](https://github.com/cszhangzhen/H2MN) 

4. **[CurvPool]** Sanders, C., Roth, A., & Liebig, T. (2023, September). Curvature-based pooling within graph neural networks. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 471-485). Cham: Springer Nature Switzerland. [**[Paper]**](https://doi.org/10.48550/arXiv.2308.16516) 

4. **[MESPool]** Xu, F., Yang, Z., Wang, L., Meng, D., & Long, J. (2023). MESPool: Molecular Edge Shrinkage Pooling for hierarchical molecular representation learning and property prediction. Briefings in Bioinformatics, 25(1). [**[Paper]**](https://doi.org/10.1093/bib/bbad423) [**[Code]**](https://github.com/xfd997700/MESPool) 

4. **[LGRPool]** Noravesh, F., Haffari, R., Soon, L., & Pal, A. (2025). LGRPool: Hierarchical Graph Pooling Via Local-Global Regularisation. arXiv preprint arXiv:2504.08530. [**[Paper]**](10.48550/arXiv.2504.08530)

4. **[EdgePoolV2]** Landolfi, F. (2022). Revisiting Edge Pooling in Graph Neural Networks. In ESANN. [**[Paper]**](https://www.esann.org/sites/default/files/proceedings/2022/ES2022-92.pdf)

4. **[IM-MPNN]** Finder, S. E., Weber, R. S., Eliasof, M., Freifeld, O., & Treister, E. (2025). Improving the effective receptive field of message-passing neural networks. arXiv preprint arXiv:2505.23185. [**[Paper]**](http://arxiv.org/abs/2505.23185) [**[Code]**](https://github.com/BGU-CS-VIL/IM-MPNN) 

4. **[MagEdge]** Limbeck, K., Mezrag, L., Wolf, G., & Rieck, B. (2026). Geometry-aware edge pooling for graph neural networks. Advances in Neural Information Processing Systems, 38, 157770-157808. [**[Paper]**](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e76c24095f4f6fcd77a2e594062085a9-Abstract-Conference.html) [**[Code]**](https://github.com/aidos-lab/mag_edge_pool) 

4. **[CIV]** Gao, Z., Xiao, L., Du, X., & Xing, Y. (2025). CIV: Leveraging causal subgraphs of vulnerability for noise reduction in vulnerability detection. Expert Systems with Applications, 130392. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0957417425040072) [**[Code]**](https://github.com/boomboomgzy/CIV) 

### Hybrid Pooling
1. **[LookHops]** Gao Z, Lin H, Li StanZ (2020) LookHops: light multi-order convolution and pooling for graph classification. arXiv preprint arXiv:201215741 [**[Paper]**](https://arxiv.org/abs/2012.15741)
2. **[ProxPool]** Gao X, Dai W, Li C, et al (2021b) Multiscale Representation Learning of Graph Data With Node Affinity. IEEE Trans Signal Inf Process Netw 7:30–44.  [**[Paper]**](https://doi.org/10.1109/TSIPN.2020.3044913)

3. **[ASPool]** Yu H, Yuan J, Yao Y, Wang C (2022) Not all edges are peers: Accurate structure-aware graph pooling networks. Neural Networks 156:58–66. [**[Paper]**](https://doi.org/10.1016/j.neunet.2022.09.004)

4. **[Co-Pooling]** Zhou Xiaowei and Yin J and TIW (2023) Edge but not Least: Cross-View Graph Pooling. In: Machine Learning and Knowledge Discovery in Databases. pp 344–359 [**[Paper]**](https://doi.org/10.1007/978-3-031-26390-3_21)

4. **[WGDPool]** Xiao, Z., Chen, H., Xiao, L., Wei, P., Wang, T., & Li, K. (2024). WGDPool: A broad scope extraction for weighted graph data. Expert Systems with Applications, 249, 123678. [**[Paper]**](https://doi.org/10.1016/j.eswa.2024.123678)

5. **[HPM-GMN]** Huang, Y., Miao, S., Zou, J., Mu, Y., Tian, Y., Wang, J., ... & Yao, S. (2025). Hpm-gmn: Hierarchical pooling multi-level graph matching network. Knowledge-Based Systems, 113677. [**[Paper]**](https://doi.org/10.1016/j.knosys.2025.113677) [**[Code]**](https://github.com/ynu123/HPM-GMN)

5. **[Mpool]** Islam, M. I. K., Khanov, M., & Akbas, E. (2023, May). Mpool: motif-based graph pooling. In Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp. 105-117). Cham: Springer Nature Switzerland. [**[Paper]**](10.1007/978-3-031-33377-4_9)

5. **[GPS]** Ju, W., Gu, Y., Mao, Z., Qiao, Z., Qin, Y., Luo, X., ... & Zhang, M. (2025). GPS: Graph contrastive learning via multi-scale augmented views from adversarial pooling. Science China Information Sciences, 68(1), 112101. [**[Paper]**](10.1007/s11432-022-3952-3)

5. **[DSMVPool]** Ali, W., Vascon, S., Stadelmann, T., & Pelillo, M. (2025). Multi-view graph pooling via dominant sets for graph classification. Pattern Recognition, 112786. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0031320325014499)

5. **[ENAHPool]** Zhao, Z., Bai, L., Cui, L., Li, M., Lyu, Z., Xu, L., ... & Hancock, E. R. (2025, July). ENAHPool: The Edge-Node Attention-based Hierarchical Pooling for Graph Neural Networks. In ICML. [**[Paper]**](https://openreview.net/forum?id=zVJGILkCRZ)

5. Grover, A., Giaretta, L., Bourgerie, R., & Girdzijauskas, S. (2026). Is One Token All It Takes? Graph Pooling Tokens for LLM-based GraphQA. arXiv preprint arXiv:2604.00342. [**[Paper]**](http://arxiv.org/abs/2604.00342) [**[Code]**](https://github.com/Agrover112/G-Retriever/tree/all_good/)

5. **[GAS]** Liu, N., Jian, S., & Li, D. (2025). Substructure-Aware Graph Representation Learning via Adaptive Node Absorbing. IEEE Transactions on Big Data. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11314744) [**[Code]**](https://github.com/liuning222/GAS)

5. **[ADP]** Cao, Z., Xu, L., Zhang, R., Zhang, J., Pei, H., Zhou, D., & Qiu, J. (2025). ADP: graph adaptive pooling based on edge understanding with graph pooling information bottleneck. IEEE Transactions on Consumer Electronics. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11296911)


## Graph Unpooling
1. **[UL]** Guo Y, Zou D, Lerman G (2023) An Unpooling Layer for Graph Generation. In: Proceedings of The 26th International Conference on Artificial Intelligence and Statistics. PMLR, pp 3179–3209 [**[Paper]**](https://proceedings.mlr.press/v206/guo23a.html) [**[Code]**](https://github.com/guo00413/graph_unpooling) 

1. **[pastUnpool]** Amouzad, A., Dehghanian, Z., Saravani, S., Amirmazlaghani, M., & Roshanfekr, B. (2024). Graph isomorphism u-net. Expert Systems with Applications, 236, 121280. [**[Paper]**](https://doi.org/10.1016/j.eswa.2023.121280) 

1. Mesgaran, M., & Hamza, A. B. (2023). A graph encoder–decoder network for unsupervised anomaly detection. Neural Computing and Applications, 35(32), 23521-23535. [**[Paper]**](https://doi.org/10.1016/j.eswa.2023.121280) 

1. **[Auto-UCPool]** Peng, H., Yao, M., Long, X., Luo, Y., Liang, Q., Pan, J., ... & Chen, Y. (2026). Contextual stroke classification in online handwritten documents with auto clustering pooling method. Pattern Recognition, 113356. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0031320326003213) 

## Graph Pooling for Complex Graphs

### Hypergraph

1. **[DyH2GNet]** Hayat, M. K., Xue, S., Wu, J., Khan, B., & Yang, J. (2025, March). Self-supervised time-aware heterogeneous hypergraph learning for dynamic graph-level classification. In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining (pp. 213-221). [**[Paper]**](https://dl.acm.org/doi/10.1145/3701551.3703504) [**[Code]**](https://github.com/khizer-hayat/DyH2GNet)

### Heterophilic Graph

1. **[Maxcutpool]** Abate, C., & Bianchi, F. M. (2025, May). Maxcutpool: differentiable feature-aware maxcut for pooling in graph neural networks. In International Conference on Learning Representations (Vol. 2025, pp. 52506-52533). [**[Paper]**](https://openreview.net/forum?id=xlbXRJ2XCP)

### Dynamic Graph
1. Rastgoo, R., Kiani, K., & Escalera, S. (2026). Hierarchical Graph Frequency-Selective Diffusion Model for Personalized Sign Language Production. IEEE Transactions on Multimedia. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11656694)

1. Khodayar, M., Bavil, A. F., & Saffari, M. (2026). Spatiotemporal Traffic Speed Forecasting via Conditional Flow-Based Generative Deep Learning With Dynamic Graph Pooling. IEEE Transactions on Emerging Topics in Computational Intelligence. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11456093)

1. Mei, T., Zhu, M., Zhang, R., Wang, S., Xiang, J., & He, X. (2026). A Gradient-Guided Spatio-Temporal Graph Convolutional Network for Population-Level Major Depressive Disorder Classification. IEEE Journal of Biomedical and Health Informatics. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11599418)

### Heterogeneous Graph

1. **[Hetero-pooling]** Yao, W., Ju, C., Yang, G., & Chu, J. (2025). HPGoG: Hetero-Pooling Integrated with Graph-of-Graphs to Alleviate Class Imbalance for Fault Tracing of Transformer Substation Devices. Digital Signal Processing, 105822. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S1051200425008383)

1. **[UaG]** Wu, C., Wu, F., Huang, Y., & Xie, X. (2021, August). User-as-graph: User modeling with heterogeneous graph pooling for news recommendation. In IJCAI (pp. 1624-1630). [**[Paper]**](https://www.ijcai.org/proceedings/2021/224)

1. **[RePooL]** Baek, H. Y., Choi, J., Seo, J., Jin, X., Lee, D., & Oh, B. (2025, November). Relation-faceted graph pooling with llm guidance for dynamic span-aware information extraction. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management (pp. 109-118). [**[Paper]**](https://dl.acm.org/doi/10.1145/3746252.3761243) [**[Code]**](https://github.com/hyeyoon100/RePooL)

1. **[HINPool]** Hong, M. Y., Teng, Y. C., Lin, S. E., Wang, C. Y., & Lin, C. (2026, March). HINPool: A Unified Heterogeneous Graph Pooling Framework for Accurate Molecular and Protein Property Prediction. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 40, No. 26, pp. 21752-21760). [**[Paper]**](https://ojs.aaai.org/index.php/AAAI/article/view/39326) [**[Code]**](https://github.com/ntuidssplab/HINPool)

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

## Libraries
| Librariy | URL  | Language and Platform | Description    |
|-------|-------|-------|-------|
|  |  |  |  |
| PyTorch Geometric (PyG) | https://www.pyg.org/ | Python, PyTorch | PyG is a library built upon  PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.It consists of various methods for deep learning on graphs and other irregular structures. [**[Home]**](https://www.pyg.org/) [**[Docs]**](https://pytorch-geometric.readthedocs.io/en/latest/) [**[Codes]**](https://github.com/pyg-team/pytorch_geometric) [**[Guides]**](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html) |
| Deep Graph Library | https://www.dgl.ai/ | Python, PyTorch/ TensorFlow/ MXNet | DGL is a Python package built for easy implementation of graph neural network model family, on top of existing DL frameworks (currently supporting PyTorch, MXNet and TensorFlow). It offers a versatile control of message passing, speed optimization via auto-batching and highly tuned sparse matrix kernels, and multi-GPU/CPU training to scale to graphs of hundreds of millions of nodes and edges. [**[Home]**](https://www.dgl.ai/) [**[Docs]**](https://docs.dgl.ai/) [**[Codes]**](https://github.com/dmlc/dgl) [**[Guides]**](https://docs.dgl.ai/tutorials/blitz/index.html) |
| TensorFlow GNN (TF-GNN) | https://github.com/tensorflow/gnn | Python, TensorFlow | TensorFlow GNN is a library to build Graph Neural Networks on the TensorFlow platform. It provides a tfgnn.GraphTensor type to represent graphs with a heterogeneous schema, that is, multiple types of nodes and edges; tools for data preparation, notably a graph sampler to convert a huge database into a stream of reasonably-sized subgraphs for training and inference; a collection of ready-to-use models and Keras layers to do your own GNN modeling; a high-level API for training orchestration. [**[Home]**](https://github.com/tensorflow/gnn) [**[Docs]**](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/README.md) [**[Codes]**](https://github.com/tensorflow/gnn) [**[Guides]**](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/overview.md) |
| tf_geometric | https://github.com/CrawlScript/tf_geometric | Python, TensorFlow | Tf_geometric is a Efficient and Friendly Graph Neural Network Library for TensorFlow 1.x and 2.x, inspired by rusty1s/pytorch_geometric. [**[Home]**](https://github.com/CrawlScript/tf_geometric) [**[Docs]**](https://tf-geometric.readthedocs.io/en/latest/) [**[Codes]**](https://github.com/CrawlScript/tf_geometric) [**[Guides]**](https://tf-geometric.readthedocs.io/en/latest/wiki/quickstart.html) [**[Paper]**](https://arxiv.org/abs/2101.11552) |
| Jraph | https://github.com/google-deepmind/jraph | Python, JAX | Jraph (pronounced “giraffe”) is a lightweight library for working with graph neural networks in jax. It provides a data structure for graphs, a set of utilites for working with graphs, and a ‘zoo’ of forkable graph neural network models. [**[Home]**](https://github.com/google-deepmind/jraph) [**[Docs]**](https://jraph.readthedocs.io/en/latest/) [**[Codes]**](https://github.com/google-deepmind/jraph) [**[Guides]**](https://github.com/google-deepmind/jraph#quick-start) |
| Spektral | https://graphneural.network/ | Python, TensorFlow & Keras | Spektral is a Python library for graph deep learning, based on the Keras API and TensorFlow 2. The main goal of this project is to provide a simple but flexible framework for creating graph neural networks (GNNs). [**[Home]**](https://graphneural.network/) [**[Docs]**](https://graphneural.network/) [**[Codes]**]() [**[Guides]**](https://graphneural.network/getting-started/) [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/9321429) |
| GraphNeuralNetworks.jl | https://github.com/CarloLucibello/GraphNeuralNetworks.jl | Julia, Flux | GraphNeuralNetworks.jl is a fresh, performant and flexible graph neural network library based on Flux.jl. [**[Home]**](https://github.com/CarloLucibello/GraphNeuralNetworks.jl) [**[Docs]**](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/) [**[Codes]**](https://github.com/CarloLucibello/GraphNeuralNetworks.jl) [**[Guides]**](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/tree/master/examples) |
| GeometricFlux.jl | https://github.com/FluxML/GeometricFlux.jl | Julia, Flux | GeometricFlux is a geometric deep learning library for Flux. This library aims to be compatible with packages from JuliaGraphs ecosystem and have support of CUDA GPU acceleration with CUDA. Message passing scheme is implemented as a flexbile framework and fused with Graph Network block scheme. GeometricFlux is compatible with other packages that are composable with Flux. [**[Home]**](https://github.com/FluxML/GeometricFlux.jl) [**[Docs]**](https://github.com/FluxML/GeometricFlux.jl/blob/master/docs/src/index.md) [**[Codes]**](https://github.com/FluxML/GeometricFlux.jl) [**[Guides]**](https://github.com/FluxML/GeometricFlux.jl/tree/master/docs/tutorials) |


## Evaluation, Review and Analysis
1. Mesquita, D., Souza, A., & Kaski, S. (2020). Rethinking pooling in graph neural networks. Advances in Neural Information Processing Systems, 33, 2220-2231. [**[Paper]**](https://proceedings.neurips.cc/paper/2020/hash/1764183ef03fc7324eb58c3842bd9a57-Abstract.html) [**[Code]**](https://github.com/AaltoPML/Rethinking-pooling-in-GNNs) 

1. Grattarola, D., Zambon, D., Bianchi, F. M., & Alippi, C. (2022). Understanding pooling in graph neural networks. IEEE Transactions on Neural Networks and Learning Systems. [**[Paper]**](https://doi.org/10.1109/TNNLS.2022.3190922) [**[Code]**](https://github.com/danielegrattarola/SRC) 

1. Bianchi, F. M., & Lachi, V. (2024). The expressive power of pooling in graph neural networks. Advances in Neural Information Processing Systems, 36. [**[Paper]**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e26f31de8b13ec569bf507e6ae2cd952-Abstract-Conference.html) [**[Code]**](https://github.com/FilippoMB/The-expressive-power-of-pooling-in-GNNs) 

1. Liu, C., Zhan, Y., Wu, J., Li, C., Du, B., Hu, W., ... & Tao, D. (2022). Graph pooling for graph neural networks: Progress, challenges, and opportunities. arXiv preprint arXiv:2204.07321.  [**[Paper]**](https://arxiv.org/abs/2204.07321)

1. Cheung, M., Shi, J., Jiang, L., Wright, O., & Moura, J. M. (2019, November). Pooling in graph convolutional neural networks. In 2019 53rd Asilomar Conference on Signals, Systems, and Computers (pp. 462-466). IEEE. [**[Paper]**](https://doi.org/10.1109/IEEECONF44664.2019.9048796) 

1. Errica, F., Podda, M., Bacciu, D., & Micheli, A. (2019, September). A Fair Comparison of Graph Neural Networks for Graph Classification. In International Conference on Learning Representations. [**[Paper]**](https://openreview.net/forum?id=HygDF6NFPB) [**[Code]**](https://github.com/diningphil/gnn-comparison) 

1. Yang, Z., Zhang, G., Wu, J., Yang, J., Sheng, Q. Z., Xue, S., ... & Lio, P. (2024). State of the art and potentialities of graph-level learning. ACM Computing Surveys, 57(2), 1-40. [**[Paper]**](https://doi.org/10.1145/3695863) [**[Github]**](https://github.com/ZhenyuYangMQ/Awesome-Graph-Level-Learning) 

1. Lachi, V., Moallemy-Oureh, A., Roth, A., & Welke, P. (2023, December). Graph Pooling Provably Improves Expressivity. In NeurIPS 2023 Workshop: New Frontiers in Graph Learning. [**[Paper]**](https://openreview.net/forum?id=lR5NYB9zrv) 

1. Hashemi, M., Gong, S., Ni, J., Fan, W., Prakash, B. A., & Jin, W. (2024, August). A comprehensive survey on graph reduction: sparsification, coarsening, and condensation. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (pp. 8058-8066). [**[Paper]**](https://dl.acm.org/doi/abs/10.24963/ijcai.2024/891) 

1. Job, S., Tao, X., Cai, T., Li, L., Sheng, Q. Z., Xie, H., & Yong, J. (2025). Causal integration in graph neural networks toward enhanced classification: benchmarking and advancements for robust performance. World Wide Web, 28(3), 30. [**[Paper]**](https://doi.org/10.1007/s11280-025-01343-1) 

1. Li, Z. P., Wang, S. G., Zhang, Q. H., Pan, Y. J., Xiao, N. A., Guo, J. Y., ... & Huang, D. S. (2024). Graph pooling for graph-level representation learning: a survey. Artificial Intelligence Review, 58(2), 45. [**[Paper]**](https://doi.org/10.1007/s10462-024-10949-2) 

1. Wang, P., Luo, J., Shen, Y., Zhang, M., Qin, S., Heng, S., & Luo, X. (2024). A comprehensive graph pooling benchmark: Effectiveness, robustness and generalizability. arXiv preprint arXiv:2406.09031. [**[Paper]**](https://openreview.net/forum?id=Onw93uJCWO)[**[Code]**](https://github.com/goose315/Graph_Pooling_Benchmark) 

1. Abate, C., Marisca, I., & Bianchi, F. M. (2025). Torch Geometric Pool: the PyTorch library for pooling in Graph Neural Networks. arXiv preprint arXiv:2512.12642. [**[Paper]**](https://openreview.net/forum?id=1wnHkzahst)

1. Luo, Y., Shi, L., & Wu, X. M. (2025). Can classic gnns be strong baselines for graph-level tasks? simple architectures meet excellence. arXiv preprint arXiv:2502.09263. [**[Paper]**](http://arxiv.org/abs/2502.09263)

1. von Pichowski, J., Hrabošová, A., Scholtes, I., & Blöcker, C. (2026). The Role of Node Features in Graph Pooling. arXiv preprint arXiv:2605.06250. [**[Paper]**](http://arxiv.org/abs/2605.06250)


## Applications in Multi-omics Data
### Genomics
1. **[HGNN]** Hou, W., Wang, Y., Zhao, Z., Cong, Y., Pang, W., & Tian, Y. (2024). Hierarchical graph neural network with subgraph perturbations for key gene cluster discovery in cancer staging. Complex & Intelligent Systems, 10(1), 111-128. [**[Paper]**](https://doi.org/10.1007/s40747-023-01068-6)

1. **[GCNN]** Ramirez, R., Chiu, Y. C., Hererra, A., Mostavi, M., Ramirez, J., Chen, Y., ... & Jin, Y. F. (2020). Classification of cancer types using graph convolutional neural networks. Frontiers in physics, 8, 203. [**[Paper]**](https://doi.org/10.3389/fphy.2020.00203) [**[Code]**](https://github.com/RicardoRamirez2020/GCN_Cancer)

1. **[Surv_GCNN]** Ramirez, R., Chiu, Y. C., Zhang, S., Ramirez, J., Chen, Y., Huang, Y., & Jin, Y. F. (2021). Prediction and interpretation of cancer survival using graph convolution neural networks. Methods, 192, 120-130. [**[Paper]**](https://doi.org/10.1016/j.ymeth.2021.01.004)

1. **[GNN-SubNet]** Pfeifer, B., Saranti, A., & Holzinger, A. (2022). GNN-SubNet: disease subnetwork detection with explainable graph neural networks. Bioinformatics, 38(Supplement_2), ii120-ii126. [**[Paper]**](https://doi.org/10.1093/bioinformatics/btac478) [**[Code]**](https://github.com/pievos101/GNN-SubNet)

1. **[GLRP]** Chereda, H., Bleckmann, A., Menck, K., Perera-Bel, J., Stegmaier, P., Auer, F., ... & Beißbarth, T. (2021). Explaining decisions of graph convolutional neural networks: patient-specific molecular subnetworks responsible for metastasis prediction in breast cancer. Genome medicine, 13, 1-16. [**[Paper]**]( https://doi.org/10.1186/s13073-021-00845-7) [**[Code]**]( https://gitlab.gwdg.de/UKEBpublic/graph-lrp)

1. **[CytoCommunity]** Hu, Y., Rong, J., Xu, Y., Xie, R., Peng, J., Gao, L., & Tan, K. (2024). Unsupervised and supervised discovery of tissue cellular neighborhoods from cell phenotypes. Nature Methods, 1-12. [**[Paper]**](https://doi.org/10.1038/s41592-023-02124-2) [**[Code]**](https://github.com/tanlabcode/CytoCommunity) [**[Code]**](https://www.zenodo.org/record/8335454)

1. **[sigGCN]** Wang, T., Bai, J., & Nabavi, S. (2021). Single-cell classification using graph convolutional networks. BMC bioinformatics, 22, 1-23. [**[Paper]**](https://doi.org/10.1186/s12859-021-04278-2) [**[Code]**](https://github.com/NabaviLab/sigGCN)

0.  Rhee, S., Seo, S., & Kim, S. (2018, July). Hybrid approach of relation network and localized graph convolutional filtering for breast cancer subtype classification. In Proceedings of the 27th International Joint Conference on Artificial Intelligence (pp. 3527-3534). [**[Paper]**](https://doi.org/10.24963/ijcai.2018/490)

0. **[PathGNN]** Liang, B., Gong, H., Lu, L., & Xu, J. (2022). Risk stratification and pathway analysis based on graph neural network and interpretable algorithm. BMC bioinformatics, 23(1), 394. [**[Paper]**](https://doi.org/10.1186/s12859-022-04950-1) [**[Code]**](https://github.com/BioAI-kits/PathGNN) 

0. **[MOFNet]** Zhang, G., Zhang, C., Li, P., Sun, D., Yang, Z., & Liu, Z. P. (2025). MOFNet: a deep learning framework for multi-omics data fusion in cancer subtype classification. Molecular Omics, 21(6), 690-705. [**[Paper]**](https://doi.org/10.1039/D5MO00221D)

0. **[GraphLooper]** Wang, S., Li, Z., Feng, H., Cui, Z., Guo, Z. H., Liao, Q., ... & Huang, D. S. (2026). GraphLooper: predicting chromatin loops based on hierarchical multi-view graph pooling method. Briefings in Bioinformatics, 27(3), bbag217. [**[Paper]**](https://academic.oup.com/bib/article/doi/10.1093/bib/bbag217/8675950) [**[Code]**](https://github.com/wangguoguoa/GraphLooper) 

0. **[HRCHY-CytoCommunity]** Xie, R., Wang, Z., Liu, J., Xu, H., Xu, Y., Lin, J., ... & Gao, L. (2026). HRCHY-CytoCommunity identifies hierarchical tissue organization in cell-type spatial maps. Nature Communications, 17(1), 3312. [**[Paper]**](https://doi.org/10.1038/s41467-026-70069-z) [**[Code]**](https://github.com/huBioinfo/HRCHYCytoCommunity) 

### Radiomics
1. **[BrainGNN]** Li, X., Zhou, Y., Dvornek, N., Zhang, M., Gao, S., Zhuang, J., ... & Duncan, J. S. (2021). Braingnn: Interpretable brain graph neural network for fmri analysis. Medical Image Analysis, 74, 102233. [**[Paper]**](https://doi.org/10.1016/j.media.2021.102233) [**[Code]**](https://github.com/xxlya/BrainGNN_Pytorch) 
0. **[DH-SAGPool]** Zhang, S., Wang, J., Yu, S., Wang, R., Han, J., Zhao, S., ... & Lv, J. (2023). An explainable deep learning framework for characterizing and interpreting human brain states. Medical Image Analysis, 83, 102665. [**[Paper]**](https://doi.org/10.1016/j.media.2022.102665)
0. **[GAT-LI]** Hu, J., Cao, L., Li, T., Dong, S., & Li, P. (2021). GAT-LI: a graph attention network based learning and interpreting method for functional brain network classification. BMC bioinformatics, 22(1), 1-20. [**[Paper]**](https://doi.org/10.1186/s12859-021-04295-1) [**[Code]**](https://github.com/largeapp/gat-li) 
0. **[HSGPL]** Tang, H., Ma, G., Guo, L., Fu, X., Huang, H., & Zhan, L. (2022). Contrastive brain network learning via hierarchical signed graph pooling model. IEEE Transactions on Neural Networks and Learning Systems. [**[Paper]**](https://doi.org/10.1109/TNNLS.2022.3220220)
0. **[GC+P]** Gopinath, K., Desrosiers, C., & Lombaert, H. (2020). Learnable pooling in graph convolutional networks for brain surface analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(2), 864-876.  [**[Paper]**](https://doi.org/10.1109/TPAMI.2020.3028391) [**[Code]**](https://github.com/kharitz/learnpool.git) 
0. **[Multi-Channel Pooling]** Song, X., Zhou, F., Frangi, A. F., Cao, J., Xiao, X., Lei, Y., ... & Lei, B. (2022). Multicenter and Multichannel Pooling GCN for Early AD Diagnosis Based on Dual-Modality Fused Brain Network. IEEE Transactions on Medical Imaging, 42(2), 354-367. [**[Paper]**](https://doi.org/10.1109/TMI.2022.3187141) [**[Code]**](https://github.com/Xuegang-S/MMP-GCN)
0. **[MM-GNN]** Sebenius, I., Campbell, A., Morgan, S. E., Bullmore, E. T., & Liò, P. (2021, October). Multimodal graph coarsening for interpretable, MRI-based brain graph neural network. In 2021 IEEE 31st International Workshop on Machine Learning for Signal Processing (MLSP) (pp. 1-6). IEEE. [**[Paper]**](https://doi.org/10.1109/MLSP52302.2021.9690626) 
0. **[PR-GNN]** Li, X., Zhou, Y., Dvornek, N. C., Zhang, M., Zhuang, J., Ventola, P., & Duncan, J. S. (2020). Pooling regularized graph neural network for fmri biomarker analysis. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part VII 23 (pp. 625-635). Springer International Publishing. [**[Paper]**](https://doi.org/10.1007/978-3-030-59728-3_61)
0. **[SA-GCN]** Zhao, F., Li, N., Pan, H., Chen, X., Li, Y., Zhang, H., ... & Cheng, D. (2022). Multi-view feature enhancement based on self-attention mechanism graph convolutional network for autism spectrum disorder diagnosis. Frontiers in human neuroscience, 16, 918969. [**[Paper]**](https://doi.org/10.3389/fnhum.2022.918969) 
0. **[TMGCN]** Gao, Y., Tang, Y., Zhang, H., Yang, Y., Dong, T., & Jia, Q. (2022). Sex differences of cerebellum and cerebrum: evidence from graph convolutional network. Interdisciplinary Sciences: Computational Life Sciences, 14(2), 532-544. [**[Paper]**](https://doi.org/10.1007/s12539-021-00498-5) 
0. **[CNN-GCN]** Gao, Z., Lu, Z., Wang, J., Ying, S., & Shi, J. (2022). A convolutional neural network and graph convolutional network based framework for classification of breast histopathological images. IEEE Journal of Biomedical and Health Informatics, 26(7), 3163-3173. [**[Paper]**](https://doi.org/10.1109/JBHI.2022.3153671) 
0. **[b-HGFN]** Di, D., Zhang, J., Lei, F., Tian, Q., & Gao, Y. (2022). Big-hypergraph factorization neural network for survival prediction from whole slide image. IEEE Transactions on Image Processing, 31, 1149-1160. [**[Paper]**](https://doi.org/10.1109/TIP.2021.3139229) 
0. **[HACT]** Pati, P., Jaume, G., Foncubierta-Rodriguez, A., Feroce, F., Anniciello, A. M., Scognamiglio, G., ... & Gabrani, M. (2022). Hierarchical graph representations in digital pathology. Medical image analysis, 75, 102264. [**[Paper]**](https://doi.org/10.1016/j.media.2021.102264) [**[Code]**](https://github.com/histocartography/hact-net) 
0. **[MULTIPLAI]** Martin-Gonzalez, P., Crispin-Ortuzar, M., & Markowetz, F. (2021). Predictive modelling of highly multiplexed tumour tissue images by graph neural networks. In Interpretability of Machine Intelligence in Medical Image Computing, and Topological Data Analysis and Its Applications for Medical Data: 4th International Workshop, iMIMIC 2021, and 1st International Workshop, TDA4MedicalData 2021, Held in Conjunction with MICCAI 2021, Strasbourg, France, September 27, 2021, Proceedings 4 (pp. 98-107). Springer International Publishing. [**[Paper]**](https://doi.org/10.1007/978-3-030-87444-5_10) [**[Code]**]( https://github.com/markowetzlab/MULTIPLAI) 
0. **[GNN-MIL]** Adnan, M., Kalra, S., & Tizhoosh, H. R. (2020). Representation learning of histopathology images using graph neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 988-989). [**[Paper]**](https://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Adnan_Representation_Learning_of_Histopathology_Images_Using_Graph_Neural_Networks_CVPRW_2020_paper.html)

0. **[CGC-Net]** Zhou, Y., Graham, S., Alemi Koohbanani, N., Shaban, M., Heng, P. A., & Rajpoot, N. (2019). Cgc-net: Cell graph convolutional network for grading of colorectal cancer histology images. In Proceedings of the IEEE/CVF international conference on computer vision workshops (pp. 0-0). [**[Paper]**](https://openaccess.thecvf.com/content_ICCVW_2019/html/VRMI/Zhou_CGC-Net_Cell_Graph_Convolutional_Network_for_Grading_of_Colorectal_Cancer_ICCVW_2019_paper.html) 

0. Studer, L., Wallau, J., Dawson, H., Zlobec, I., & Fischer, A. (2021, January). Classification of intestinal gland cell-graphs using graph neural networks. In 2020 25th International conference on pattern recognition (ICPR) (pp. 3636-3643). IEEE. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/9412535) [**[Code]**](https://github.com/waljan/GNNpT1) 

0. Zhou, H., He, L., Zhang, Y., Shen, L., & Chen, B. (2022, March). Interpretable graph convolutional network of multi-modality brain imaging for alzheimer’s disease diagnosis. In 2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI) (pp. 1-5). IEEE. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/9761449) 

1. **[MS-GWNN]** Zhang, M., Dong, B., & Li, Q. (2022, March). MS-GWNN: multi-scale graph wavelet neural network for breast cancer diagnosis. In 2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI) (pp. 1-5). IEEE. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/9761464) 

1. **[GCN-based MIL]** Zhao, Y., Yang, F., Fang, Y., Liu, H., Zhou, N., Zhang, J., ... & Yao, J. (2020). Predicting lymph node metastasis using histopathological images based on multiple instance learning with deep graph convolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4837-4846). [**[Paper]**](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhao_Predicting_Lymph_Node_Metastasis_Using_Histopathological_Images_Based_on_Multiple_CVPR_2020_paper.html) 

1. **[CBHIR]** Zheng, Y., Jiang, B., Shi, J., Zhang, H., & Xie, F. (2019). Encoding histopathological wsis using gnn for scalable diagnostically relevant regions retrieval. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part I 22 (pp. 550-558). Springer International Publishing. [**[Paper]**](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_61) 

1. Wang, J., Chen, R. J., Lu, M. Y., Baras, A., & Mahmood, F. (2020, April). Weakly supervised prostate tma classification via graph convolutional networks. In 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI) (pp. 239-243). IEEE. [**[Paper]**](https://doi.org/10.1109/ISBI45749.2020.9098534) 

1. **[CBIR]** Ozen, Y., Aksoy, S., Kösemehmetoğlu, K., Önder, S., & Üner, A. (2021, January). Self-supervised learning with graph neural networks for region of interest retrieval in histopathology. In 2020 25th International conference on pattern recognition (ICPR) (pp. 6329-6334). IEEE. [**[Paper]**]( https://doi.org/10.1109/ICPR48806.2021.9412903) 

1. **[STAGIN]** Kim, B. H., Ye, J. C., & Kim, J. J. (2021). Learning dynamic graph representation of brain connectome with spatio-temporal attention. Advances in Neural Information Processing Systems, 34, 4314-4327. [**[Paper]**](https://proceedings.neurips.cc/paper_files/paper/2021/hash/22785dd2577be2ce28ef79febe80db10-Abstract.html)  [**[Code]**](https://github.com/egyptdj/stagin)

1. **[ContrastPool]** Xu, J., Bian, Q., Li, X., Zhang, A., Ke, Y., Qiao, M., ... & Gulyás, B. (2024). Contrastive Graph Pooling for Explainable Classification of Brain Networks. IEEE Transactions on Medical Imaging. [**[Paper]**]( https://doi.org/10.1109/TMI.2024.3392988)  [**[Code]**](https://github.com/AngusMonroe/ContrastPool)

1. **[CLHOI]** Liu, S., & Gui, R. (2024). Fusing multi-scale fMRI features using a brain-inspired multi-channel graph neural network for major depressive disorder diagnosis. Biomedical Signal Processing and Control, 90, 105837. [**[Paper]**](https://doi.org/10.1016/j.bspc.2023.105837) 

1. **[DSL-GNN]** Cao, J., Yang, L., Sarrigiannis, P. G., Blackburn, D., & Zhao, Y. (2024). Dementia classification using a graph neural network on imaging of effective brain connectivity. Computers in Biology and Medicine, 168, 107701. [**[Paper]**](https://doi.org/10.1016/j.compbiomed.2023.107701) 

1. **[BrainTGL]** Liu, L., Wen, G., Cao, P., Hong, T., Yang, J., Zhang, X., & Zaiane, O. R. (2023). BrainTGL: A dynamic graph representation learning model for brain network analysis. Computers in Biology and Medicine, 153, 106521.  [**[Paper]**](https://doi.org/10.1016/j.compbiomed.2022.106521)

1. **[PL-pooling]** Chan, T. H., Cendra, F. J., Ma, L., Yin, G., & Yu, L. (2023). Histopathology whole slide image analysis with heterogeneous graph representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 15661-15670).  [**[Paper]**](https://doi.org/10.1109/CVPR52729.2023.01503) [**[Code]**](https://github.com/HKU-MedAI/WSI-HGNN)

1. **[SFC-GNN]** Tong, W., Li, Y. X., Zhao, X. Y., Chen, Q. Q., Gao, Y. B., Li, P., & Wu, E. Q. (2023). fMRI-based brain disease diagnosis: a graph network approach. IEEE Transactions on Medical Robotics and Bionics.  [**[Paper]**](https://doi.org/10.1109/TMRB.2023.3270481)

1. **[GCN-SZ]** Chen, X., Ke, P., Huang, Y., Zhou, J., Li, H., Peng, R., ... & Wu, K. (2023). Discriminative analysis of schizophrenia patients using graph convolutional networks: A combined multimodal MRI and connectomics analysis. Frontiers in Neuroscience, 17, 1140801.   [**[Paper]**](https://doi.org/10.3389/fnins.2023.1140801) [**[Code]**](https://github.com/CXY-scut/GCN-SZ.git)

1. **[GRN]** Gürler, Z., Gharsallaoui, M. A., Rekik, I., & Alzheimer’s Disease Neuroimaging Initiative. (2023). Template-based graph registration network for boosting the diagnosis of brain connectivity disorders. Computerized Medical Imaging and Graphics, 103, 102140.   [**[Paper]**](https://doi.org/10.1016/j.compmedimag.2022.102140) [**[Code]**](https://github.com/basiralab/GRN)

1. **[BioX-CPath]** Gallagher-Syed, A., Senior, H., Alwazzan, O., Pontarini, E., Bombardieri, M., Pitzalis, C., ... & Slabaugh, G. (2025). BioX-CPath: Biologically-driven Explainable Diagnostics for Multistain IHC Computational Pathology. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 10372-10383).   [**[Paper]**](https://openaccess.thecvf.com/content/CVPR2025/html/Gallagher-Syed_BioX-CPath_Biologically-driven_Explainable_Diagnostics_for_Multistain_IHC_Computational_Pathology_CVPR_2025_paper.html) [**[Code]**]( https://github.com/AmayaGS/BioX-CPath)

1. **[DomainGCN]** Ding, R., Li, L., Soans, R., Shah, T., Krishnan, R., Sze, M. A., ... & Chen, A. (2025, April). Predicting ulcer in H&E images of inflammatory bowel disease using domain-knowledge-driven graph neural network. In 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI) (pp. 1-5). IEEE.  [**[Paper]**](10.1109/ISBI60581.2025.10980783) 

1. **[BRS-GCN]** Qin, Z., Li, Y., Song, X., & Chai, L. (2025). Classification of Neuropsychiatric Disorders via Brain-Region-Selected Graph Convolutional Network. IEEE Transactions on Neural Systems and Rehabilitation Engineering.  [**[Paper]**](10.1109/TNSRE.2025.3565627) 

1. **[GAFD]** Wang, Q., Zhang, J., Li, L., Qiao, L., Yap, P. T., & Liu, M. (2025). Graph augmentation guided federated knowledge distillation for multisite functional MRI analysis. Pattern Recognition, 164, 111559.  [**[Paper]**](https://doi.org/10.1016/j.patcog.2025.111559) 

1. Nazari, R., Salehi, M., & Shoeibi, A. (2025). An explainable connectome convolutional transformer for multimodal autism spectrum disorder classification. International journal of neural systems, 35(08), 2550043.  [**[Paper]**](https://www.worldscientific.com/doi/abs/10.1142/S0129065725500431) 

1. **[SAGEFusionNet]** Kumar, S., Hazarika, S., & Gupta, C. N. (2025). SAGEFusionNet: An auxiliary supervised graph neural network for brain age prediction as a neurodegenerative biomarker. Brain Sciences, 15(7), 752.  [**[Paper]**](https://www.mdpi.com/2076-3425/15/7/752) [**[Code]**](https://github.com/NeuralLabIITGuwahati/SAGEFusionNet)

1. **[DML‐GNN]** Liu, S., Sun, C., Li, J., Wang, S., & Zhao, L. (2025). DML‐GNN: ASD Diagnosis Based on Dual‐Atlas Multi‐Feature Learning Graph Neural Network. International Journal of Imaging Systems and Technology, 35(2), e70038.  [**[Paper]**](https://onlinelibrary.wiley.com/doi/abs/10.1002/ima.70038) 

1. **[KMGCN]** Zeng, X., Gong, J., Li, W., & Yang, Z. (2025). Knowledge-driven multi-graph convolutional network for brain network analysis and potential biomarker discovery. Medical Image Analysis, 99, 103368.  [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S1361841524002937) 

1. **[MSSTAN]** Kong, Y., Zhang, X., Wang, W., Zhou, Y., Li, Y., & Yuan, Y. (2024). Multi-scale spatial-temporal attention networks for functional connectome classification. IEEE Transactions on Medical Imaging, 44(1), 475-488.  [**[Paper]**](https://ieeexplore.ieee.org/document/10643532/) 

1. **[BrainRGIN]** Thapaliya, B., Akbas, E., Chen, J., Sapkota, R., Ray, B., Suresh, P., ... & Liu, J. (2025). Brain networks and intelligence: A graph neural network based approach to resting state fmri data. Medical Image Analysis, 101, 103433. [**[Paper]**](https://github.com/bishalth01/BrainRGIN/) 

1. **[GNN-LSTM]** Tang, J., Chen, J., Hu, M., Hu, Y., Zhang, Z., & Xiao, L. (2024). Diagnosis of autism spectrum disorder (ASD) by dynamic functional connectivity using GNN-LSTM. Sensors, 25(1), 156. [**[Paper]**](https://www.mdpi.com/1424-8220/25/1/156) 

1. **[CS-GCN]** Bera, A., Bhattacharjee, D., & Krejcar, O. (2025). Fluorescence microscopy and histopathology image based cancer classification using graph convolutional network with channel splitting. Biomedical Signal Processing and Control, 103, 107400.  [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S1746809424014587) [**[Code]**](https://github.com/asish-bera/CS-GCN)

1. Herzberg, W., Hauptmann, A., & Hamilton, S. J. (2023). Domain independent post-processing with graph U-nets: applications to electrical impedance tomographic imaging. Physiological measurement, 44(12), 125008. [**[Paper]**](https://doi.org/10.1088/1361-6579/ad0b3d) [**[Code]**](https://github.com/sarahjhamilton/open2Deit_agar)

1. **[ToPoMesh]** Chen, J., Zhu, Q., Xie, B., & Li, T. (2025). ToPoMesh: accurate 3D surface reconstruction from CT volumetric data via topology modification. Medical & Biological Engineering & Computing, 63(10), 3083-3098. [**[Paper]**](https://doi.org/10.1007/s11517-025-03381-3)

1. **[PBNA]** Chen, J., Zeng, S., Yang, J., & Du, Z. (2025). Panoramic brain network analyzer: A residual graph network with attention mechanism for autism spectrum disorder diagnosis. Pattern Recognition Letters, 196, 109-116. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0167865525002107) 

1. **[CAE]** Lei, B., Cai, G., Zhu, Y., Wang, T., Dong, L., Zhao, C., ... & Wang, R. (2024). Self-Supervised Multi-Scale Multi-Modal Graph Pool Transformer for Sellar Region Tumor Diagnosis. IEEE Journal of Biomedical and Health Informatics, 29(4), 2758-2771. [**[Paper]**](https://ieeexplore.ieee.org/document/10750372)

1. **[DyHG]** Chen, Y., Li, J., Zhu, L., Xu, Y., Guan, T., Shi, H., ... & Han, A. (2025). Dynamic hypergraph representation for bone metastasis analysis. Computer Methods and Programs in Biomedicine, 108966.  [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0169260725003839) [**[Code]**](https://github.com/Yuxuan-Chen7/DyHG)

1. **[TTGAU-Net]** Zhao, Z., Li, W., Ding, X., Sun, J., & Xu, L. X. (2025). TTGA U-Net: Two-stage two-stream graph attention U-Net for hepatic vessel connectivity enhancement. Computerized Medical Imaging and Graphics, 122, 102514.  [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0895611125000230) [**[Code]**](https://github.com/muxin0527/TTGAU-Net)

1. **[SAGEFusionNet]** Kumar, S., Hazarika, S., & Gupta, C. N. (2025). SAGEFusionNet: An auxiliary supervised graph neural network for brain age prediction as a neurodegenerative biomarker. Brain Sciences, 15(7), 752.  [**[Paper]**](https://www.mdpi.com/2076-3425/15/7/752) [**[Code]**](https://github.com/NeuralLabIITGuwahati/SAGEFusionNet)

1. Hao, X., Ma, M., Tao, J., Cao, J., Qin, J., Liu, F., ... & Ming, D. (2025). A hierarchical graph convolutional network with infomax-guided graph embedding for population-based ASD detection. IEEE Journal of Biomedical and Health Informatics, 29(6), 4349-4361. [**[Paper]**](https://ieeexplore.ieee.org/document/10909093) 

1. **[S2GL]** Huang, Y., Chen, G., Wen, X., Wei, L., Zhang, H., & Shen, D. (2026). Unlocking shared-specific features of multi-modal brain graphs for accurate psychiatric diagnosis. Pattern Recognition, 113997.  [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0031320326009623) [**[Code]**](https://github.com/HyinRPM/S2GL)

1. **[CDHM]** Wang, G., Wang, J., Li, Y., Ekerete, I., Zheng, H., Qiao, L., & Zhang, L. (2025, December). Cross-Attentioned Dynamic Hierarchical Representation Learning Utilizing Mamba Fusion for Brain Network Analysis. In 2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM) (pp. 5517-5524). IEEE.  [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11356013)

1. **[MH-GCAT]** Liu, Y., Chen, W., & Pan, Y. (2026). Multimodal Population Graph Networks with Self-Attention Mechanisms for Autism Spectrum Disorder Identification. IEEE Journal of Biomedical and Health Informatics.  [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11475591)

1. **[DualEdge-GAT]** Pakravan, M. (2026). Explainable graph attention network on resting state fMRI dynamic functional connectivity reveals cerebellar hub breakdown in chronic knee osteoarthritis pain. Iranian Journal of Science and Technology, Transactions of Electrical Engineering, 50(1), 289-300.  [**[Paper]**](https://doi.org/10.1007/s40998-025-01012-z) [**[Code]**](https://github.com/mnPakravan/DualEdge_GAT_fMRI_Chronic-Kneee-Osteoarthritis-Pain)

1. Lee, J. W., Shayan, H., & Jo, H. G. (2026). Graph neural network-based modeling of functional brain connectivity in autism spectrum disorder using the ABIDE-I and II datasets. Psychiatry Research: Neuroimaging, 112310.  [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0925492726001757)

1. **[ACSI-SGNN]** Yang, Z., Sheng, J., Zhang, Q., Gong, Z., Zhang, R., Wang, B., ... & Huang, R. (2026). A novel neural network model with SAGPooling graph decodes changes in cognitive trajectories. NeuroImage, 122060.  [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S1053811926003757)

1. **[GCNSP]** Su, J., Qin, J., Shen, H., Zeng, L. L., Li, B., Liu, J., ... & Hu, D. (2026). Multi‐Site Transfer Classification of Major Depressive Disorder: An fMRI Study in 3335 Subjects. Advanced Science, 13(19), e02817.  [**[Paper]**](https://onlinelibrary.wiley.com/doi/abs/10.1002/advs.202502817) [**[Code]**](https://github.com/Qin-J/Multi-site-transfer-classification-of-major-depressive-disorder)

1. **[VMoGE]** Ding, J. E., Zilverstand, A., Yang, S., Yang, A. C. C., & Liu, F. (2026). Variational Mixture of Graph Neural Experts for Alzheimer’s Disease Recognition across Frequency Bands in EEG Brain Networks. IEEE Transactions on Automation Science and Engineering.  [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11657393)

1. **[BrainOSM]** Guo, G., Wen, G., Liu, L., Song, R., Cao, P., Yang, J., & Zaiane, O. R. (2025). BrainOSM: Outlier screening for multi-view functional brain network analysis. Computer Methods and Programs in Biomedicine, 109092.  [**[Paper]**](https://linkinghub.elsevier.com/retrieve/pii/S0169260725005097)

1. **[MAMGAT-Net]** Belgaum, M. R. (2026). MAMGAT-Net: A Multimodal Graph Attention and Multi-Scale Feature Fusion Framework for Early Breast Cancer Diagnosis from Mammography and Histopathological Images. International Journal of Advanced Computer Science & Applications, 17(7), 305.  [**[Paper]**](https://openurl.ebsco.com/contentitem/gcd:196223363?sid=ebsco:plink:crawler-gcd&id=ebsco:gcd:196223363&crl=c&jrnl=2158107X) [**[Code]**](https://github.com/HyinRPM/S2GL)

1. **[TCM-GNN]** Cao, R., Hu, X., Xiao, L., Qu, G., Huo, H., Calhoun, V. D., ... & Sun, X. (2026). Cooperative Multiplex GNN for High-Grade Glioma Survival Prediction from Preoperative Multi-Modal Radiomics-Based Brain Networks. IEEE Transactions on Medical Imaging.  [**[Paper]**](https://ieeexplore.ieee.org/document/11455358) [**[Code]**](https://github.com/ZiLaoTou/TCM-GNN)

1. **[HistoPAG‑Net]** Rahman, T., Tarkhan, A., Chellappa, R., & Baras, A. S. (2026, April). HistoPAG‑Net: a multiscale position‑aware GNN for weakly supervised histopathology analysis. In Medical Imaging 2026: Digital and Computational Pathology (Vol. 13932, pp. 267-277). SPIE.  [**[Paper]**](https://doi.org/10.1117/12.3087003)

1. **[Path-HGT]** Chowdhury, S. H., Hasan, M. I., Hossen, M. T., Saad, M. S. S., Shamim, A., Chowdhury, A. A. A., & Ifty, R. A. (2026, April). Path-HGT: A Hierarchical Graph Transformer Framework for Gigapixel Pathology Image Analysis. In 2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence & Networking (QPAIN) (pp. 1-6). IEEE.  [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11545947)

### Proteomics

1. **[HIGH-PPI]** Gao, Z., Jiang, C., Zhang, J., Jiang, X., Li, L., Zhao, P., ... & Li, J. (2023). Hierarchical graph learning for protein–protein interaction. Nature Communications, 14(1), 1093. [**[Paper]**](https://doi.org/10.1038/s41467-023-36736-1) [**[Code]**](https://github.com/zqgao22/HIGH-PPI)

1. **[3DProtDTA]** Voitsitskyi, T., Stratiichuk, R., Koleiev, I., Popryho, L., Ostrovsky, Z., Henitsoi, P., ... & Starosyla, S. (2023). 3DProtDTA: a deep learning model for drug-target affinity prediction based on residue-level protein graphs. RSC advances, 13(15), 10261-10272. [**[Paper]**](https://doi.org/10.1039/D3RA00281K) [**[Code]**](https://github.com/vtarasv/3d-prot-dta.git)

1. **[SS-GNN]** Zhang, S., Jin, Y., Liu, T., Wang, Q., Zhang, Z., Zhao, S., & Shan, B. (2023). SS-GNN: a simple-structured graph neural network for affinity prediction. ACS omega, 8(25), 22496-22507. [**[Paper]**](https://pubs.acs.org/doi/full/10.1021/acsomega.3c00085) [**[Code]**]( https://github.com/xianyuco/SS-GNN)

1. **[HEAL]** Gu, Z., Luo, X., Chen, J., Deng, M., & Lai, L. (2023). Hierarchical graph transformer with contrastive learning for protein function prediction. Bioinformatics, 39(7), btad410. [**[Paper]**](https://doi.org/10.1093/bioinformatics/btad410) [**[Code]**](https://github.com/ZhonghuiGu/HEAL)


1. **[MedusaGraph]** Jiang, H., Wang, J., Cong, W., Huang, Y., Ramezani, M., Sarma, A., ... & Kandemir, M. T. (2022). Predicting protein–ligand docking structure with graph neural network. Journal of chemical information and modeling, 62(12), 2923-2932. [**[Paper]**](https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00127) [**[Code]**](https://github.com/j9650/MedusaGraph) 

1.  Wang, X., Hu, J., & Zhang, X. (2023, July). Drug-target affinity prediction based on self-attention graph pooling and mutual interaction neural network. In International Conference on Intelligent Computing (pp. 776-790). Singapore: Springer Nature Singapore. [**[Paper]**](https://link.springer.com/chapter/10.1007/978-981-99-4749-2_66) 

1. **[ProteinGCN]** Sanyal, S., Anishchenko, I., Dagar, A., Baker, D., & Talukdar, P. (2020). ProteinGCN: Protein model quality assessment using graph convolutional networks. BioRxiv, 2020-04. [**[Paper]**](https://www.biorxiv.org/content/10.1101/2020.04.06.028266v1.abstract) [**[Code]**](https://github.com/malllabiisc/ProteinGCN)

1. **[GAT-GO]** Lai, B., & Xu, J. (2022). Accurate protein function prediction via graph attention networks with predicted structure information. Briefings in Bioinformatics, 23(1), bbab502. [**[Paper]**](https://academic.oup.com/bib/article/23/1/bbab502/6457163) 

1. **[GNN-DTI]** Lim, J., Ryu, S., Park, K., Choe, Y. J., Ham, J., & Kim, W. Y. (2019). Predicting drug–target interaction using a novel graph neural network with 3D structure-embedded graph representation. Journal of chemical information and modeling, 59(9), 3981-3988. [**[Paper]**](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00387) [**[Code]**](https://github.com/jaechanglim/GNN_DTI)

1. **[Struct2GO]** Jiao, P., Wang, B., Wang, X., Liu, B., Wang, Y., & Li, J. (2023). Struct2GO: protein function prediction based on graph pooling algorithm and AlphaFold2 structure information. Bioinformatics, 39(10), btad637. [**[Paper]**](https://doi.org/10.1093/bioinformatics/btad637) [**[Code]**](https://github.com/lyjps/Struct2GO)

2. **[SIGN]** Li, S., Zhou, J., Xu, T., Huang, L., Wang, F., Xiong, H., ... & Xiong, H. (2021, August). Structure-aware interactive graph neural networks for the prediction of protein-ligand binding affinity. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (pp. 975-985). [**[Paper]**](https://doi.org/10.1145/3447548.3467311) [**[Code]**](https://github.com/agave233/SIGN) 

0. **[APMNet]** Shen, H., Zhang, Y., Zheng, C., Wang, B., & Chen, P. (2021). A Cascade graph convolutional network for predicting protein–ligand binding affinity. International journal of molecular sciences, 22(8), 4023. [**[Paper]**](https://doi.org/10.3390/ijms22084023)

0. **[Affinity-by-GNN]** Nikolaienko, T., Gurbych, O., & Druchok, M. (2022). Complex machine learning model needs complex testing: Examining predictability of molecular binding affinity by a graph neural network. Journal of Computational Chemistry, 43(10), 728-739.  [**[Paper]**](https://doi.org/10.1002/jcc.26831) [**[Code]**](https://github.com/SoftServeInc/affinity-by-GNN) 

0. **[GINet]** Réau, M., Renaud, N., Xue, L. C., & Bonvin, A. M. (2023). DeepRank-GNN: a graph neural network framework to learn patterns in protein–protein interfaces. Bioinformatics, 39(1), btac759. [**[Paper]**](https://doi.org/10.1093/bioinformatics/btac759) [**[Code]**](https://github.com/DeepRank) 

0. **[GraphBAR]** Son, J., & Kim, D. (2021). Development of a graph convolutional neural network model for efficient prediction of protein-ligand binding affinities. PloS one, 16(4), e0249404. [**[Paper]**](https://doi.org/10.1371/journal.pone.0249404) [**[Code]**](http://github.com/jtson82/graphbar) 

0. **[EGNA]** Xia, C., Feng, S. H., Xia, Y., Pan, X., & Shen, H. B. (2023). Leveraging scaffold information to predict protein–ligand binding affinity with an empirical graph neural network. Briefings in Bioinformatics, 24(1), bbac603. [**[Paper]**](https://doi.org/10.1093/bib/bbac603) [**[Code]**]( https://github.com/chunqiux/EGNA) 

0. **[GIGN]** Yang, Z., Zhong, W., Lv, Q., Dong, T., & Yu-Chian Chen, C. (2023). Geometric Interaction Graph Neural Network for Predicting Protein–Ligand Binding Affinities from 3D Structures (GIGN). The Journal of Physical Chemistry Letters, 14(8), 2020-2033. [**[Paper]**](https://doi.org/10.1021/acs.jpclett.2c03906) [**[Code]**](https://github.com/guaguabujianle/GIGN) 

0. **[GraphSite]** Shi, W., Singha, M., Pu, L., Srivastava, G., Ramanujam, J., & Brylinski, M. (2022). GraphSite: Ligand Binding Site Classification with Deep Graph Learning. Biomolecules, 12(8), 1053. [**[Paper]**](https://doi.org/10.3390/biom12081053) [**[Code]**](https://github.com/shiwentao00/Graphsite) 

0. **[IGN]** Jiang, D., Hsieh, C. Y., Wu, Z., Kang, Y., Wang, J., Wang, E., ... & Hou, T. (2021). InteractionGraphNet: a novel and efficient deep graph representation learning framework for accurate protein–ligand interaction predictions. Journal of medicinal chemistry, 64(24), 18209-18232. [**[Paper]**](https://doi.org/10.1021/acs.jmedchem.1c01830) [**[Code]**](https://github.com/zjujdj/InteractionGraphNet/tree/master) 

0. **[InteractionNet]** Cho, H., Lee, E. K., & Choi, I. S. (2020). Layer-wise relevance propagation of InteractionNet explains protein–ligand interactions at the atom level. Scientific reports, 10(1), 21155. [**[Paper]**](https://doi.org/10.1038/s41598-020-78169-6) [**[Code]**](https://github.com/blackmints/InteractionNet) 

0. **[MP-GNN]** Li, X. S., Liu, X., Lu, L., Hua, X. S., Chi, Y., & Xia, K. (2022). Multiphysical graph neural network (MP-GNN) for COVID-19 drug design. Briefings in Bioinformatics, 23(4), bbac231. [**[Paper]**](https://doi.org/10.1093/bib/bbac231) [**[Code]**]( https://github.com/Alibaba-DAMO-DrugAI/MGNN) 

0. **[PSG-BAR]** Pandey, M., Radaeva, M., Mslati, H., Garland, O., Fernandez, M., Ester, M., & Cherkasov, A. (2022). Ligand binding prediction using protein structure graphs and residual graph attention networks. Molecules, 27(16), 5114. [**[Paper]**](https://doi.org/10.3390/molecules27165114) [**[Code]**]( https://github.com/diamondspark/PSG-BAR) 

0. **[QATEN]** Zhang, P., Xia, C., & Shen, H. B. (2023). High-accuracy protein model quality assessment using attention graph neural networks. Briefings in Bioinformatics, 24(2), bbac614. [**[Paper]**](https://doi.org/10.1093/bib/bbac614) [**[Server]**]( http://www.csbio.sjtu.edu.cn/bioinf/QATEN/) [**[Code]**](https://github.com/CQ-zhang-2016/QATEN) 

0. **[SGPPI]** Huang, Y., Wuchty, S., Zhou, Y., & Zhang, Z. (2023). SGPPI: structure-aware prediction of protein–protein interactions in rigorous conditions with graph convolutional network. Briefings in Bioinformatics, 24(2), bbad020. [**[Paper]**](https://doi.org/10.1093/bib/bbad020) [**[Code]**]( https://github.com/emerson106/SGPPI) 

0. **[Graph-CNN]** Torng, W., & Altman, R. B. (2019). Graph convolutional neural networks for predicting drug-target interactions. Journal of chemical information and modeling, 59(10), 4131-4149. [**[Paper]**](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00628) 

0. **[DeepFRI]** Gligorijević, V., Renfrew, P. D., Kosciolek, T., Leman, J. K., Berenberg, D., Vatanen, T., ... & Bonneau, R. (2021). Structure-based protein function prediction using graph convolutional networks. Nature communications, 12(1), 3168. [**[Paper]**](https://doi.org/10.1038/s41467-021-23303-9) [**[Server]**](https://beta.deepfri.flatironinstitute.org/) [**[Code]**](https://github.com/flatironinstitute/DeepFRI)

0.  Hermosilla Casajús, P., Schäfer, M., Lang, M., Fackelmann, G., Vázquez Alcocer, P. P., Kozliková, B., ... & Ropinski, T. (2021). Intrinsic-extrinsic convolution and pooling for learning on 3D protein structures. In International Conference on Learning Representations, ICLR 2021: Vienna, Austria, May 04 2021 (pp. 1-16). OpenReview. net. [**[Paper]**](https://openreview.net/forum?id=l0mSUROpwY) [**[Code]**](https://github.com/phermosilla/IEConv_proteins)

0. **[MSNGO]** Wang, B., Cui, B., Chen, S., Wang, X., Wang, Y., & Li, J. (2025). MSNGO: multi-species protein function annotation based on 3D protein structure and network propagation. Bioinformatics, 41(5), btaf285. [**[Paper]**](https://doi.org/10.1093/bioinformatics/btaf285) [**[Code]**](https://github.com/blingbell/MSNGO)

0. **[MGU-PPIS]** Shao, D., Zou, Y., Ma, L., & Yi, S. (2025). Multiscale and global-local U-Net for protein-protein interaction site prediction. Computational Biology and Chemistry, 108485. [**[Paper]**](https://doi.org/10.1016/j.compbiolchem.2025.108485) [**[Code]**](http://github.com/1415647714/MGUPPIS)

0. **[PNAbind]** Sagendorf, J. M., Mitra, R., Huang, J., Chen, X. S., & Rohs, R. (2024). Structure-based prediction of protein-nucleic acid binding using graph neural networks. Biophysical Reviews, 16(3), 297-314. [**[Paper]**](https://doi.org/10.1007/s12551-024-01201-w) [**[Code]**](https://github.com/jaredsagendorf/pnabind)

0. **[MLSDTA]** Peng, X., Ouyang, C., Liu, Y., Yu, Y., Liu, J., & Chen, M. (2024). Multimodal drug target binding affinity prediction using graph local substructure. IEEE Journal of Biomedical and Health Informatics, 29(3), 1625-1634. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/10495096) [**[Code]**](https://github.com/usethisforlearn/MLSDTA) [**[Code]**](https://zenodo.org/records/10867023)

0. **[DICCA-DTA]** Uma, E., & Mala, T. (2025). DICCA-DTA: Diffusion and contextualized capsule attention guided factorized cross-pooling for drug-target affinity prediction. Computational Biology and Chemistry, 118, 108472. [**[Paper]**](https://doi.org/10.1016/j.compbiolchem.2025.108472)

0. **[DynHeter-DTA]** Li, C., & Li, G. (2025). DynHeter-DTA: dynamic heterogeneous graph representation for drug-target binding affinity prediction. International Journal of Molecular Sciences, 26(3), 1223. [**[Paper]**](https://www.mdpi.com/1422-0067/26/3/1223) [**[Code]**](https://github.com/jaredsagendorf/pnabind)

0. **[FGNN]** Sun, W., Xu, J., Zhang, W., Li, X., Zeng, Y., & Zhang, P. (2024). Funnel graph neural networks with multi-granularity cascaded fusing for protein–protein interaction prediction. Expert Systems with Applications, 257, 125030. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0957417424018979) 

0. **[ASAG-DTA]** Li, C., & Li, G. (2026). Adaptive Self-Attention Graph Pooling for Drug–Target Affinity Prediction. International Journal of Molecular Sciences, 27(13), 5861. [**[Paper]**](https://www.mdpi.com/1422-0067/27/13/5861)

0. **[RicciBind]** Li, S., Ren, C. X., Li, Y., Huang, Z., Pan, Y., Tang, M., & Yan, H. (2026). Curvature-Guided Geometric Representation for Protein-Ligand Binding Affinity Prediction. arXiv preprint arXiv:2606.14159. [**[Paper]**](http://arxiv.org/abs/2606.14159)

0. **[DegradoMap]** Cheng, B., Zhang, J., & Jin, A. (2026, June). DegradoMap: Multi-Modal Protein Representations Enable Pre-Synthesis Prediction of PROTAC Degradability. In Proceedings of the 17th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (pp. 1-10). [**[Paper]**](https://dl.acm.org/doi/10.1145/3807503.3819985) [**[Code]**](https://github.com/bryanc5864/DegradoMap)

0. **[BioPhysPool]** Gu, Y., Zhao, S., Liu, L., & Ding, Y. (2026). A Biophysical Prior-Based Hierarchical Graph Pooling Strategy for Protein Function Prediction. IEEE Transactions on Computational Biology and Bioinformatics. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/11557205)


### Metabolomics
1. Baranwal, M., Magner, A., Elvati, P., Saldinger, J., Violi, A., & Hero, A. O. (2020). A deep learning architecture for metabolic pathway prediction. Bioinformatics, 36(8), 2547-2553. [**[Paper]**](https://doi.org/10.1093/bioinformatics/btz954) [**[Code]**](https://github.com/baranwa2/MetabolicPathwayPrediction)

1. **[GMAMDA]** Hu, B., Su, Y., Tian, X., Chen, C., Chen, C., & Lv, X. (2025). GMAMDA: Predicting Metabolite–Disease Associations Based on Adaptive Hardness Negative Sampling and Adaptive Graph Multiple Convolution. Journal of Chemical Information and Modeling, 65(10), 5242-5254. [**[Paper]**](https://doi.org/10.1021/acs.jcim.5c00694) [**[Code]**](https://github.com/baranwa2/MetabolicPathwayPrediction)

### Multi-Omics
1. **[PathomicFusion]** Chen, R. J., Lu, M. Y., Wang, J., Williamson, D. F., Rodig, S. J., Lindeman, N. I., & Mahmood, F. (2020). Pathomic fusion: an integrated framework for fusing histopathology and genomic features for cancer diagnosis and prognosis. IEEE Transactions on Medical Imaging, 41(4), 757-770. [**[Paper]**](10.1109/TMI.2020.3021387) [**[Code]**]( https://github.com/mahmoodlab/PathomicFusion)

0.  Zhuo, L., Chen, Y., Song, B., Liu, Y., & Su, Y. (2022). A model for predicting ncRNA–protein interactions based on graph neural networks and community detection. Methods, 207, 74-80. [**[Paper]**](https://doi.org/10.1016/j.ymeth.2022.09.001) 

0.  Li, B., & Nabavi, S. (2024). A multimodal graph neural network framework for cancer molecular subtype classification. BMC bioinformatics, 25(1), 27. [**[Paper]**](https://doi.org/10.1186/s12859-023-05622-4)

0.  **[MO-GCN]** Wang, H., Peng, R., Huang, Y., Liang, L., Wang, W., Zhu, B., ... & Wu, K. (2025). MO-GCN: A multi-omics graph convolutional network for discriminative analysis of schizophrenia. Brain Research Bulletin, 221, 111199. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0361923025000115) 

0.  **[TriMoGCL]** Kang, C., Liu, Z., & Zhang, H. (2025). A comprehensive graph neural network method for predicting triplet motifs in disease–drug–gene interactions. Bioinformatics, 41(2), btaf023. [**[Paper]**](https://doi.org/10.1093/bioinformatics/btaf023) [**[Code]**](https://github.com/zhanglabNKU/TriMoGCL)


## Applications in Biological Computing and Molecular Modeling
### Molecular Modeling
1. **[AETP]** Pham, P. (2025). An enhancement of multi-scope topological graph pooling and representation learning with attention for molecular graph classification. Computational Biology and Chemistry, 108548. [**[Paper]**](https://doi.org/10.1016/j.compbiolchem.2025.108548) 

1. Monem, S., Abdel-Hamid, A. H., & Hassanien, A. E. (2025). Drug toxicity prediction model based on enhanced graph neural network. Computers in Biology and Medicine, 185, 109614. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0010482524016998) 

1. **[GGI-DDI]** Yu, H., Wang, J., Zhao, S. Y., Silver, O., Liu, Z., Yao, J., & Shi, J. Y. (2024). GGI-DDI: Identification for key molecular substructures by granule learning to interpret predicted drug–drug interactions. Expert Systems with Applications, 240, 122500. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0957417423030026) 

1. Liu, M., Li, C., Chen, R., Cao, D., & Zeng, X. (2024). Geometric deep learning for drug discovery. Expert Systems with Applications, 240, 122498. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0957417423030002) [**[Code]**](https://github.com/3146830058/Geometry-Deep-Learning-for-Drug-Discovery)


1. **[CardiOT]** Zhang, X., Wang, H., Du, Z., Zhuo, L., Fu, X., Cao, D., ... & Li, K. (2024). CardiOT: towards interpretable drug cardiotoxicity prediction using optimal transport and Kolmogorov--Arnold networks. IEEE Journal of Biomedical and Health Informatics, 29(3), 1759-1770. [**[Paper]**](https://ieeexplore.ieee.org/document/10772400) [**[Code]**](https://github.com/2014402680/ CCT)

1. **[MolecBioNet]** Chen, M., Zhang, M., & Qu, C. (2025, August). Towards interpretable drug-drug interaction prediction: A graph-based approach with molecular and network-level explanations. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2 (pp. 203-214). [**[Paper]**](https://dl.acm.org/doi/10.1145/3711896.3737163) [**[Code]**](https://github.com/MengjieChan/MolecBioNet)

1. **[MSAIGNN]** Li, L., Yu, H., & Wang, Z. (2025). Attention-based interpretable multiscale graph neural network for MOFs. Journal of Chemical Theory and Computation, 21(3), 1369-1381. [**[Paper]**](https://doi.org/10.1021/acs.jctc.4c01525) 

1. **[UniMatch]** Li, R., Li, M., Liu, W., Zhou, Y., Zhou, X., Yao, Y., ... & Chen, H. (2025, May). Unimatch: Universal matching from atom to task for few-shot drug discovery. In International Conference on Learning Representations (Vol. 2025, pp. 54677-54703). [**[Paper]**](https://openreview.net/forum?id=v9EjwMM55Y)  [**[Code]**](https://github.com/Lirain21/UniMatch.git)

1. **[AUGUR]** Kouroudis, I., Poonam, Misciasci, N., Mayr, F., Müller, L., Gu, Z., & Gagliardi, A. (2025). AUGUR, a flexible and efficient optimization algorithm for identification of optimal adsorption sites. npj Computational Materials, 11(1), 136. [**[Paper]**](https://www.nature.com/articles/s41524-025-01630-5) [**[Code]**](https://github.com/benkour/AUGUR)

1. **[MolPool]** Leenhouts, R. J., Larsson, T., Verhelst, S., & Vermeire, F. H. (2025). Property prediction of fuel mixtures using pooled graph neural networks. Fuel, 381, 133218. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0016236124023676) 

1. **[3DGraphX]** Liu, X., Luo, D., Gao, W., & Liu, Y. (2025, July). 3DGraphX: Explaining 3D molecular graph models via incorporating chemical priors. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1 (pp. 859-870). [**[Paper]**](https://dl.acm.org/doi/10.1145/3690624.3709302)  [**[Code]**](https://github.com/xufliu/3DGraphX)

1. **[FPPOOL]** Cui, C., Feng, S., Song, C., Yu, S., Lu, S., Chen, Y. Z., & Shen, W. (2026). Bridging molecular fingerprints and graph neural networks through knowledge-guided pooling. [**[Paper]**](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001102/v1) [**[Code]**](https://github.com/xiaocui3737/FPPooling/tree/master)


1. **[MARCC]** Astero, M., Li, A., Casiraghi, E., & Rousu, J. (2026). A cross-attentive multi-task graph learning framework for chemical reaction modeling. Bioinformatics, 42(5), btag193. [**[Paper]**](https://doi.org/10.1093/bioinformatics/btag193)  [**[Code]**](https://github.com/maryamastero/MARCC)

1. **[GraphFormer-CL]** Amiri, M., & Zare, O. (2026). Graph former-CL: A novel graph transformer with contrastive learning framework for enhanced drug-drug interaction prediction. PloS one, 21(1), e0339971. [**[Paper]**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0339971) 

1. **[MPMol]** Ji, Y., Wan, G., Zhan, Y., & Du, B. (2023). Metapath-fused heterogeneous graph network for molecular property prediction. Information Sciences, 629, 155-168. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S0020025523000907)  [**[Code]**](https://github.com/valerieJJ/MPMol)

1. **[IMPGNN]** Wang, H., Zhang, H., Zhang, A., Tang, J., & Li, P. (2026). Bond-Aware Molecular Graph Learning With Multi-Graph Interleaved Message Passing. IEEE Journal of Biomedical and Health Informatics. [**[Paper]**](https://ieeexplore.ieee.org/document/11419755)  [**[Code]**](https://github.com/xufliu/3DGraphX)


### Biological Computing
1. **[BF-GCN]** Pham, P. (2025). Li, C., Tang, T., Pan, Y., Yang, L., Zhang, S., Chen, Z., ... & Xu, P. (2024). An efficient graph learning system for emotion recognition inspired by the cognitive prior graph of eeg brain network. IEEE Transactions on Neural Networks and Learning Systems, 36(4), 7130-7144. [**[Paper]**](https://doi.org/10.1109/TNNLS.2024.3405663)  [**[Code]**](https://github.com/UESTC-BAC/BF-GCN)

1. **[PND‐Net]** Bera, A., Bhattacharjee, D., & Krejcar, O. (2024). PND-Net: plant nutrition deficiency and disease classification using graph convolutional network. Scientific Reports, 14(1), 15537. [**[Paper]**](https://doi.org/10.1038/s41598-024-66543-7) 

1. Sun, K., Peng, C., Yu, S., Han, Z., & Xia, F. (2024). From electroencephalogram data to brain networks: Graph-learning-based brain disease diagnosis. IEEE Intelligent Systems, 39(2), 21-29. [**[Paper]**](https://doi.org/10.1109/MIS.2024.3352972) 

1. **[WSGMB]** Pan, S., Jiang, X., & Zhang, K. (2024). WSGMB: weight signed graph neural network for microbial biomarker identification. Briefings in Bioinformatics, 25(1), bbad448. [**[Paper]**](https://doi.org/10.1093/bib/bbad448) [**[Code]**](https://github.com/panshuheng/WSGMB/tree/master)

1. **[DearLLM]** Xu, Y., Jiang, X., Chu, X., Qiu, R., Feng, Y., Ding, H., ... & Xie, B. (2025, April). DearLLM: Enhancing Personalized Healthcare via Large Language Models-Deduced Feature Correlations. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 1, pp. 941-949). [**[Paper]**](https://doi.org/10.1609/aaai.v39i1.32079) [**[Code]**](https://github.com/Artessay/DearLLM)

1. **[HyPoHGF]** Liu, S., Xie, G., Yang, Z., Chen, W., Wang, D., Wu, Q., ... & Yu, S. (2026). HyPoHGF: Hybrid pooling heterogeneous graph fusion for acupuncture prescription recommendation. Information Fusion, 104298. [**[Paper]**](https://www.sciencedirect.com/science/article/pii/S1566253526001776)

1. **[Casenet]** Pham, D. T., Van Nguyen, D., Tran, T. T., & Nguyen, V. A. (2026). Casenet: a novel HIV forecasting model using attention-based spatio-temporal graph network leveraging city-wide infection correlations. Neural Computing and Applications, 38(5), 119. [**[Paper]**](https://doi.org/10.1007/s00521-025-11804-3)

1. **[MGF-Net]** Yin, H., & Yin, X. (2026). Montage-guided fusion network for long-term scalp electroencephalography seizure detection. Frontiers in Neuroinformatics, 20, 1942480. [**[Paper]**](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2026.1942480/full)

1. **[JTFEE-Net]** Zhang, C., Hao, Y., Wen, X., Zhou, M., Zhang, Y., Wu, J., & Cao, R. (2026, July). Joint Time-Frequency Encoding and Emotion-Embedded Graph Neural Network for EEG Emotion Recognition. In International Conference on Intelligent Computing (pp. 272-284). Singapore: Springer Nature Singapore. [**[Paper]**](https://link.springer.com/chapter/10.1007/978-981-92-3563-6_22)

## Update
* 2024-05-20: New sections: Libraries, Genomics, Metabolomics, Multi-omics; new papers; and revised title.
* 2025-10-29: New sections: Applications in Biological Computing and Molecular Modeling, and new papers in Global Pooling, Clustering Pooling, Edge Pooling, Hybrid Pooling, Graph Unpooling, Evaluation, Review and Analysis, Radiomics, and Proteomics.
* 2026-11-09: New sections: Graph Pooling for Complex Graphs, News about our new paper, and new papers in all sections.
