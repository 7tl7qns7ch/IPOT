## Inducing Point Operator Transformer: A Flexible and Scalable Architecture for Solving PDEs (AAAI 2024)

Implementation for [Inducing Point Operator Transformer: A Flexible and Scalable Architecture for Solving PDEs](https://arxiv.org/abs/2312.10975), accepted at AAAI 2024. 

## Abstract

Solving partial differential equations (PDEs) by learning the solution operators has emerged as an attractive alternative to traditional numerical methods. However, implementing such architectures presents two main challenges: flexibility in handling irregular and arbitrary input and output formats and scalability to large discretizations. Most existing architectures are limited by their desired structure or infeasible to scale large inputs and outputs. To address these issues, we introduce an attention-based model called an inducing point operator transformer (IPOT). Inspired by inducing points methods, IPOT is designed to handle any input function and output query while capturing global interactions in a computationally efficient way. By detaching the inputs/outputs discretizations from the processor with a smaller latent bottleneck, IPOT offers flexibility in processing arbitrary discretizations and scales linearly with the size of inputs/outputs. Our experimental results demonstrate that IPOT achieves strong performances with manageable computational complexity on an extensive range of PDE benchmarks and real-world weather forecasting scenarios, compared to state-of-the-art methods.

## Poster
[AAAI2024_poster.pdf](https://github.com/7tl7qns7ch/IPOT/files/13869461/AAAI2024_poster.pdf)

## Citation
```
@article{lee2024ipot,
  title={Inducing Point Operator Transformer: A Flexible and Scalable Architecture for Solving PDEs},
  author={Seungjun Lee and Taeil Oh},
  journal={The 38th AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
