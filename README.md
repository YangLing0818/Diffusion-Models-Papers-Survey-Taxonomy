# Diffusion Models: A Comprehensive Survey of Methods and Applications
This repo is constructed for collecting and categorizing papers about diffusion models according to our survey paper——[_**Diffusion Models: A Comprehensive Survey of Methods and Applications**_](https://arxiv.org/abs/2209.00796)

[Algorithm Taxonomy](#1)

[Application Taxonomy](#2)

[Connections with Other Generative Models](#3)

<p id="1"></p>

## Algorithm Taxonomy
### Sampling-Acceleration Enhancement
#### Discretization Optimization
[Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse
problems through stochastic contraction](https://openaccess.thecvf.com/content/CVPR2022/html/Chung_Come-Closer-Diffuse-Faster_Accelerating_Conditional_Diffusion_Models_for_Inverse_Problems_Through_Stochastic_CVPR_2022_paper.html)

[Diffusion Schrödinger bridge with applications to score-based
generative modeling](https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html)

[Score-Based Generative Modeling with Critically-Damped Langevin Diffusion](https://openreview.net/forum?id=CzceR82CYc)

[ Gotta Go Fast When Generating Data with
Score-Based Models](https://arxiv.org/abs/2105.14080)

[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)

[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)

[ DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model
Sampling in Around 10 Step](https://arxiv.org/abs/2206.00927)

[Score-Based Generative Modeling
through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS)

[Learning Fast Samplers for Diffusion Models by Differentiating
Through Sample Quality](https://openreview.net/forum?id=VFBjuF8HEp)

[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)
#### Non-Markovian Process
[Denoising Diffusion Implicit Models](https://openreview.net/forum?id=St1giarCHLP)

[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)

[ gDDIM: Generalized denoising diffusion implicit models](https://arxiv.org/abs/2206.05564)

[Learning fast samplers for diffusion models by differentiating through
sample quality](https://openreview.net/forum?id=VFBjuF8HEp)
#### Partial Sampling
[Progressive Distillation for Fast Sampling of Diffusion Models](https://openreview.net/forum?id=TIdIXIpzhoI)

[Accelerating Diffusion Models via Early Stop of the Diffusion Process](https://arxiv.org/abs/2205.12524)

[Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed](https://arxiv.org/abs/2101.02388)

[ Truncated Diffusion Probabilistic Models](https://arxiv.org/abs/2202.09671)
### Likelihood-Maximization Enhancement
#### Noise Schedule Optimization
[Variational diffusion models](https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html)

[ Improved denoising diffusion probabilistic models](https://proceedings.mlr.press/v139/nichol21a.html)
#### Learnable Reverse Variance
[Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models](https://openreview.net/forum?id=0xiJLKH-ufZ)

[ Improved denoising diffusion probabilistic models](https://proceedings.mlr.press/v139/nichol21a.html)

#### Objectives Designing
[Maximum likelihood training of score-based diffusion models](https://proceedings.neurips.cc/paper/2021/hash/0a9fdbb17feb6ccb7ec405cfb85222c4-Abstract.html)

[ Maximum Likelihood Training for Score-based Diffusion
ODEs by High Order Denoising Score Matching](https://proceedings.mlr.press/v162/lu22f.html)

[A variational perspective on diffusion-based generative models and score matching](https://proceedings.neurips.cc/paper/2021/hash/c11abfd29e4d9b4d4b566b01114d8486-Abstract.html)

[Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs
Theory](https://proceedings.neurips.cc/paper/2021/hash/c11abfd29e4d9b4d4b566b01114d8486-Abstract.html)
### Data-Generalization Enhancement
#### Feature Space Unification
[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)

[Score-based generative modeling in latent space](https://proceedings.neurips.cc/paper/2021/hash/5dca4c6b9e244d24a30b4c45601d9720-Abstract.html)

[Riemannian Score-Based Generative
Modeling](https://arxiv.org/abs/2202.02763)

[ Diffusion priors in variational autoencoders](https://orbi.uliege.be/handle/2268/262334)
#### Data-Dependent Transition Kernels
[ GeoDiff: A Geometric Diffusion Model for Molecular
Conformation Generation](https://openreview.net/forum?id=PzcvxEMzvQC)

[Permutation invariant graph generation via
score-based generative modeling](http://proceedings.mlr.press/v108/niu20a)

[Vector quantized diffusion model
for text-to-image synthesis](https://openaccess.thecvf.com/content/CVPR2022/html/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.html)

[Structured Denoising Diffusion Models in Discrete
State-Spaces](https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html)

[Vector Quantized Diffusion Model with CodeUnet for Text-to-Sign
Pose Sequences Generation](https://arxiv.org/abs/2208.09141)

<p id="2"></p>

## Application Taxonomy
### Computer Vision
### Natural Language Processing
### Multi-Modal Learning
### Molecular Graph Modeling
### Time-Series Modeling
### Adversarial Purification
### Waveform Signal Processing

<p id="3"></p>

## Connections with Other Generative Models
### Variational Autoencoder
### Generative Adversarial Network
### Normalizing Flow
### Autoregressive Models
### Energy-Based Models
## Citing
If you find this work useful, please cite our paper:
```
@article{yang2022diffusion,
  title={Diffusion Models: A Comprehensive Survey of Methods and Applications},
  author={Yang, Ling and Zhang, Zhilong and Hong, Shenda},
  journal={arXiv preprint arXiv:2209.00796},
  year={2022}
}
```
