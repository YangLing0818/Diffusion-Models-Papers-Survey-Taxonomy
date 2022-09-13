# Diffusion Models: A Comprehensive Survey of Methods and Applications
This repo is constructed for collecting and categorizing papers about diffusion models according to our survey paper——[_**Diffusion Models: A Comprehensive Survey of Methods and Applications**_](https://arxiv.org/abs/2209.00796)

[Algorithm Taxonomy](#1)
* [Sampling-Acceleration Enhancement](#1.1)
  - [Discretization Optimization](#1.1.1)
  - [Non-Markovian Process](#1.1.2)
  - [Partial Sampling](#1.1.3)
* [Likelihood-Maximization Enhancement](#1.2)
  - [Noise Schedule Optimization](#1.2.1)
  - [Learnable Reverse Variance](#1.2.2)
  - [Objectives Designing](#1.2.3)
* [Data-Generalization Enhancement](#1.3)
  - [Feature Space Unification](#1.3.1)
  - [Data-Dependent Transition Kernels](#1.3.2)

[Application Taxonomy](#2)
* [Computer Vision](#2.1)
* [Natural Language Processing](#2.2)
* [Multi-Modal Learning](#2.3)
* [Molecular Graph Modeling](#2.4)
* [Time-Series Modeling](#2.5)
* [Adversarial Purification](#2.6)
* [Waveform Signal Processing](#2.7)

[Connections with Other Generative Models](#3)
* [Variational Autoencoder](#3.1)
* [Generative Adversarial Network](#3.2)
* [Normalizing Flow](#3.3)
* [Autoregressive Models](#3.5)
* [Energy-Based Models](#3.6)

<p id="1"></p>

## Algorithm Taxonomy
<p id="1.1"></p>

### 1. Sampling-Acceleration Enhancement
<p id="1.1.1"></p>

#### 1.1. Discretization Optimization
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
<p id="1.1.2"></p>

#### 1.2. Non-Markovian Process
[Denoising Diffusion Implicit Models](https://openreview.net/forum?id=St1giarCHLP)

[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)

[ gDDIM: Generalized denoising diffusion implicit models](https://arxiv.org/abs/2206.05564)

[Learning fast samplers for diffusion models by differentiating through
sample quality](https://openreview.net/forum?id=VFBjuF8HEp)
<p id="1.1.3"></p>

#### 1.3. Partial Sampling
[Progressive Distillation for Fast Sampling of Diffusion Models](https://openreview.net/forum?id=TIdIXIpzhoI)

[Accelerating Diffusion Models via Early Stop of the Diffusion Process](https://arxiv.org/abs/2205.12524)

[Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed](https://arxiv.org/abs/2101.02388)

[ Truncated Diffusion Probabilistic Models](https://arxiv.org/abs/2202.09671)
<p id="1.2"></p>

### 2. Likelihood-Maximization Enhancement
<p id="1.2.1"></p>

#### 2.1. Noise Schedule Optimization
[Variational diffusion models](https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html)

[ Improved denoising diffusion probabilistic models](https://proceedings.mlr.press/v139/nichol21a.html)
<p id="1.2.2"></p>

#### 2.2. Learnable Reverse Variance
[Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models](https://openreview.net/forum?id=0xiJLKH-ufZ)

[ Improved denoising diffusion probabilistic models](https://proceedings.mlr.press/v139/nichol21a.html)
<p id="1.2.3"></p>

#### 2.3. Objectives Designing
[Maximum likelihood training of score-based diffusion models](https://proceedings.neurips.cc/paper/2021/hash/0a9fdbb17feb6ccb7ec405cfb85222c4-Abstract.html)

[ Maximum Likelihood Training for Score-based Diffusion
ODEs by High Order Denoising Score Matching](https://proceedings.mlr.press/v162/lu22f.html)

[A variational perspective on diffusion-based generative models and score matching](https://proceedings.neurips.cc/paper/2021/hash/c11abfd29e4d9b4d4b566b01114d8486-Abstract.html)

[Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs
Theory](https://proceedings.neurips.cc/paper/2021/hash/c11abfd29e4d9b4d4b566b01114d8486-Abstract.html)
<p id="1.3"></p>

### 3. Data-Generalization Enhancement
<p id="1.3.1"></p>

#### 3.1. Feature Space Unification
[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)

[Score-based generative modeling in latent space](https://proceedings.neurips.cc/paper/2021/hash/5dca4c6b9e244d24a30b4c45601d9720-Abstract.html)

[Riemannian Score-Based Generative
Modeling](https://arxiv.org/abs/2202.02763)

[ Diffusion priors in variational autoencoders](https://orbi.uliege.be/handle/2268/262334)
<p id="1.3.2"></p>

#### 3.2. Data-Dependent Transition Kernels
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
<p id="2.1"></p>

### 1. Computer Vision
<p id="2.2"></p>

### 2. Natural Language Processing
<p id="2.3"></p>

### 3. Multi-Modal Learning
<p id="2.4"></p>

### 4. Molecular Graph Modeling
<p id="2.5"></p>

### 5. Time-Series Modeling
<p id="2.6"></p>

### 6. Adversarial Purification
<p id="2.7"></p>

### 7. Waveform Signal Processing

<p id="3"></p>

## Connections with Other Generative Models
<p id="3.1"></p>

### 1. Variational Autoencoder
<p id="3.2"></p>

### 2. Generative Adversarial Network
<p id="3.3"></p>

### 3. Normalizing Flow
<p id="3.4"></p>

### 4. Autoregressive Models
<p id="3.5"></p>

### 5. Energy-Based Models
## Citing
If you find this work useful, please cite our paper:
```
@article{yang2022diffusion,
  title={Diffusion Models: A Comprehensive Survey of Methods and Applications},
  author={Yang, Ling and Zhang, Zhilong and Hong, Shenda and Xu, Runsheng and Zhao, Yue and Shao Yingxia and Yang, Ming-Hsuan and Zhang, Wentao and Cui, Bin},
  journal={arXiv preprint arXiv:2209.00796},
  year={2022}
}
```
