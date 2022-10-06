# Diffusion Models: A Comprehensive Survey of Methods and Applications
This repo is constructed for collecting and categorizing papers about diffusion models according to our survey paper——[_**Diffusion Models: A Comprehensive Survey of Methods and Applications**_](https://arxiv.org/abs/2209.00796)
# Overview
<div aligncenter><img width="800" alt="image" src="https://user-images.githubusercontent.com/62683396/193984590-e424d030-b6cd-49d4-b4d1-d9d4e4d13a27.png">


# Catalogue
## [Algorithm Taxonomy](#1)
### [Sampling-Acceleration Enhancement](#1.1)
  - [Learning-Free Sampling](#1.1.1)
  - [Learning-Based Sampling](#1.1.2)
### [Likelihood-Maximization Enhancement](#1.2)
  - [Noise Schedule Optimization](#1.2.1)
  - [Learnable Reverse Variance](#1.2.2)
  - [Continuous-Time VLB](#1.2.3)
  - [Exact Log likelihood](#1.2.4)
### [Data-Generalization Enhancement](#1.3)
  - [Manifold Structures](#1.3.1)
  - [Data with Invariant Structures](#1.3.2)
  - [Discrete Data](#1.3.3)

## [Application Taxonomy](#2)
* [Computer Vision](#2.1)
  - [Image Super Resolution, Inpainting and Translation](#2.1.1)
  - [Semantic Segementation](#2.1.2)
  - [Video Generation](#2.1.3)
  - [Point Cloud Completion and Generation](#2.1.4)
  - [Anomaly Detection](#2.1.5)
* [Natural Language Processing](#2.2)
* [Temporal Data Modeling](#2.3)
  - [Time-Series Imputation](#2.3.1)
  - [Time-Seires Forecasting](#2.3.2)
  - [Waveform Signal Processing](#2.3.3)
* [Multi-Modal Learning](#2.4)
  - [Text-to-Image Generation](#2.4.1)
  - [Text-to-Aaudio Generation](#2.4.2)
* [Robust Learning](#2.5)
* [Molecular Graph Modeling](#2.6)
* [Material Design](#2.7)
* [Inverse Problem Solving (Medical Imaging)](#2.8)



## [Connections with Other Generative Models](#3)
* [Variational Autoencoder](#3.1)
* [Generative Adversarial Network](#3.2)
* [Normalizing Flow](#3.3)
* [Autoregressive Models](#3.4)
* [Energy-Based Models](#3.5)

<p id="1"></p>

## Algorithm Taxonomy
<p id="1.1"></p >

### 1. Sampling-Acceleration Enhancement
<p id="1.1.1"></p >

#### 1.1.1 Learning-Free Sampling
<p id="1.1.1.1"></p >

##### 1.1.1.1 SDE Solver

[Score-Based Generative Modeling
through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS)

[Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse
problems through stochastic contraction](https://openaccess.thecvf.com/content/CVPR2022/html/Chung_Come-Closer-Diffuse-Faster_Accelerating_Conditional_Diffusion_Models_for_Inverse_Problems_Through_Stochastic_CVPR_2022_paper.html)


[Score-Based Generative Modeling with Critically-Damped Langevin Diffusion](https://openreview.net/forum?id=CzceR82CYc)

[ Gotta Go Fast When Generating Data with
Score-Based Models](https://arxiv.org/abs/2105.14080)

[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
<p id="1.1.1.2"></p >

##### 1.1.2 ODE Solver
[Denoising Diffusion Implicit Models](https://openreview.net/forum?id=St1giarCHLP)

[ gDDIM: Generalized denoising diffusion implicit models](https://arxiv.org/abs/2206.05564)

[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)


[ DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model
Sampling in Around 10 Step](https://arxiv.org/abs/2206.00927)


[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)
<p id="1.1.2"></p >

#### 1.2 Learning-Based Sampling
<p id="1.1.2.1"></p >

##### 1.2.1 Dynamic Programming
[Learning to Efficiently Sample from Diffusion Probabilistic Models](https://arxiv.org/abs/2106.03802)
<p id="1.1.2.2"></p >

##### 1.2.2 Knowledge Distillation
[Progressive Distillation for Fast Sampling of Diffusion Models](https://openreview.net/forum?id=TIdIXIpzhoI)

[Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed](https://arxiv.org/abs/2101.02388)
<p id="1.1.2.3"></p >

##### 1.2.3 Early Stopping
[Accelerating Diffusion Models via Early Stop of the Diffusion Process](https://arxiv.org/abs/2205.12524)

[ Truncated Diffusion Probabilistic Models](https://arxiv.org/abs/2202.09671)
<p id="1.2"></p >

### 2. Likelihood-Maximization Enhancement
<p id="1.2.1"></p >

#### 2.1. Noise Schedule Optimization
<p id="1.2.1.1"></p >

##### 2.1.1 Deterministic Schedule

[ Improved denoising diffusion probabilistic models](https://proceedings.mlr.press/v139/nichol21a.html)
<p id="1.2.1.2"></p >

##### 2.1.2 Learnable Schedule
[Variational diffusion models](https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html)
<p id="1.2.2"></p >

#### 2.2. Learnable Reverse Variance
[Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models](https://openreview.net/forum?id=0xiJLKH-ufZ)

[ Improved denoising diffusion probabilistic models](https://proceedings.mlr.press/v139/nichol21a.html)
<p id="1.2.3"></p >

#### 2.3. Continuous-Time VLB
[Maximum likelihood training of score-based diffusion models](https://proceedings.neurips.cc/paper/2021/hash/0a9fdbb17feb6ccb7ec405cfb85222c4-Abstract.html)

[A variational perspective on diffusion-based generative models and score matching](https://proceedings.neurips.cc/paper/2021/hash/c11abfd29e4d9b4d4b566b01114d8486-Abstract.html)
<p id="1.2.4"></p >

#### 2.4 Exact Log likelihood
[Score-Based Generative Modeling
through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS)

[ Maximum Likelihood Training for Score-based Diffusion
ODEs by High Order Denoising Score Matching](https://proceedings.mlr.press/v162/lu22f.html)
<p id="1.3"></p >

### 3. Data-Generalization Enhancement
<p id="1.3.1"></p >

#### 3.1. Manifold Structures
<p id="1.3.1.1"></p >

##### 3.1.1 Mapping to Manifolds
[Score-based generative modeling in latent space](https://proceedings.neurips.cc/paper/2021/hash/5dca4c6b9e244d24a30b4c45601d9720-Abstract.html)

[ Diffusion priors in variational autoencoders](https://orbi.uliege.be/handle/2268/262334)
<p id="1.3.1.2"></p >

##### 3.1.2 Diffusion on Manifolds
[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)


[Riemannian Score-Based Generative
Modeling](https://arxiv.org/abs/2202.02763)

<p id="1.3.2"></p >

#### 3.2. Data with Invariant Structures
[ GeoDiff: A Geometric Diffusion Model for Molecular
Conformation Generation](https://openreview.net/forum?id=PzcvxEMzvQC)

[Permutation invariant graph generation via
score-based generative modeling](http://proceedings.mlr.press/v108/niu20a)

[Score-based Generative Modeling of Graphs via
the System of Stochastic Differential Equations](https://proceedings.mlr.press/v162/jo22a.html)
<p id="1.3.3"></p >

#### 3.3 Discrete Data
[Vector quantized diffusion model
for text-to-image synthesis](https://openaccess.thecvf.com/content/CVPR2022/html/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.html)

[Structured Denoising Diffusion Models in Discrete
State-Spaces](https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html)

[Vector Quantized Diffusion Model with CodeUnet for Text-to-Sign
Pose Sequences Generation](https://arxiv.org/abs/2208.09141)

[Deep Unsupervised Learning using Nonequilibrium
Thermodynamics.](https://openreview.net/forum?id=rkbVIoZdWH)

<p id="2"></p>

## Application Taxonomy
<p id="2.1"></p>

### 1. Computer Vision
<p id="2.1.1"></p >

  - Image Super Resolution, Inpainting and Translation

    - [SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models](https://www.sciencedirect.com/science/article/pii/S0925231222000522)

    - [Image Super-Resolution via Iterative Refinement](https://openreview.net/forum?id=y4N8y8ZQ4c1)

    - [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)

    - [Repaint: Inpainting using denoising diffusion probabilistic models.](https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html)

    - [Palette: Image-to-image diffusion models.](https://openreview.net/forum?id=FPGs276lUeq)
    - [Generative Visual Prompt: Unifying Distributional Control of Pre-Trained Generative Models](http://arxiv.org/abs/2209.06970)
    - [Cascaded Diffusion Models for High Fidelity Image Generation.](https://www.jmlr.org/papers/v23/21-0635.html)
    - [Conditional image generation with score-based diffusion models](https://arxiv.org/abs/2111.13606)
    - [Unsupervised Medical Image Translation with Adversarial Diffusion Models](https://arxiv.org/abs/2207.08208)
    - [Score-based diffusion models for accelerated MRI](https://www.sciencedirect.com/science/article/pii/S1361841522001268)
    - [Solving Inverse Problems in Medical Imaging with Score-Based Generative Models](https://openreview.net/forum?id=vaRCHVj0uGI)
    - [MR Image Denoising and Super-Resolution Using Regularized Reverse Diffusion](https://arxiv.org/abs/2203.12621)
    - [Sdedit: Guided image synthesis and editing with stochastic differential equations](https://arxiv.org/abs/2108.01073)
<p id="2.1.2"></p >

  - Semantic Segmentation
    - [ Label-Efficient Semantic Segmentation with Diffusion Models.](https://openreview.net/forum?id=SlxSY2UZQT)
    - [Decoder Denoising Pretraining for Semantic Segmentation.](https://arxiv.org/abs/2205.11423)
<p id="2.1.3"></p >

  - Video Generation
    - [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495)
    - [Video diffusion models](https://openreview.net/forum?id=BBelR2NdDZ5)
    - [Diffusion probabilistic modeling for video generation](https://arxiv.org/abs/2203.09481)
    - [MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model.](https://arxiv.org/abs/2208.15001)
<p id="2.1.4"></p >

  - Point Cloud Completion and Generation
    - [3d shape generation and completion through point-voxel diffusion](https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_3D_Shape_Generation_and_Completion_Through_Point-Voxel_Diffusion_ICCV_2021_paper.html)
    - [Diffusion probabilistic models for 3d point cloud generation](https://openaccess.thecvf.com/content/CVPR2021/html/Luo_Diffusion_Probabilistic_Models_for_3D_Point_Cloud_Generation_CVPR_2021_paper.html)
    - [A Conditional Point Diffusion-Refinement Paradigm for 3D Point Cloud Completion](https://openreview.net/forum?id=wqD6TfbYkrn)
    - [Let us Build Bridges: Understanding and Extending Diffusion Generative Models.](https://arxiv.org/abs/2208.14699)
<p id="2.1.5"></p >

  - Anomaly Detection
    - [AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.html)
    - [Remote Sensing Change Detection (Segmentation) using Denoising Diffusion Probabilistic Models.](https://ui.adsabs.harvard.edu/abs/2022arXiv220611892G/abstract)

<p id="2.2"></p>

### 2. Natural Language Processing
  - [Structured denoising diffusion models in discrete state-spaces](https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html)
  - [Diffusion-LM Improves Controllable Text Generation.](https://arxiv.org/abs/2205.14217)
  - [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/abs/2208.04202)
 
<p id="2.3"></p>

### 3. Temporal Data Modeling
<p id="2.3.1"></p >

  - Time Series Imputation
    - [CSDI: Conditional score-based diffusion models for probabilistic time series imputation](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html)
    - [Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models](https://arxiv.org/abs/2208.09399)
    - [ Neural Markov Controlled SDE: Stochastic Optimization for Continuous-Time Data](https://openreview.net/forum?id=7DI6op61AY)
<p id="2.3.2"></p >

  - Time Series Forecasting
    - [Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting](http://proceedings.mlr.press/v139/rasul21a.html)
    - [Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models](https://arxiv.org/abs/2208.09399)
<p id="2.3.3"></p >

  - Waveform Signal Processing
    - [WaveGrad: Estimating Gradients for Waveform Generation. ](https://openreview.net/forum?id=NsMLjcFaO8O)
    - [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://openreview.net/forum?id=a-xFK8Ymz5J)  


<p id="2.4"></p>

### 4. Multi-Modal Learning
<p id="2.4.1"></p >

  - Text-to-Image Generation
    - [Blended diffusion for text-driven editing of natural images](https://openaccess.thecvf.com/content/CVPR2022/html/Avrahami_Blended_Diffusion_for_Text-Driven_Editing_of_Natural_Images_CVPR_2022_paper.html)
    - [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125)
    - [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)
    - [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)
    - [Vector quantized diffusion model for text-to-image synthesis. ](https://openaccess.thecvf.com/content/CVPR2022/html/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.html)
    - [Frido: Feature Pyramid Diffusion for Complex Scene Image Synthesis.](https://arxiv.org/abs/2208.13753)
[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)
<p id="2.4.2"></p >

  - Text-to-Audio Generation
    - [Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://proceedings.mlr.press/v139/popov21a.html)
    - [Guided-TTS 2: A Diffusion Model for High-quality Adaptive Text-to-Speech with Untranscribed Data](https://arxiv.org/abs/2205.15370)
    - [Diffsound: Discrete Diffusion Model for Text-to-sound Generation](https://arxiv.org/abs/2207.09983)
    - [ItôTTS and ItôWave: Linear Stochastic Differential Equation Is All You Need For Audio Generation](https://ui.adsabs.harvard.edu/abs/2021arXiv210507583W/abstract)
    - [Zero-Shot Voice Conditioning for Denoising Diffusion TTS Models](https://arxiv.org/abs/2206.02246)
    - [EdiTTS: Score-based Editing for Controllable Text-to-Speech.](https://arxiv.org/abs/2110.02584)
    - [ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech.](https://arxiv.org/abs/2207.06389)
<p id="2.5"></p>

### 5. Robust Learning
  - [Diffusion Models for Adversarial Purification](https://arxiv.org/abs/2205.07460)
  - [Adversarial purification with score-based generative models](http://proceedings.mlr.press/v139/yoon21a.html)
  - [Threat Model-Agnostic Adversarial Defense using Diffusion Models](https://arxiv.org/abs/2207.08089)
  - [Guided Diffusion Model for Adversarial Purification](https://arxiv.org/abs/2205.14969)
  - [Guided Diffusion Model for Adversarial Purification from Random Noise](https://arxiv.org/abs/2206.10875)
  - [PointDP: Diffusion-driven Purification against Adversarial Attacks on 3D Point Cloud Recognition.](https://arxiv.org/abs/2208.09801)
<p id="2.6"></p>

### 6. Molecular Graph Modeling
  - [Torsional Diffusion for Molecular Conformer Generation.](https://openreview.net/forum?id=D9IxPlXPJJS)
  - [Equivariant Diffusion for Molecule Generation in 3D](https://proceedings.mlr.press/v162/hoogeboom22a.html)
  - [Protein Structure and Sequence Generation with Equivariant Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2205.15019)
  - [GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation](https://openreview.net/forum?id=PzcvxEMzvQC)
  - [Diffusion probabilistic modeling of protein backbones in 3D for the motif-scaffolding problem](https://arxiv.org/abs/2206.04119)
  - [Diffusion-based Molecule Generation with Informative Prior Bridge](https://arxiv.org/abs/2209.00865)
  - [Learning gradient fields for molecular conformation generation](http://proceedings.mlr.press/v139/shi21b.html)
  - [Predicting molecular conformation via dynamic graph score matching. ](https://proceedings.neurips.cc/paper/2021/hash/a45a1d12ee0fb7f1f872ab91da18f899-Abstract.html)
<p id="2.7"></p>

### 7. Material Design
  - [Crystal Diffusion Variational Autoencoder for Periodic Material Generation](https://arxiv.org/abs/2110.06197)
  - [Antigen-specific antibody design and optimization with diffusion-based generative models](https://www.biorxiv.org/content/10.1101/2022.07.10.499510v1)
<p id="2.8"></p>

### 8. Inverse Problem Solving (Medical Imaging)
  - [Solving Inverse Problems in Medical Imaging with Score-Based Generative Models](https://openreview.net/forum?id=vaRCHVj0uGI)
  - [MR Image Denoising and Super-Resolution Using Regularized Reverse Diffusion](https://arxiv.org/abs/2203.12621)
  - [Score-based diffusion models for accelerated MRI](https://arxiv.org/abs/2110.05243)



<p id="3"></p>

## Connections with Other Generative Models
<p id="3.1"></p>

### 1. Variational Autoencoder
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)
- [A variational perspective on diffusion-based generative models and score matching](https://proceedings.neurips.cc/paper/2021/hash/c11abfd29e4d9b4d4b566b01114d8486-Abstract.html)
- [Score-based generative modeling in latent space](https://proceedings.neurips.cc/paper/2021/hash/5dca4c6b9e244d24a30b4c45601d9720-Abstract.html)
<p id="3.2"></p>

### 2. Generative Adversarial Network
  - [Diffusion-GAN: Training GANs with Diffusion. ](https://arxiv.org/abs/2206.02262)
  - [Tackling the generative learning trilemma with denoising diffusion gans](https://openreview.net/forum?id=JprM0p-q0Co)
<p id="3.3"></p>

### 3. Normalizing Flow
  - [Diffusion Normalizing Flow](https://proceedings.neurips.cc/paper/2021/hash/876f1f9954de0aa402d91bb988d12cd4-Abstract.html)
  - [Interpreting diffusion score matching using normalizing flow](https://openreview.net/forum?id=jxsmOXCDv9l)
<p id="3.4"></p>

### 4. Autoregressive Models
  - [Autoregressive Diffusion Models. ](https://openreview.net/forum?id=Lm8T39vLDTE)
  - [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting. ](http://proceedings.mlr.press/v139/rasul21a.html)
<p id="3.5"></p>

### 5. Energy-Based Models
  - [Learning Energy-Based Models by Diffusion Recovery Likelihood](https://openreview.net/forum?id=v_1Soh8QUNc)
  - [Latent Diffusion Energy-Based Model for Interpretable Text Modeling](https://proceedings.mlr.press/v162/yu22h.html)
## Citing
If you find this work useful, please cite our paper:
```
@article{Yang2022DiffusionMA,
  title={Diffusion Models: A Comprehensive Survey of Methods and Applications},
  author={Yang, Ling and Zhang, Zhilong and Hong, Shenda},
  journal={arXiv preprint arXiv:2209.00796},
  year={2022}
}
```
or 
```
@article{Yang2022DiffusionMA,
  title={Diffusion Models: A Comprehensive Survey of Methods and Applications},
  author={Ling Yang and Zhilong Zhang and Yang Song and Shenda Hong and Runsheng Xu and Yue Zhao and Yingxia Shao and Wentao Zhang and Bin Cui and Ming-Hsuan Yang},
  journal={arXiv preprint arXiv:2209.00796},
  year={2022}
}
```
