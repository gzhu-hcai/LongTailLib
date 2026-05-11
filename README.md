# FedLTLib: A Comprehensive Benchmark for Federated Long-Tail Learning

## Introduction

**FedLTLib** is an open\-source, unified benchmark platform designed specifically for **federated long\-tail learning** scenarios\. It aims to address the ubiquitous class imbalance and non\-independent and identically distributed \(non\-IID\) data distribution challenges in real\-world federated learning systems\.

This library provides standardized pipelines for generating diverse federated long\-tail datasets, as well as integrated implementations of state\-of\-the\-art \(SOTA\) federated long\-tail learning algorithms\. FedLTLib enables fair, reproducible, and systematic empirical evaluation for federated long\-tail learning research, eliminating the inconsistency of experimental settings across different studies\.

## Key Features

- **Standardized Long\-Tail Federated Dataset Generation**: Supports global/local long\-tail distribution construction, configurable imbalance factors, and multiple non\-IID partition strategies \(Dirichlet, Pathological, Extended Dirichlet\)\.

- **Diverse Non\-IID Configuration**: Covers client\-level data quantity balance/imbalance, adjustable Dirichlet alpha parameters, and flexible client number settings to simulate real industrial and edge computing scenarios\.

- **Integrated SOTA Algorithm Suite**: Integrates 9 mainstream federated long\-tail learning algorithms, supporting multiple ResNet backbones and unified training configuration\.

- **Reproducible Experimental Pipeline**: Unified parameter naming and running commands, ensuring consistent and reproducible benchmark results for academic research\.

## Environment Requirements

This project is developed based on Python and PyTorch\. Please configure the following basic environment for stable operation:

- Python >= 3.8

- PyTorch >= 1.10

- torchvision

- numpy

- matplotlib

- scipy

## Federated Long\-Tail Dataset Generation

FedLTLib supports flexible customization of federated long\-tail datasets via configurable hyperparameters\. We take the CIFAR\-10 dataset as the default benchmark dataset\.

### Parameter Definition

|Parameter|Description|
|---|---|
|`iid / noniid`|Data partition mode: Independent and Identically Distributed / Non\-Independent and Identically Distributed|
|`\- / balance`|Client data volume setting: Imbalanced client data quantity / Balanced client data quantity|
|`dir / pat / exdir`|Non\-IID partition strategy: Standard Dirichlet / Pathological Non\-IID / Extended Dirichlet \(enhanced non\-IID\)|
|`longtail`|Enable class long\-tail distribution construction|
|`global / local`|Long\-tail distribution scope: Global long\-tail / Local client\-level long\-tail|
|`IF`|Imbalance Factor, controlling the degree of class distribution imbalance|
|`alpha`|Dirichlet distribution parameter, controlling the degree of data heterogeneity among clients|
|`Client Number`|Total number of federated learning clients|

### Dataset Generation Command

The following command generates a **global long\-tail, Dirichlet\-based non\-IID, client\-imbalanced** CIFAR\-10 dataset:

```Plain Text
python dataset/generate_Cifar10.py noniid - dir longtail global 50 0.5 20
```

*Configuration Explanation: IF=50, Dirichlet alpha=0\.5, total client number=20*

## Supported Federated Long\-Tail Learning Algorithms

FedLTLib integrates 9 mainstream SOTA federated long\-tail learning methods with unified training pipelines\. All experiments support customizable backbone networks and global training rounds\.

### Model Training Commands

All training tasks adopt the unified dataset generated above \(Cifar10\-IF50\-α0\.5\-global\-NC20 by default\)\. The core training parameters include dataset path, algorithm type, backbone network, global training rounds, and device ID\.

#### 1\. CReFF

```Plain Text
python main.py -data Cifar10-IF50-α0.5-global-NC20 -algo CREFF -m ResNet8 -gr 200 -did 0
```

#### 2\. CLIP2FL

```Plain Text
python main.py -data Cifar10-IF50-α0.5-global-NC20 -m ResNet8 -algo CLIP2FL -gr 200 -did 0
```

#### 3\. CCVR

```Plain Text
python main.py -data Cifar10-IF50-α0.5-global-NC20 -algo CCVR -m resnet8 -gr 200 -did 0
```

#### 4\. RUCR

```Plain Text
python main.py -data Cifar10-IF50-α0.5-global-NC20 -algo RUCR -m resnet8 -gr 200 -did 0
```

#### 5\. FedETF

```Plain Text
python main.py -data Cifar10-IF50-α0.5-global-NC20 -m resnet20 -algo fedetf -gr 200 -did 0
```

#### 6\. FedLoGe

```Plain Text
python main.py -data Cifar10-IF50-α0.5-global-NC40 -algo fedloge -m resnet18 -gr 200 -did 0
```

#### 7\. FedNH

```Plain Text
python main.py -data Cifar10-IF50-α0.5-global-NC100 -algo fednh -m resnet18 -gr 200 -did 0
```

#### 8\. FedIC

```Plain Text
python main.py -data Cifar10-IF50-α0.5-global-NC20 -algo fedic -m resnet8 -gr 200 -did 0
```

#### 9\. FedGraB

```Plain Text
python main.py -data Cifar10-IF50-α0.5-global-NC40 -algo fedgrab -m resnet18 -gr 200 -did 0
```

## Parameter Instruction

- `\-data`: Specify the path and configuration of the generated federated long\-tail dataset

- `\-algo`: Specify the federated long\-tail learning algorithm to be trained

- `\-m`: Specify the backbone network \(ResNet8/ResNet18/ResNet20\)

- `\-gr`: Set the number of global federated training rounds

- `\-did`: Specify the GPU device ID for training

## Citation

If you use FedLTLib for your research, please star this repository and cite our project in your publications\.

## Contribution \&amp; Support

We welcome all forms of contributions, including new algorithm integration, dataset expansion, code optimization, and bug fixes\. For questions and technical support, please submit an Issue or initiate a Pull Request\.

## License

This project is open\-source under the **MIT License**\. Free for academic and non\-commercial use\.


