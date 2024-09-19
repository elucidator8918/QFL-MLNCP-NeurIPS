# QFed+FHE: Quantum Federated Learning with Secure Fully Homomorphic Encryption (FHE)



## Overview

Welcome to the Quantum Federated Learning (QFL) repository, which utilizes Secure Fully Homomorphic Encryption (FHE). This initiative seeks to advance privacy-preserving multimodal machine learning in quantum environments.

Quantum Federated Learning - QFL combines the principles of federated learning with the computational power of quantum computing. In a federated learning setup, multiple decentralized devices collaboratively train a model without sharing their local data. Each device trains the model on its own data and only shares the model updates. MQFL takes this a step further by leveraging quantum computing to handle the complex computations involved in the training process, thereby improving efficiency and potentially unlocking new capabilities in machine learning.

Secure Fully Homomorphic Encryption - Secure Fully Homomorphic Encryption (FHE) is a form of encryption that allows computations to be performed on encrypted data without needing to decrypt it first. This ensures that sensitive data remains private and secure even while being processed. By integrating FHE into the federated learning framework, we can ensure that model updates are encrypted throughout the entire process, maintaining privacy and security across all participants.

Technical Paper Link - https://arxiv.org/abs/2409.11430

<!-- 
The comprehensive results section can be seen:

| **Model**                                                                                  | **Num Rounds** | **Num Clients** | **Dataset Used**        | **FHE Activated** | **Central Validation Accuracy** | **Loss** | **Training Accuracy** | **Simulation Time**        |
|--------------------------------------------------------------------------------------------|---------------|-----------------|-------------------------|-------------------|-------------------------------|---------|-----------------------|---------------------------|
| [FHE-FedQNN-CIFAR](https://github.com/elucidator8918/QFL-MLNCP-NeurIPS/blob/main/src/FHE_FedQNN_CIFAR.ipynb) | 20            | 10              | CIFAR-10                | Yes               | 70.12%                        | 1.24    | 99.1%                 | 9389.21 sec (156.5 min)   |
| [FHE-FedQNN-MRI](https://github.com/elucidator8918/QFL-MLNCP-NeurIPS/blob/main/src/FHE_FedQNN_MRI.ipynb)   | 20            | 10              | Brain MRI Scan          | Yes               | 88.75%                        | 0.36    | 99.6%                 | 7309.01 sec (121.8 min)   |
| [Standard-FedQNN-CIFAR](https://github.com/elucidator8918/QFL-MLNCP-NeurIPS/blob/main/src/Standard_FedQNN_CIFAR.ipynb) | 20            | 10              | CIFAR-10                | No                | 72.16%                        | 1.202   | 97.15%                | 9090.41 sec (151.5 min)   |
| [Standard-FedQNN-MRI](https://github.com/elucidator8918/QFL-MLNCP-NeurIPS/blob/main/src/Standard_FedQNN_MRI.ipynb)   | 20            | 10              | Brain MRI Scan          | No                | 89.71%                        | 0.338   | 100%                  | 7537.57 sec (125.6 min)   | -->

## Repository Structure

```
.
├── src/
│   ├── utils/
│   ├── FHE_FedQNN_CIFAR.ipynb
│   ├── FHE_FedQNN_MRI.ipynb
│   ├── Standard_FedQNN_CIFAR.ipynb
│   └── Standard_FedQNN_MRI.ipynb
├── run-cpu.sh
├── run-gpu.sh
├── .gitignore
└── README.md
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/elucidator8918/QFL-MLNCP-NeurIPS.git
cd QFL-MLNCP-NeurIPS
```

### Install Dependencies

#### For CPU

```bash
conda create -n fed python=3.10.12 anaconda
conda init
conda activate fed
bash run-cpu.sh
```

#### For GPU

```bash
conda create -n fed python=3.10.12 anaconda
conda init
conda activate fed
bash run-gpu.sh
```

### Running Experiments

Choose the appropriate notebook based on your dataset and encryption preference:

#### FHE-enabled Quantum Federated Learning

- **CIFAR-10 Dataset:**
  - Notebook: `src/FHE_FedQNN_CIFAR.ipynb`
  - Description: CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is widely used for image classification tasks.

- **MRI Scan Dataset:**
  - Notebook: `src/FHE_FedQNN_MRI.ipynb`
  - Description: This dataset contains MRI scans used for medical image analysis, particularly for detecting and diagnosing conditions based on scan data.  

#### Standard Quantum Federated Learning

- **CIFAR-10 Dataset:**
  - Notebook: `src/Standard_FedQNN_CIFAR.ipynb`
  - Description: The same CIFAR-10 dataset, utilized without the FHE layer, for benchmarking and comparison.

- **MRI Scan Dataset:**
  - Notebook: `src/Standard_FedQNN_MRI.ipynb`
  - Description: The same MRI scan dataset, used without FHE to evaluate the performance of standard federated learning models.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact

For inquiries, please reach out here:
- forsomethingnewsid@gmail.com
- pavana.karanth17@gmail.com