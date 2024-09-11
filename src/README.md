# Multimodal Quantum Federated Learning with Fully Homomorphic Encryption (FHE)

This directory contains notebooks and scripts for running experiments on different datasets using Quantum Federated Learning models, with and without Fully Homomorphic Encryption (FHE). Our approach enhances data privacy and security while leveraging the computational advantages of quantum neural networks.

## Running Experiments

Choose the appropriate notebook based on your dataset and encryption preference:

### FHE-enabled Quantum Federated Learning

- **CIFAR-10 Dataset:**
  - **Notebook:** `FHE_FedQNN_CIFAR.ipynb`
  - **Description:** The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is widely used for image classification tasks.

- **DNA Sequence Dataset:**
  - **Notebook:** `FHE_FedQNN_DNA.ipynb`
  - **Description:** This dataset includes DNA sequences used for various biological and genetic studies, focusing on sequence classification and pattern recognition.

- **MRI Scan Dataset:**
  - **Notebook:** `FHE_FedQNN_MRI.ipynb`
  - **Description:** This dataset contains MRI scans used for medical image analysis, particularly for detecting and diagnosing conditions based on scan data.

- **PCOS Dataset:**
  - **Notebook:** `FHE_FedQNN_PCOS.ipynb`
  - **Description:** This dataset is used for analyzing Polycystic Ovary Syndrome (PCOS). It is employed for developing and assessing models aimed at detecting and diagnosing PCOS based on this data.

- **RAVDESS Multimodal Dataset:**
  - **Notebook:** `FHE_FedQNN_MMF.ipynb`
  - **Description:** This dataset includes audio and visual recordings from the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) project. It is used for multimodal emotion recognition and analysis, leveraging both audio and video data to develop and evaluate models for detecting and interpreting emotional expressions.

- **DNA+MRI Multimodal Dataset:**
  - **Notebook:** `FHE_FedQNN_DNA+MRI.ipynb`
  - **Description:** It is used as a MoE with Multimodaility leveraging both DNA Sequence and MRI scans data to develop and evaluate models for detecting and interpreting tumors and dna classes.

### Standard Quantum Federated Learning

- **CIFAR-10 Dataset:**
  - **Notebook:** `Standard_FedQNN_CIFAR.ipynb`
  - **Description:** The same CIFAR-10 dataset, utilized without the FHE layer, for benchmarking and comparison.

- **DNA Sequence Dataset:**
  - **Notebook:** `Standard_FedQNN_DNA.ipynb`
  - **Description:** The same DNA sequence dataset, used without FHE for standard federated learning experiments.

- **MRI Scan Dataset:**
  - **Notebook:** `Standard_FedQNN_MRI.ipynb`
  - **Description:** The same MRI scan dataset, used without FHE to evaluate the performance of standard federated learning models.

- **PCOS Dataset:**
  - **Notebook:** `Standard_FedQNN_PCOS.ipynb`
  - **Description:** The same PCOS dataset, used without FHE to evaluate the performance of standard federated learning models.

- **RAVDESS Multimodal Dataset:**
  - **Notebook:** `Standard_FedQNN_MMF.ipynb`
  - **Description:** The same RAVDESS dataset, used without FHE to evaluate the performance of standard federated learning models.    

- **DNA+MRI Multimodal Dataset:**
  - **Notebook:** `Standard_FedQNN_DNA+MRI.ipynb`
  - **Description:** It is used as a MoE with Multimodaility leveraging both DNA Sequence and MRI scans data to develop and evaluate models for detecting and interpreting tumors and dna classes.

## Datasets

Download the datasets using the following commands:

```bash
# DNA Sequence Dataset
kaggle datasets download -d nageshsingh/dna-sequence-dataset
mkdir -p data/DNA
unzip dna-sequence-dataset.zip -d data/DNA
rm dna-sequence-dataset.zip

# MRI Scan Dataset
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
mkdir -p data/MRI
unzip brain-tumor-mri-dataset.zip -d data/MRI
rm brain-tumor-mri-dataset.zip
```

## Datasets Multimodal

For multimodal experiment of DNA+MRI, just proceed to download the above 2 datasets.

For multimodal experiment of RAVDESS/MMF, you will need to use the RAVDESS dataset. Follow these steps to download and prepare the dataset or just use the pickle I have created ready to be used:

0. **Directly Use the Preprocessed Pickle File**
   - You can use the preprocessed RAVDESS data pickle file that has already been created for convenience

```bash
wget -P data/MMF https://github.com/elucidator8918/MISC/raw/main/Audio_Vision_RAVDESS.pkl
```

**OR**

1. **Download the Dataset:**
   - Access and download the dataset from [Zenodo](https://zenodo.org/record/1188976).

2. **Extract Action Units:**
   - To extract action units from the dataset, you will need to install OpenFace. Follow the installation instructions provided in the [OpenFace GitHub repository](https://github.com/TadasBaltrusaitis/OpenFace).

3. **Create the Pickle File:**
   - Use the script located in `utils/dataset_MMF.py` to process the data and create the pickle file necessary for your experiments.

```bash
# Example command to install OpenFace (follow the repository instructions for detailed steps)
git clone https://github.com/TadasBaltrusaitis/OpenFace
cd OpenFace
# Follow installation steps in the repository

# After installation, use the provided script
python utils/dataset_MMF.py
```