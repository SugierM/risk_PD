# PD Model (Probability of Default) - LendingClub

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)  

---

## Project Description

This project presents an end-to-end credit risk modeling framework for estimating the **Probability of Default (PD)** of retail borrowers.

The solution covers the full modeling lifecycle, including:
- data preprocessing,
- variable selection and transformation (e.g., WoE/IV),
- model training and validation,
- performance evaluation (AUC, Gini),
- and interpretation of results (!WORK IN PROGRESS!).

The model was developed for educational and portfolio purposes using historical LendingClub data, with a focus on real-world credit risk practices.

### Main Business Objective

To develop a robust credit scoring model that quantifies borrower default risk and supports credit decision-making.

---

### RAG Module – Regulatory Variable Validation (Work in Progress)

This project additionally includes an experimental **Retrieval-Augmented Generation (RAG)** module.

### Objective

The goal of this module is to support **regulatory and policy compliance checks** by answering questions about whether specific variables can be used in credit risk modeling.

The system is designed to:
- Ingest regulatory or policy documents (PDF format)
- Index and embed document content
- Retrieve relevant fragments
- Generate contextual answers to questions such as:
  - *"Can marital status be used in the PD model?"*
  - *"Is age allowed as a predictive feature under the provided regulation?"*
  - *"Are demographic variables restricted?"*


### Current Status

This module is **currently under development**.

Planned improvements:
- To answer that I will need to experiment more.

### Business Value

In real-world credit risk environments, regulatory compliance is as important as predictive power.

This module aims to simulate:
- Model governance support
- Variable approval workflow
- AI-assisted regulatory interpretation


### Possible Risks

...

---

## Data
More in [Methodology](METHODOLOGY.md)
The project uses the publicly available **LendingClub** dataset (available, among others, on Kaggle). The dataset contains loan data issued between [2007.06–2018.12].

- **Target Variable:** Loans with status *"Charged Off"* or *"Default"*.
- **Number of Observations:** 2260668 in total.
- **Number of Features:** 145 original features

---

## Technologies and Tools (Not completed)

- **Language:** Python 
- **Data Processing:** ...
- **Visualization:** ...
- **Machine Learning:** ...

---

## Repository Structure

The project is divided into logical modules to separate exploratory analysis from execution:
Currently undergoing refactoring for better consolidation (ETA: 1-2 days).

- **`data/`** – Folder intended for datasets.  
  *(Note: Datasets are not tracked in the repository due to size. Download instructions are provided below.)*

- **`notebooks/`** – Exploratory analysis notebooks (Jupyter Notebooks).  
  Contains full EDA, visualizations, and feature transformation experiments (e.g., WoE and IV calculations).

- **`src/`** – Main source code directory.  
  Refactored and structured Python scripts organized into functions and classes.

- **`models/`** – Directory where trained model objects (`.pth` and `.joblib`) are saved, ready for inference on new data.

- **`reports/`** – Automatically generated reports, visualizations (e.g., ROC curve, probability distribution), and evaluation tables.

---

## How to Run the Project

The model-building process is fully automated. To reproduce results from raw data to the final trained model, follow the steps below:

### 1. Environment Setup

NOT YET IN A PLACE I WANT IT TO BE.

---

## Model Evaluation

- **AUC-ROC:** 0.6830 
- **Gini Coefficient:** 0.366
- **KS Statistic:** -

---