# Portuguese Automated Fact-Checking: Data Enrichment and Analysis

[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/ju-resplande/portuguese-fact-checking)
[![Paper](https://img.shields.io/badge/📖%20Paper-OpenReview-blue)](https://openreview.net/forum?id=MCAqDwCZAP)
[![Poster](https://img.shields.io/badge/🖼️%20Poster-PDF-red)](docs/poster.pdf)

This repository contains the official Python code and resources for the research paper: **"Portuguese Automated Fact-checking: Information Retrieval with Claim extraction"**.

Our work addresses the gap in Portuguese Automated Fact-Checking (AFC) by introducing a novel pipeline to systematically **enrich misinformation datasets with external web evidence** 🔎. We simulate user information-seeking behavior, leverage **Large Language Models (LLMs) for core claim extraction** 🤖, and apply a **semi-automated validation framework** 🧹 to enhance overall dataset reliability and verifiability.

## ✨ Key Features

-   **Systematic Data Enrichment**: Augments datasets by retrieving relevant evidence from the web using the Google Custom Search Engine (CSE) API.
-   **LLM-Powered Claim Extraction**: Utilizes Gemini 1.5 Flash to distill noisy text into concise, verifiable claims.
-   **Semi-Automated Validation**: Employs the Google FactCheck API and near-duplicate detection to ensure data quality and integrity.
-   **Reproducible Splits**: Provides standardized train, validation, and test splits for consistent experimentation.
-   **Model Benchmarking**: Includes scripts for hyperparameter tuning (BERTimbau) and few-shot evaluation (Gemini 1.5 Flash).

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   [Docker](https://www.docker.com/get-started) (Optional, for hyperparameter search)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    Create a `.env` file in the root directory of the project and add your API keys. This file is ignored by Git to keep your keys secure.
    ```env
    # .env
    SEARCH_ID="your_google_cse_id"
    SEARCH_KEY="your_google_api_key"
    GEMINI_API_KEY="your_gemini_api_key"
    ```

4.  **Build Docker Image (Optional):**
    If you plan to run the hyperparameter search, build the Docker image:
    ```bash
    docker build -t fact-checking-env .
    ```

## ⚙️ Running the Pipeline

The project is structured as a sequence of notebooks and scripts that should be run in order. The core workflow is visualized below:

`Data Cleaning` ➡️ `Web Evidence Enrichment` ➡️ `Claim Extraction` ➡️ `FactCheck Validation` ➡️ `Data Splitting` ➡️ `Model Experiments`

---

**1. `1_validation_cleaning.ipynb`**

Performs initial data loading, cleaning, and near-duplicate detection using the [Akin](https://github.com/jules-gom/akin) library to ensure a clean starting point.

<p align="center">
  <img width="500" alt="Near-duplicate detection process" src="https://github.com/user-attachments/assets/e12a0986-adb8-4257-b1ab-d9b5339fa28e" />
</p>

**2. `2_enrichment.ipynb`**

Implements the data enrichment process. This step queries the Google CSE API to find external evidence and then uses Gemini 1.5 Flash to extract the core claim from the original text.

<p align="center">
  <img width="612" height="141" alt="Data enrichment and claim extraction flow" src="https://github.com/user-attachments/assets/b222fa84-4cf2-41a4-949d-90a63d8151ae" />
</p>

**3. `3_google_factcheck.ipynb`**

Further enriches and validates the dataset by querying the Google FactCheck Claim Search API, cross-referencing claims with existing fact-checks.

**4. `4_make_splits.ipynb`**

Generates standardized `train`, `validation`, and `test` splits from the enriched datasets to ensure reproducible model training and evaluation.

**5. `5_hypersearch.py`**

Conducts a hyperparameter search to find the optimal settings for fine-tuning the **Bertimbau** model on our new, enriched dataset.

**6. `6_run_llm.py`**

Executes few-shot learning experiments using the **Gemini 1.5 Flash** LLM to evaluate its fact-checking capabilities on the prepared data.

---

### 📊 Analysis Notebooks

The `notebook_analysis/` directory contains additional Jupyter notebooks for in-depth analysis of the datasets, experimental procedures, and results presented in our paper.

## 📖 Citation

If you use this code or the enriched datasets in your research, please cite our paper:

```bibtex
@inproceedings{
  gomes2025portuguese,
  title={Portuguese Automated Fact-checking: Information Retrieval with Claim Extraction},
  author={Juliana Gomes and Eduardo Garcia and Arlindo R. Galv{\~a}o Filho},
  booktitle={Proceedings of the Eighth Workshop on Fact Extraction and VERification},
  year={2025},
  url={https://openreview.net/forum?id=MCAqDwCZAP}
}
```
