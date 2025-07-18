# Portuguese Automated Fact-Checking: Data Enrichment and Analysis

This repository contains the Python code and resources for the research paper: **"Portuguese Automated Fact-checking: Information Retrieval with Claim extraction"**.

Our work focuses on addressing the gap in Portuguese Automated Fact-Checking (AFC) by systematically **enriching misinformation datasets with external web evidence** 🔎. We simulate user information-seeking behavior, leverage **Large Language Models (LLMs) for core claim extraction** 🤖, and apply a **semi-automated validation framework** 🧹 to enhance dataset reliability.

- **Dataset**: https://huggingface.co/datasets/ju-resplande/portuguese-fact-checking
- **Poster**: [poster.pdf](poster.pdf)

##  Installation
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **API Keys:**
    This project requires API keys in .env for:
    - `SEARCH_ID= [enable search mecanism and set id for Google CSE]`
    - `SEARCH_KEY= [user search key for Google CSE and Google FactCheck]`
    - `GEMINI_API_KEY=[Gemini api key]`

3.  **Docker (for hyperparameter search):**
    If you wish to run the hyperparameter search using Docker:
    ```bash
    docker build -t  .
    ```



## Running the Pipeline

The project is structured as a sequence of Jupyter notebooks and Python scripts:

1.  **`1_validation_cleaning.ipynb`**: Performs initial data loading, cleaning, and near-duplicate detection using the Akin library.


<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/e12a0986-adb8-4257-b1ab-d9b5339fa28e" />
</p>

2.  **`2_enrichment.ipynb`**: Implements the data enrichment process using Google CSE API and LLM-based claim extraction (Gemini 1.5 Flash).
<p align="center">
<img width="612" height="141" alt="image" src="https://github.com/user-attachments/assets/b222fa84-4cf2-41a4-949d-90a63d8151ae" />
</p>

3.  **`3_google_factcheck.ipynb`**: Further enriches and validates data by querying the Google FactCheck Claim Search API.
4.  **`4_make_splits.ipynb`**: Generates standardized train/validation/test splits for the datasets.
5.  **`5_hypersearch.py`**: Conducts hyperparameter search for fine-tuning the Bertimbau model on the prepared datasets.
6.  **`6_run_llm.py`**: Executes few-shot learning experiments using the Gemini 1.5 Flash LLM.

The `notebook_analysis/` directory contains notebooks for detailed analysis of the datasets, experimental procedures, and results.

## Citation

If you use this code or the enriched datasets in your research, please cite our paper (details will be updated upon publication):

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
