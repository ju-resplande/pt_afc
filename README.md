# Portuguese Automatic Fact Checking
Repository for Paper "Portuguese Automated Fact-checking: Information Retrieval with Claim extraction" in review.



├── requirements.txt -> requieremnts for all process including notebook analysis
├── utils.py
├── 1_validation_cleaning.ipynb -> 1st step: Data Validation and Cleaning
├── 2_enrichment.ipynb -> 2nd step: Data Enrichment via CSE and Claim Extraction with Gemini 1.5 Flash
├── 3_google_factcheck.ipynb -> 3rd step: Data Enrichment via Google FactCheck
├── 4_make_splits.ipynb -> 4th step: Split data for training
├── Dockerfile -> Dockerfile for hypersearch
├── 5_hypersearch.py -> 5th step: Hypersearch using Portuguese BERT (Bertimbau)
├── 6_run_llm.py -> 6th step: Few-shot using Gemini 1.5 Flash
└── notebook_analysis
    ├── data_analysis.ipynb
    ├── experiment_analysis.ipynb
    ├── result_analysis.ipynb
    └── utils.py