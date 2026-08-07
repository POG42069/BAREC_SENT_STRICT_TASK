# BAREC 2026 Sentence-Level Strict Track

## Overview

This repository contains our system for the [BAREC Shared Task 2026](https://barec.camel-lab.com/sharedtask2026) sentence-level Strict Track.

The system fine-tunes an Arabic Transformer regressor using D3Tok, the normalized surface sentence, text statistics, domain, and text class. It trains five independent seeds with MSE loss, selects each best checkpoint using development-set QWK, averages their raw scores, and generates a CodaBench-ready submission. Training uses only the official Train split; Dev is used for model selection, while Open/Blind Test is used only for inference.

## Steps

1. Clone the repository and install the dependencies:

   ```bash
   git clone https://github.com/POG42069/BAREC_SENT_STRICT_TASK.git
   cd BAREC_SENT_STRICT_TASK
   python -m pip install -r requirements.txt
   camel_data -i disambig-bert-unfactored-msa
   ```

2. Check the pipeline on a small subset:

   ```bash
   python train.py --smoke-test
   ```

3. Train the ensemble and create the Open Test submission:

   ```bash
   python train.py
   ```

   The final file is saved as `outputs/prediction.zip`.

4. For the private Blind Test, add the Hugging Face token as a private `HF_TOKEN` environment variable or Kaggle Secret, then run:

   ```bash
   python train_blind.py --download-only
   python train_blind.py
   ```

   The final Blind Test file is saved as `outputs/blind/prediction.zip`.
