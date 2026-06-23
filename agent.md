You are an expert AI/Machine Learning Engineer specializing in NLP and PyTorch/HuggingFace. Your task is to implement the winning methodology of the "!MSA Team" from the BAREC Shared Task 2025 (Arabic Readability Assessment).

I need a complete, end-to-end Python pipeline. The main entry point must be a single file named `train.py`. When I run `python train.py`, it should execute the entire process: data preprocessing, model training, ensembling, and generating a `prediction.zip` file formatted correctly for CodaBench submission.

### 1. `train.py` Structure & Configuration
At the very top of `train.py`, create a highly visible and easily modifiable `Configuration` section (using a dataclass or a config dictionary). It MUST include exactly these hyper-parameters based on the paper:
* `BATCH_SIZE = 16`
* `LEARNING_RATE = 2e-5`
* `EPOCHS = 5`
* `OPTIMIZER = "AdamW"`
* `EARLY_STOPPING_PATIENCE = 2`
* `MAX_LENGTH = 512`
* `DATA_PATHS`: Dictionary containing paths for train, dev, and test sets.
* `OUTPUT_DIR`: Directory to save checkpoints and final zip.

### 2. Data Preprocessing Pipeline
Implement a `BARECDataset` class and a preprocessing function that does the following:
1.  **Cleaning:** Use regex to remove redundant punctuation, normalize special characters, and trim extra spaces.
2.  **Morphological Tokenization:** Use the `camel_tools` library (specifically `D3TOK`) to tokenize the Arabic text while preserving morphological segments.
3.  **Class Imbalance Handling:** Calculate inverse-frequency class weights for the 19 readability levels. The formula is: `w_j = N_{total} / (19 * n_j)` where `n_j` is the number of samples in class `j`. Pass these weights to the Loss functions.

### 3. Model Architectures & Loss Functions
The ensemble relies on 4 HuggingFace base models trained with 3 different loss functions. 
* **Base Models:** 1. `aubmindlab/bert-base-arabertv2`
    2. `aubmindlab/araelectra-base-discriminator`
    3. `UBC-NLP/MARBERT`
    4. `CAMeL-Lab/bert-base-arabic-camelbert-msa`
* **Loss Functions (Implement switching logic in the model class):**
    1.  `CE` (Cross-Entropy Loss): Standard multi-class classification.
    2.  `MSE` (Mean Squared Error): Regression over the 1-19 scale.
    3.  `COR` (Conditional Ordinal Regression): Implement the CORAL (Rank-consistent ordinal regression) framework logic.

*Note: For the sake of this script, implement a training loop that iterates through these model/loss combinations, or allow the configuration block to specify which combinations to train and ensemble.*

### 4. Ensembling Strategy (Crucial)
After generating predictions from the trained models, implement a custom `Ensembler` class with the following logic:
1.  **Confidence-Weighted Averaging:** Combine predictions using `W = sum(p_i * c_i) / sum(c_i)`. 
    * For CE models: `c_i` is the max softmax probability.
    * For MSE models: `c_i` is the inverse variance (or a static high confidence heuristic if variance isn't tracked).
2.  **Rule-based Adjustment (Borderline cases):** If two models are being compared and their predicted levels differ by exactly 1 (e.g., `abs(p1 - p2) == 1`), the final prediction for that pair is `max(p1, p2)`.
3.  **Document-Level Aggregation:** The dataset provides sentence IDs. Group sentences by the first 7 characters of their ID (which represents the Document ID). The document's readability score is the MAXIMUM sentence-level prediction within that document: `R_doc = max(R_sentences)`.
4.  **Post-processing Skew Fix (The 16/17 Rule):** If ANY individual model in the ensemble predicts level 16 or 17 for a document, override the ensemble's averaged prediction and output that high label (16 or 17) instead. Also, apply `math.floor()` instead of `math.ceil()` for borderline decimal predictions to fix distribution skew.

### 5. Output Generation (CodaBench Format)
Finally, write a function that formats the final predictions. 
1. Create the required submission file (usually a `predictions.tsv` or `.json` mapping `id` to `label`).
2. Programmatically zip this file into `prediction.zip` in the root directory.
3. Print a success message: "Training complete. Submission file saved as prediction.zip".

Please write modular, clean, well-commented Python code. Provide the `train.py` code and a `requirements.txt` containing necessary libraries like `torch`, `transformers`, `camel-tools`, `scikit-learn`, `pandas`.