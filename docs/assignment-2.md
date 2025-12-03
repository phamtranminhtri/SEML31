# Assignment 2: Text Data
**Group TN01 - Team SEML31**  

> Colab Notebook: [Assignment 2](https://colab.research.google.com/github/phamtranminhtri/SEML31/blob/main/notebooks/assignment-2.ipynb)

## 1. Overview

- **Dataset:** [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) containing 1.6 million tweets.
- **Objective:** Predict the sentiment (binary classification) of a user based on tweet text.
- **Labels:** Although the original dataset specification lists three labels (positive, neutral, negative), the actual data contains only two: **Negative (0)** and **Positive (4)**.

## 2. Exploratory Data Analysis (EDA)

- **Balance:** The dataset is perfectly balanced with 800k samples for each class.
- **Word Count Distribution:** The text length by word count follows a right-skewed distribution (Mode < Median < Mean), typical for natural language. ![word_dist](./image/Assignment%202/word_dist.png)
- **Character Count Distribution:** The character count distribution is appeared to be **bimodal**. It features a broad peak around 40-50 characters and a sharp, narrow spike at 140 characters. This spike directly reflects the historic Twitter character limit, indicating many users maximized their message length. ![char_dist](./image/Assignment%202/char_dist.png)
- **Word Frequency:** The most frequent words are common English stop words (*i, to, the, a*), which carry little sentiment value on their own.

## 3. Preprocessing

We applied the following pipeline to clean the raw text:
1.  **Lowercase:** Normalize all text to lowercase.
2.  **Noise Removal:** Remove URLs, usernames (e.g., @user), and special characters/punctuation.
3.  **Stop Word Removal:** Applied only for TF-IDF models to reduce noise.
4.  **Lemmatization:** Convert words to their base forms (e.g., *learning* $\to$ *learn*) to consolidate vocabulary.

## 4. Feature Extraction

Due to the dataset size, we utilized a stratified subset of **100,000 samples** for training and testing. We employed two distinct embedding strategies:

1.  **TF-IDF (Bag of Words):** Focuses on word frequency. We removed stop words to prioritize meaningful content.
    *   *Variants:* Max features (2500 vs 5000) and Dimensionality Reduction (SVD 300).
2.  **BERT (Contextual Embeddings):** A pre-trained transformer model. We **kept stop words** because BERT relies on sentence structure and context to generate accurate embeddings (768 dimensions).

**Feature Configurations:**

| ID | Feature Type | Dimensions | Preprocessing |
|:---|:---|:---|:---|
| 1 | TF-IDF | 2500 | No stop words |
| 2 | TF-IDF | 5000 | No stop words |
| 3 | TF-IDF + SVD | 2500 -> 300 | No stop words |
| 4 | TF-IDF + SVD | 5000 -> 300 | No stop words |
| 5 | BERT | 768 | Keep stop words |

## 5. Modeling Strategy

We evaluated **30 models** across three categories using Accuracy, Recall, Precision, and F1-score.

### A. Linear Models
*   **Algorithms:** Logistic Regression, LinearSVC.
*   **Hyperparameters:** Regularization parameter $C \in \{0.5, 1, 2\}$.
*   **Input:** TF-IDF (2500/5000) and BERT. SVD was skipped for linear models as they handle high-dimensional sparse data well.

### B. Tree-based Models
*   **Algorithms:** RandomForest, XGBoost.
*   **Hyperparameters:** $n\_estimators \in \{50, 100\}$.
*   **Input:** TF-IDF (with SVD 300) and BERT. SVD is crucial here to reduce sparsity for tree splits.

### C. Neural Networks (MLP)
*   **Architecture:**
    *   *BERT:* Hidden layers [256, 32].
    *   *TF-IDF (SVD):* Hidden layers [128, 32].
*   **Training:** Adam optimizer, Early Stopping (patience=5), Dropout.

## 6. Results & Discussion

![Logistic Regression](./image/Assignment%202/logistic.png)
![Linear SVC](./image/Assignment%202/SVC.png)
![Random Forest](./image/Assignment%202/Random_Forest.png)
![XGBoost](./image/Assignment%202/XGBoost.png)
![MLP](./image/Assignment%202/MLP.png)
![Best Models](./image/Assignment%202/Best_models.png)

## 7. Key Insights

1.  **BERT Supremacy:**
    BERT embeddings consistently outperformed TF-IDF across almost all classifiers. This confirms that capturing **context and semantic meaning** (bidirectional) is superior to simple keyword frequency (TF-IDF) for sentiment analysis, especially in short, informal texts like tweets.

2.  **Effectiveness of Linear Models:**
    Surprisingly, Linear Models (Logistic Regression/SVC) performed very competitively with TF-IDF features.
    *   *Insight:* High-dimensional sparse text data is often linearly separable.
    *   *Regularization:* Lower $C$ values (0.5) yielded slightly better accuracy (~0.5% gain), suggesting that stronger regularization helps generalize better on noisy social media data.

3.  **Tree-based Model Limitations:**
    Random Forest and XGBoost showed only marginal improvements when increasing estimators from 50 to 100 (~1% gain). Given the high computational cost, the return on investment for scaling up tree models on this specific task is low compared to using a better embedding (BERT).

4.  **MLP Performance:**
    The MLP trained on BERT embeddings achieved the highest overall metrics. The combination of dense, rich semantic vectors from BERT and the non-linear learning capability of the MLP allowed the model to capture subtle sentiment nuances that linear models missed.
    