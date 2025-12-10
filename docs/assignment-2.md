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

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-cly1{text-align:left;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-n863{background-color:#96fffb;text-align:left;vertical-align:middle}
</style>

### Logistic Regression
<table class="tg"><thead>
  <tr>
    <th class="tg-uzvj">classifier</th>
    <th class="tg-uzvj">model_config</th>
    <th class="tg-uzvj">preprocessing</th>
    <th class="tg-uzvj">accuracy</th>
    <th class="tg-uzvj">precision</th>
    <th class="tg-uzvj">recall</th>
    <th class="tg-uzvj">f1</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-cly1">Logistic Regression</td>
    <td class="tg-cly1">0.5</td>
    <td class="tg-cly1">tfidf_2500</td>
    <td class="tg-cly1">0.7721</td>
    <td class="tg-cly1">0.7911</td>
    <td class="tg-cly1">0.7635</td>
    <td class="tg-cly1">0.7771</td>
  </tr>
  <tr>
    <td class="tg-cly1">Logistic Regression</td>
    <td class="tg-cly1">0.5</td>
    <td class="tg-cly1">tfidf_5000</td>
    <td class="tg-cly1">0.7760</td>
    <td class="tg-cly1">0.7941</td>
    <td class="tg-cly1">0.7678</td>
    <td class="tg-cly1">0.7807</td>
  </tr>
  <tr>
    <td class="tg-n863">Logistic Regression</td>
    <td class="tg-n863">0.5</td>
    <td class="tg-n863">bert</td>
    <td class="tg-n863">0.7834</td>
    <td class="tg-n863">0.7866</td>
    <td class="tg-n863">0.7831</td>
    <td class="tg-n863">0.7848</td>
  </tr>
  <tr>
    <td class="tg-cly1">Logistic Regression</td>
    <td class="tg-cly1">1.0</td>
    <td class="tg-cly1">tfidf_2500</td>
    <td class="tg-cly1">0.7728</td>
    <td class="tg-cly1">0.7925</td>
    <td class="tg-cly1">0.7639</td>
    <td class="tg-cly1">0.7779</td>
  </tr>
  <tr>
    <td class="tg-cly1">Logistic Regression</td>
    <td class="tg-cly1">1.0</td>
    <td class="tg-cly1">tfidf_5000</td>
    <td class="tg-cly1">0.7778</td>
    <td class="tg-cly1">0.7957</td>
    <td class="tg-cly1">0.7695</td>
    <td class="tg-cly1">0.7824</td>
  </tr>
  <tr>
    <td class="tg-cly1">Logistic Regression</td>
    <td class="tg-cly1">1.0</td>
    <td class="tg-cly1">bert</td>
    <td class="tg-cly1">0.7831</td>
    <td class="tg-cly1">0.7865</td>
    <td class="tg-cly1">0.7826</td>
    <td class="tg-cly1">0.7846</td>
  </tr>
  <tr>
    <td class="tg-cly1">Logistic Regression</td>
    <td class="tg-cly1">2.0</td>
    <td class="tg-cly1">tfidf_2500</td>
    <td class="tg-cly1">0.7720</td>
    <td class="tg-cly1">0.7929</td>
    <td class="tg-cly1">0.7624</td>
    <td class="tg-cly1">0.7773</td>
  </tr>
  <tr>
    <td class="tg-cly1">Logistic Regression</td>
    <td class="tg-cly1">2.0</td>
    <td class="tg-cly1">tfidf_5000</td>
    <td class="tg-cly1">0.7782</td>
    <td class="tg-cly1">0.7962</td>
    <td class="tg-cly1">0.7698</td>
    <td class="tg-cly1">0.7828</td>
  </tr>
  <tr>
    <td class="tg-cly1">Logistic Regression</td>
    <td class="tg-cly1">2.0</td>
    <td class="tg-cly1">bert</td>
    <td class="tg-cly1">0.7831</td>
    <td class="tg-cly1">0.7859</td>
    <td class="tg-cly1">0.7829</td>
    <td class="tg-cly1">0.7844</td>
  </tr>
</tbody></table>

### Linear SVC
<table class="tg"><thead>
  <tr>
    <th class="tg-uzvj">classifier</th>
    <th class="tg-uzvj">model_config</th>
    <th class="tg-uzvj">preprocessing</th>
    <th class="tg-uzvj">accuracy</th>
    <th class="tg-uzvj">precision</th>
    <th class="tg-uzvj">recall</th>
    <th class="tg-uzvj">f1</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-cly1">Linear SVC</td>
    <td class="tg-cly1">0.5</td>
    <td class="tg-cly1">tfidf_2500</td>
    <td class="tg-cly1">0.7702</td>
    <td class="tg-cly1">0.7952</td>
    <td class="tg-cly1">0.7587</td>
    <td class="tg-cly1">0.7765</td>
  </tr>
  <tr>
    <td class="tg-cly1">Linear SVC</td>
    <td class="tg-cly1">0.5</td>
    <td class="tg-cly1">tfidf_5000</td>
    <td class="tg-cly1">0.7750</td>
    <td class="tg-cly1">0.7962</td>
    <td class="tg-cly1">0.7652</td>
    <td class="tg-cly1">0.7804</td>
  </tr>
  <tr>
    <td class="tg-n863">Linear SVC</td>
    <td class="tg-n863">0.5</td>
    <td class="tg-n863">bert</td>
    <td class="tg-n863">0.7839</td>
    <td class="tg-n863">0.7883</td>
    <td class="tg-n863">0.7828</td>
    <td class="tg-n863">0.7856</td>
  </tr>
  <tr>
    <td class="tg-cly1">Linear SVC</td>
    <td class="tg-cly1">1.0</td>
    <td class="tg-cly1">tfidf_2500</td>
    <td class="tg-cly1">0.7692</td>
    <td class="tg-cly1">0.7943</td>
    <td class="tg-cly1">0.7577</td>
    <td class="tg-cly1">0.7755</td>
  </tr>
  <tr>
    <td class="tg-cly1">Linear SVC</td>
    <td class="tg-cly1">1.0</td>
    <td class="tg-cly1">tfidf_5000</td>
    <td class="tg-cly1">0.7724</td>
    <td class="tg-cly1">0.7931</td>
    <td class="tg-cly1">0.7630</td>
    <td class="tg-cly1">0.7778</td>
  </tr>
  <tr>
    <td class="tg-n863">Linear SVC</td>
    <td class="tg-n863">1.0</td>
    <td class="tg-n863">bert</td>
    <td class="tg-n863">0.7839</td>
    <td class="tg-n863">0.7883</td>
    <td class="tg-n863">0.7828</td>
    <td class="tg-n863">0.7856</td>
  </tr>
  <tr>
    <td class="tg-cly1">Linear SVC</td>
    <td class="tg-cly1">2.0</td>
    <td class="tg-cly1">tfidf_2500</td>
    <td class="tg-cly1">0.7690</td>
    <td class="tg-cly1">0.7941</td>
    <td class="tg-cly1">0.7576</td>
    <td class="tg-cly1">0.7754</td>
  </tr>
  <tr>
    <td class="tg-cly1">Linear SVC</td>
    <td class="tg-cly1">2.0</td>
    <td class="tg-cly1">tfidf_5000</td>
    <td class="tg-cly1">0.7710</td>
    <td class="tg-cly1">0.7920</td>
    <td class="tg-cly1">0.7615</td>
    <td class="tg-cly1">0.7764</td>
  </tr>
  <tr>
    <td class="tg-n863">Linear SVC</td>
    <td class="tg-n863">2.0</td>
    <td class="tg-n863">bert</td>
    <td class="tg-n863">0.7839</td>
    <td class="tg-n863">0.7884</td>
    <td class="tg-n863">0.7828</td>
    <td class="tg-n863">0.7856</td>
  </tr>
</tbody></table>

### Random Forest
<table class="tg"><thead>
  <tr>
    <th class="tg-uzvj">classifier</th>
    <th class="tg-uzvj">model_config</th>
    <th class="tg-uzvj">preprocessing</th>
    <th class="tg-uzvj">accuracy</th>
    <th class="tg-uzvj">precision</th>
    <th class="tg-uzvj">recall</th>
    <th class="tg-uzvj">f1</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-cly1">Random Forest</td>
    <td class="tg-cly1">50.0</td>
    <td class="tg-cly1">svd_2500</td>
    <td class="tg-cly1">0.7056</td>
    <td class="tg-cly1">0.6891</td>
    <td class="tg-cly1">0.7145</td>
    <td class="tg-cly1">0.7016</td>
  </tr>
  <tr>
    <td class="tg-cly1">Random Forest</td>
    <td class="tg-cly1">50.0</td>
    <td class="tg-cly1">svd_5000</td>
    <td class="tg-cly1">0.7087</td>
    <td class="tg-cly1">0.6867</td>
    <td class="tg-cly1">0.7202</td>
    <td class="tg-cly1">0.7030</td>
  </tr>
  <tr>
    <td class="tg-cly1">Random Forest</td>
    <td class="tg-cly1">100.0</td>
    <td class="tg-cly1">svd_2500</td>
    <td class="tg-cly1">0.7160</td>
    <td class="tg-cly1">0.7071</td>
    <td class="tg-cly1">0.7216</td>
    <td class="tg-cly1">0.7143</td>
  </tr>
  <tr>
    <td class="tg-n863">Random Forest</td>
    <td class="tg-n863">100.0</td>
    <td class="tg-n863">svd_5000</td>
    <td class="tg-n863">0.7191</td>
    <td class="tg-n863">0.7026</td>
    <td class="tg-n863">0.7284</td>
    <td class="tg-n863">0.7152</td>
  </tr>
</tbody></table>

### XGBoost
<table class="tg"><thead>
  <tr>
    <th class="tg-uzvj">classifier</th>
    <th class="tg-uzvj">model_config</th>
    <th class="tg-uzvj">preprocessing</th>
    <th class="tg-uzvj">accuracy</th>
    <th class="tg-uzvj">precision</th>
    <th class="tg-uzvj">recall</th>
    <th class="tg-uzvj">f1</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-cly1">XGBoost</td>
    <td class="tg-cly1">50.0</td>
    <td class="tg-cly1">svd_2500</td>
    <td class="tg-cly1">0.7242</td>
    <td class="tg-cly1">0.7430</td>
    <td class="tg-cly1">0.7177</td>
    <td class="tg-cly1">0.7301</td>
  </tr>
  <tr>
    <td class="tg-cly1">XGBoost</td>
    <td class="tg-cly1">50.0</td>
    <td class="tg-cly1">svd_5000</td>
    <td class="tg-cly1">0.7282</td>
    <td class="tg-cly1">0.7383</td>
    <td class="tg-cly1">0.7253</td>
    <td class="tg-cly1">0.7318</td>
  </tr>
  <tr>
    <td class="tg-cly1">XGBoost</td>
    <td class="tg-cly1">50.0</td>
    <td class="tg-cly1">bert</td>
    <td class="tg-cly1">0.7563</td>
    <td class="tg-cly1">0.7579</td>
    <td class="tg-cly1">0.7570</td>
    <td class="tg-cly1">0.7575</td>
  </tr>
  <tr>
    <td class="tg-cly1">XGBoost</td>
    <td class="tg-cly1">100.0</td>
    <td class="tg-cly1">svd_2500</td>
    <td class="tg-cly1">0.7260</td>
    <td class="tg-cly1">0.7435</td>
    <td class="tg-cly1">0.7200</td>
    <td class="tg-cly1">0.7315</td>
  </tr>
  <tr>
    <td class="tg-cly1">XGBoost</td>
    <td class="tg-cly1">100.0</td>
    <td class="tg-cly1">svd_5000</td>
    <td class="tg-cly1">0.7314</td>
    <td class="tg-cly1">0.7432</td>
    <td class="tg-cly1">0.7277</td>
    <td class="tg-cly1">0.7354</td>
  </tr>
  <tr>
    <td class="tg-n863">XGBoost</td>
    <td class="tg-n863">100.0</td>
    <td class="tg-n863">bert</td>
    <td class="tg-n863">0.7621</td>
    <td class="tg-n863">0.7621</td>
    <td class="tg-n863">0.7636</td>
    <td class="tg-n863">0.7629</td>
  </tr>
</tbody></table>

### MLP
<table class="tg"><thead>
  <tr>
    <th class="tg-uzvj">classifier</th>
    <th class="tg-uzvj">model_config</th>
    <th class="tg-uzvj">preprocessing</th>
    <th class="tg-uzvj">accuracy</th>
    <th class="tg-uzvj">precision</th>
    <th class="tg-uzvj">recall</th>
    <th class="tg-uzvj">f1</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-n863">MLP</td>
    <td class="tg-n863">Dropout + Scheduler + Early Stopping</td>
    <td class="tg-n863">bert</td>
    <td class="tg-n863">0.7909</td>
    <td class="tg-n863">0.7738</td>
    <td class="tg-n863">0.8026</td>
    <td class="tg-n863">0.7879</td>
  </tr>
  <tr>
    <td class="tg-cly1">MLP</td>
    <td class="tg-cly1">Dropout + Scheduler + Early Stopping</td>
    <td class="tg-cly1">svd</td>
    <td class="tg-cly1">0.7437</td>
    <td class="tg-cly1">0.7524</td>
    <td class="tg-cly1">0.7411</td>
    <td class="tg-cly1">0.7467</td>
  </tr>
</tbody></table>

### Best Models
<table class="tg"><thead>
  <tr>
    <th class="tg-uzvj">classifier</th>
    <th class="tg-uzvj">model_config</th>
    <th class="tg-uzvj">preprocessing</th>
    <th class="tg-uzvj">accuracy</th>
    <th class="tg-uzvj">precision</th>
    <th class="tg-uzvj">recall</th>
    <th class="tg-uzvj">f1</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-cly1">Logistic Regression</td>
    <td class="tg-cly1">0.5</td>
    <td class="tg-cly1">bert</td>
    <td class="tg-cly1">0.7834</td>
    <td class="tg-cly1">0.7866</td>
    <td class="tg-cly1">0.7831</td>
    <td class="tg-cly1">0.7848</td>
  </tr>
  <tr>
    <td class="tg-cly1">Linear SVC</td>
    <td class="tg-cly1">2.0</td>
    <td class="tg-cly1">bert</td>
    <td class="tg-cly1">0.7839</td>
    <td class="tg-cly1">0.7884</td>
    <td class="tg-cly1">0.7828</td>
    <td class="tg-cly1">0.7856</td>
  </tr>
  <tr>
    <td class="tg-cly1">Random Forest</td>
    <td class="tg-cly1">100.0</td>
    <td class="tg-cly1">svd_5000</td>
    <td class="tg-cly1">0.7191</td>
    <td class="tg-cly1">0.7026</td>
    <td class="tg-cly1">0.7284</td>
    <td class="tg-cly1">0.7152</td>
  </tr>
  <tr>
    <td class="tg-cly1">XGBoost</td>
    <td class="tg-cly1">100.0</td>
    <td class="tg-cly1">bert</td>
    <td class="tg-cly1">0.7621</td>
    <td class="tg-cly1">0.7621</td>
    <td class="tg-cly1">0.7636</td>
    <td class="tg-cly1">0.7629</td>
  </tr>
  <tr>
    <td class="tg-n863">MLP</td>
    <td class="tg-n863">Dropout + Scheduler + Early Stopping</td>
    <td class="tg-n863">bert</td>
    <td class="tg-n863">0.7909</td>
    <td class="tg-n863">0.7738</td>
    <td class="tg-n863">0.8026</td>
    <td class="tg-n863">0.7879</td>
  </tr>
</tbody></table>

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
    