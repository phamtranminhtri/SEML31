# SEML31


Ho Chi Minh City University of Technology (HCMUT) – Vietnam National University-Ho Chi Minh City (VNU-HCMC).  
Machine Learning (CO3117) assignments, group TN01, team SEML31.

## 1. General information
- Course name: Machine Learning (CO3117).
- Semester 251, academic year: 2025-2026.
- Instructor: Lê Thành Sách, email: ltsach@hcmut.edu.vn.
- Team members:

  | Name               | Student ID | Email address                   |
  |--------------------|------------|---------------------------------|
  | Nguyễn Thái Học    | 2311100    | hoc.nguyen2311100@hcmut.edu.vn  |
  | Nguyễn Thái Sơn    | 2312968    | son.nguyenthai2023@hcmut.edu.vn |
  | Lương Minh Thuận   | 2313348    | thuan.luong1808@hcmut.edu.vn    |
  | Phạm Trần Minh Trí | 2313622    | tri.phamtranminh@hcmut.edu.vn   |

## 2. Assignment information

### Goals of this assignment
- Implement the traditional machine learning pipeline: exploratory data analysis, data preprocessing, feature extraction, model training, and evaluation.
- Apply machine learning techniques to different data types: tabular data, text, and images; understand their similarities and differences.

### How to run the notebooks
Open the notebooks via Google Colab, then click `Run All`. The whole notebook should take a few minutes to execute.

- _Requirements (taken from Google Colab default environment 2025-10-14):_ Python 3.12.12

  | package      | version     |
  |--------------|-------------|
  | numpy        | 2.0.2       |
  | pandas       | 2.2.2       |
  | scikit-learn | 1.6.1       |
  | matplotlib   | 3.10.0      |
  | seaborn      | 0.13.2      |
  | torch        | 2.8.0+cu126 |

- _Datasets:_
  - Tabular: [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
  - Text: [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140)
  - Image: [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

### Project's folder structure
- `notebooks/`: the Jupyter/Colab notebooks (`.ipynb`).
- `modules/`: the nessesary Python module files (`.py`).
- `report/`: the report files and source LaTeX code (`.pdf` and `.tex`).
- `features/`: the extracted feature vectors (`.npy` and `.h5`).

### Reports and Colab notebooks
- _Report:_ <a target="_blank" href="https://github.com/phamtranminhtri/SEML31/blob/d5b6518a542077f1c856ba698a7195eb2309bdf8/reports/additional_assignment/build/main.pdf">Please click here</a>
Report: 
- _Notebooks:_

| Assignment | Content             | Dataset                                                                                                           | Notebook source                 | Open in Google Colab                                                                                                                                                                                                               |
|------------|---------------------|-------------------------------------------------------------------------------------------------------------------|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1          | Tabular data        | [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)                        | `/notebooks/assignment-1.ipynb` | <a target="_blank" href="https://colab.research.google.com/github/phamtranminhtri/SEML31/blob/main/notebooks/assignment-1.ipynb">   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| 2          | Text data           | [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140)             | `/notebooks/assignment-2.ipynb` | <a target="_blank" href="https://colab.research.google.com/github/phamtranminhtri/SEML31/blob/main/notebooks/assignment-2.ipynb">   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| 3          | Image data          | [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) | `/notebooks/assignment-3.ipynb` | <a target="_blank" href="https://colab.research.google.com/github/phamtranminhtri/SEML31/blob/main/notebooks/assignment-3.ipynb">   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| Extension (Algorithm only)  | Hidden Markov model |  Random Generated Dataset                                                                        |                                 | <a target="_blank" href="https://colab.research.google.com/drive/1_ZlC4lpIubvzOZH-wS6uRUw16ODDYQgk?usp=sharing">   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| Extension (Applied)  | Chord Prediction | [Music_and_Labs Dataset](https://github.com/caiomiyashiro/music_and_science/tree/master/Chord%20Recognition/lab_and_musics)                                                         | `/notebooks/extended-assignment.ipynb`| <a target="_blank" href="https://colab.research.google.com/drive/1Aln8cKUV86DrpdODOCq5xshaCkAdy1pd?usp=sharing">   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
