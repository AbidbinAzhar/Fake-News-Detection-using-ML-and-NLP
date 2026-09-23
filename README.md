
## Fake News Detection Using Machine Learning and Natural Language Processing

Fake-News-Detection-using-ML-and-NLP is a Natural Language Processing (NLP) and machine learning project designed to classify news articles as **Fake** or **Real**.

The project implements an end-to-end text classification pipeline involving exploratory data analysis, text preprocessing, TF-IDF feature extraction, supervised machine learning, model comparison, and final evaluation.

Three machine learning classifiers are trained and compared using a reproducible **70:10:20 train-validation-test split**. The model demonstrating the strongest validation performance is selected as the final classifier and evaluated on the held-out test set.

---

# Scientific Motivation

The rapid growth of online news platforms has made it increasingly difficult to distinguish reliable news from misleading or fabricated information.

Fake news can influence public perception, spread misinformation, and reduce trust in digital information sources. Automated text classification techniques provide a computational approach for identifying potentially misleading news articles based on linguistic and textual patterns.

Traditional machine learning algorithms combined with Natural Language Processing techniques can provide effective and computationally efficient approaches for text classification.

This project explores the application of:

* Text preprocessing and normalization
* Exploratory data analysis
* TF-IDF feature extraction
* Supervised machine learning
* Model comparison
* Statistical evaluation
* Error analysis

The project provides a reproducible framework for experimenting with classical machine learning approaches to fake news detection.

---

# Project Objectives

The primary objectives of this project are:

* Classify news articles as Fake or Real
* Perform exploratory analysis of the news dataset
* Clean and preprocess textual data
* Convert textual information into numerical representations using TF-IDF
* Train multiple machine learning classifiers
* Compare model performance using validation data
* Select the best-performing model
* Evaluate the final model on an unseen test dataset
* Perform error analysis to understand model limitations
* Maintain a reproducible machine learning workflow

---

# Key Features

* Exploratory Data Analysis of news articles
* Text preprocessing and normalization
* Duplicate and irrelevant data handling
* Train-validation-test split using a 70:10:20 ratio
* TF-IDF feature extraction
* Multinomial Naive Bayes classification
* Logistic Regression classification
* Linear Support Vector Machine classification
* Model comparison using multiple evaluation metrics
* Final test-set evaluation
* Error analysis
* Reproducible machine learning pipeline
* Fixed random seed for consistent experimentation

---

# Workflow

```text
                 Fake News Dataset
                        │
                        ▼
             Exploratory Data Analysis
                        │
                        ▼
                Data Preparation
                        │
                        ▼
              Text Preprocessing
                        │
                        ▼
        Train–Validation–Test Split
                  (70:10:20)
                        │
                        ▼
             TF-IDF Feature Extraction
                        │
                        ▼
                  Model Training
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
        Naive Bayes  Logistic    Linear
                    Regression     SVM
             │          │          │
             └──────────┼──────────┘
                        ▼
                Model Comparison
                        │
                        ▼
               Best Model Selection
                        │
                        ▼
              Final Test Evaluation
                        │
                        ▼
                  Error Analysis
```

---

# Dataset Characteristics

## Dataset Source
The dataset consists of separate files containing fake and real news articles.

### Dataset Files

```text
Fake.csv
True.csv
```

### Features

| Feature   | Description                     |
| --------- | ------------------------------- |
| `Title`   | Title of the news article       |
| `Text`    | Main textual content            |
| `Subject` | Subject/category of the article |
| `Date`    | Publication date                |

### Target Variable

The two source files are converted into a binary classification target:

```text
0 → Fake News
1 → Real News
```

---

# Data Preparation

The dataset preparation process includes:

* Loading the fake and real news datasets
* Assigning binary target labels
* Combining the datasets
* Examining missing values
* Checking duplicate records
* Exploring class distribution
* Combining relevant textual fields where appropriate
* Cleaning textual content
* Preparing the final dataset for machine learning

The dataset is subsequently divided into training, validation, and test subsets using a **70:10:20 split**.

---

# Text Preprocessing

The textual data is processed before feature extraction.

The preprocessing pipeline may include:

### Lowercase Conversion

Converts text into a consistent lowercase representation.

```text
Input:
Donald Trump Announces New Policy

Output:
donald trump announces new policy
```

### Punctuation and Special Character Handling

Unnecessary punctuation and special characters are removed where appropriate.

```text
Input:
Breaking News!!! Market rises sharply.

Output:
breaking news market rises sharply
```

### Whitespace Normalization

Extra spaces and unnecessary whitespace are normalized.

### Text Normalization

The cleaned text is prepared for numerical feature extraction using TF-IDF.

---

# Feature Extraction

## TF-IDF

The project uses **Term Frequency–Inverse Document Frequency (TF-IDF)** to convert news articles into numerical feature vectors.

TF-IDF assigns higher importance to words that are frequent within a particular document but relatively uncommon across the complete collection of documents.

The resulting numerical representation is used as input to the machine learning classifiers.

An important aspect of the pipeline is that the **TF-IDF vectorizer is fitted only on the training data** to prevent information leakage from the validation and test datasets.

---

# Model Selection

Three supervised machine learning classifiers are evaluated:

## 1. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classifier commonly used for text classification tasks.

It is computationally efficient and provides a strong baseline for document classification.

## 2. Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of a document belonging to a particular class.

It is widely used for high-dimensional text classification problems.

## 3. Linear Support Vector Machine

Linear SVM is a linear classification algorithm that attempts to find an optimal decision boundary between different classes.

It is particularly suitable for high-dimensional sparse feature representations such as TF-IDF.

The three models are compared using the validation dataset, and the model with the strongest validation performance is selected for final evaluation.

---

# Evaluation Metrics

The classifiers are evaluated using multiple metrics:

### Accuracy

Measures the proportion of correctly classified articles.

### Precision

Measures how many articles predicted as a particular class actually belong to that class.

### Recall

Measures how many articles belonging to a particular class are correctly identified.

### F1-Score

Provides a combined measure of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

Using multiple metrics provides a more comprehensive assessment than accuracy alone.

---

# Technologies Used

## Programming & Data Processing

* Python
* Pandas
* NumPy

## Natural Language Processing

* NLTK
* Regular Expressions (Regex)
* TF-IDF

## Machine Learning

* Scikit-learn
* Multinomial Naive Bayes
* Logistic Regression
* Linear Support Vector Machine

## Data Visualization

* Matplotlib
* Seaborn

## Development Environment

* Google Colab
* Jupyter Notebook

---

# Project Structure

```text
Fake-News-Detection-using-ML-and-NLP/
│
├── Fake_News_Detection.ipynb
├── README.md
├── requirements.txt
│
└── data/
    ├── Fake.csv
    └── True.csv
```

> **Note:** The dataset files may not be included directly in the GitHub repository if their size exceeds GitHub's standard web upload limit. In that case, the dataset location or download instructions should be documented separately.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/AbidbinAzhar/Fake-News-Detection-using-ML-and-NLP.git

cd Fake-News-Detection-using-ML-and-NLP
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

The project uses the following major Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
```

The complete dependency list is provided in:

```text
requirements.txt
```

---

# Usage

## Open the Notebook

Open:

```text
Fake_News_Detection.ipynb
```

The notebook can be executed using **Google Colab** or **Jupyter Notebook**.

## Prepare the Dataset

Place the following files in the expected dataset location:

```text
Fake.csv
True.csv
```

For the recommended repository structure:

```text
data/
├── Fake.csv
└── True.csv
```

## Run the Pipeline

Execute the notebook cells sequentially:

```text
1. Import Libraries
        ↓
2. Load Dataset
        ↓
3. Exploratory Data Analysis
        ↓
4. Data Preprocessing
        ↓
5. Train/Validation/Test Split
        ↓
6. TF-IDF Feature Extraction
        ↓
7. Model Training
        ↓
8. Validation Evaluation
        ↓
9. Model Selection
        ↓
10. Final Test Evaluation
        ↓
11. Error Analysis
```

---

# Model Comparison

The project evaluates the following models:

| Model                   | Feature Representation | Evaluation        |
| ----------------------- | ---------------------- | ----------------- |
| Multinomial Naive Bayes | TF-IDF                 | Validation & Test |
| Logistic Regression     | TF-IDF                 | Validation & Test |
| Linear SVM              | TF-IDF                 | Validation & Test |

The final classifier is selected based on validation performance rather than the test set.

---

# Results

Among the evaluated models, **Linear Support Vector Machine (Linear SVM)** achieved the strongest overall validation performance and was selected as the final classifier.

The selected model is subsequently evaluated on the previously unseen test dataset using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

> **Recommendation:** Add your actual numerical results here once your final notebook is complete. For example, you can include a table comparing all three models rather than only stating which model performed best.

### Example Results Table

| Model                   | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ----------------------- | -------: | --------: | -----: | -------: | ------: |
| Multinomial Naive Bayes |        — |         — |      — |        — |       — |
| Logistic Regression     |        — |         — |      — |        — |       — |
| Linear SVM              |        — |         — |      — |        — |       — |

---

# Error Analysis

Error analysis is performed to investigate incorrectly classified news articles.

The analysis can help identify patterns such as:

* Ambiguous news articles
* Articles containing misleading language
* Articles with limited textual information
* Satirical or sarcastic content
* Similar linguistic patterns between real and fake articles
* Classification errors caused by dataset-specific vocabulary

Understanding these errors provides insight into the limitations of traditional TF-IDF-based classifiers.

---

# Applications

Potential applications of fake news classification include:

* Automated news screening systems
* Misinformation research
* News content analysis
* Media monitoring systems
* Educational NLP applications
* Research-oriented text classification
* Social media content analysis

The predictions should be treated as automated classification outputs rather than definitive judgments about the factual accuracy of individual articles.

---

# Limitations

* The models rely primarily on textual information.
* TF-IDF does not capture contextual semantics as effectively as modern transformer-based models.
* The dataset may contain biases or patterns specific to its source.
* Performance may decrease on previously unseen writing styles or domains.
* Ambiguous and sarcastic articles can be difficult to classify.
* Traditional machine learning models may not fully understand the factual meaning of an article.
* Model performance depends on the quality and distribution of the training data.

---

# Future Work

Possible improvements include:

* Implement transformer-based models such as **BERT** and **RoBERTa**
* Experiment with advanced word and sentence embeddings
* Perform hyperparameter optimization
* Explore ensemble learning approaches
* Implement explainable AI using **SHAP** or **LIME**
* Evaluate the system on additional datasets
* Develop a web-based prediction interface using **Streamlit**
* Investigate domain adaptation and cross-domain performance
* Perform more detailed error and bias analysis

---

# Reproducibility

This project follows several practices to improve reproducibility:

* Fixed random seed (`SEED = 42`)
* Deterministic train-validation-test split
* Fixed 70:10:20 dataset split
* Documented preprocessing pipeline
* TF-IDF fitted only on training data
* Separate validation and test datasets
* Complete dependency list in `requirements.txt`
* Notebook-based implementation documenting the complete workflow

---

# Research Workflow Summary

```text
                    News Articles
                         │
                         ▼
                 Data Preparation
                         │
                         ▼
                Text Preprocessing
                         │
                         ▼
                 TF-IDF Vectorization
                         │
                         ▼
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
         Naive Bayes  Logistic    Linear SVM
                      Regression
              │          │          │
              └──────────┼──────────┘
                         ▼
                  Model Comparison
                         │
                         ▼
                 Model Selection
                         │
                         ▼
                  Test Evaluation
                         │
                         ▼
                   Error Analysis
```

---

# Conclusion

This project demonstrates an end-to-end machine learning approach for fake news classification using Natural Language Processing and traditional supervised learning algorithms.

The pipeline combines text preprocessing, TF-IDF feature extraction, multiple classification algorithms, validation-based model selection, and final test evaluation.

The project provides a reproducible foundation for studying automated news classification and can be extended using modern transformer-based NLP models and explainable machine learning techniques.

---

# Disclaimer

This project is intended for **educational and research purposes**.

The classification output of the trained models should not be interpreted as definitive verification of whether a news article is factually true or false. Machine learning predictions are dependent on the training data, preprocessing pipeline, and model limitations.

---

# Citation

If you use this project for academic, educational, or research purposes, please cite:

**Azhar, A. B. (2026). Fake News Detection Using Machine Learning and Natural Language Processing. GitHub Repository.**

