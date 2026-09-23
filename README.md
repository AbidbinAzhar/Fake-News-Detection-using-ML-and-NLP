# Fake-News-Detection-using-ML-and-NLP



                               Fake News Detection using Machine Learning and NLP

Overview: 
	This project implements a machine learning pipeline for detecting fake news articles using Natural 
	Language Processing (NLP). The workflow includes exploratory data analysis, text preprocessing, 
	TF-IDF feature extraction, model comparison, and evaluation. Three machine learning models are trained 
	and compared, with the best-performing model selected based on validation performance.


Project Objective:
	The objective of this project is to classify news articles as Real or Fake using machine learning 
	techniques. 
	The implementation follows the required 70:10:20 train-validation-test split and emphasizes 
	reproducibility, proper experimentation, and model evaluation.

Dataset:
	The project uses the Fake News Detection dataset provided with the assignment.

	Files-
		Fake.csv
		True.csv

	Features-
		Title
		Text
		Subject
		Date

	Target Labels-
		0 → Fake News
		1 → Real News


Methodology:

			    Dataset
 				  │
   				  ▼
			Exploratory Data Analysis
    			  │
   				  ▼
			  Data Preprocessing
   				  │
   				  ▼
		  Train–Validation–Test Split (70:10:20)
   				  │				  
   				  ▼
			TF-IDF Feature Extraction
   				  │
   				  ▼
			   Model Training
  				  │
  				  ▼
			   Model Evaluation
  				  │
  				  ▼
		   	 Best Model Selection
   				  │
  				  ▼
			 Final Test Evaluation
   				  │
  				  ▼
			    Error Analysis


Models Evaluated:
    The following classifiers were trained and compared-
	    1. Multinomial Naive Bayes
	    2. Logistic Regression
	    3. Linear Support Vector Machine (Linear SVM)
    The final model was selected based on validation performance.


Evaluation Metrics:
	The models were evaluated using-
					Accuracy
					Precision
					Recall
					F1-Score
					ROC-AUC

Technologies Used:
		Programming - Python
		Libraries:
			Pandas
			NumPy
			Matplotlib
			Seaborn
			Scikit-learn
			NLTK
			Regular Expressions (Regex)


Project Structure:
			FirstName_LastName/
			│
			├── Fake_News_Detection.ipynb
			├── README.md
			├── requirements.txt
			├── Fake.csv
			└── True.csv


Installation:	
	Install the required libraries- pip install -r requirements.txt


Running the Project:
	1. Place Fake.csv and True.csv in the project directory.
	2. Open Fake_News_Detection.ipynb.
	3. Run all notebook cells from top to bottom.

Results:
	Among the evaluated models, Linear Support Vector Machine (Linear SVM) achieved the best overall 
	performance and was selected as the final classifier.

Limitations:
		1. The model relies only on textual information.
		2. TF-IDF does not capture contextual semantics.
		3. Performance may decrease on ambiguous or sarcastic news articles.

Future Work: 
	Possible improvements include
		1. Transformer-based models (e.g., BERT, RoBERTa)
		2. Ensemble learning
		3. Hyperparameter optimization
		4. Explainable AI techniques (SHAP/LIME)


Reproducibility:
	1. Fixed random seed (SEED = 42)
	2. Deterministic train-validation-test split (70:10:20)
	3. Documented preprocessing pipeline
	4. TF-IDF fitted only on the training data
	5. Complete dependency list in requirements.txt



