# Employee Attrition Prediction

### Description

This project aims to build a machine learning system capable of predicting whether an employee will leave their position, using data collected by the Human Resources department. It covers all phases of the model development cycle: exploratory data analysis (EDA), preprocessing, model selection and evaluation, hyperparameter tuning, and final prediction generation.

---

### Repository Structure

- 01_EDA_and_Model_Train.ipynb: Main notebook containing exploratory analysis, preprocessing, model training, evaluation, and final model selection.
- 02_Final_Model_Predictions.ipynb: Notebook that loads the final model and generates predictions on new data.
- final_model.pkl: File containing the trained final model.
- predictions.csv: File containing the generated predictions.
- README.md: This file.                  

---

### Objectives

- Apply multiple classification algorithms (KNN, Trees, Linear Models, SVM).
- Compare preprocessing configurations (scaling, imputation).
- Tune hyperparameters using cross-validation.
- Estimate model performance on unseen data.
- Identify relevant features for prediction.

---

### Technologies Used

- Python 3.x  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

### Performance Evaluation

#### Schemes used:
- **Holdout**: Train/test split in a 2/3 - 1/3 proportion for final estimation.
- **Cross-validation**: For internal evaluation and hyperparameter tuning.

#### Reported metrics:
- Balanced Accuracy  
- TPR (True Positive Rate)  
- TNR (True Negative Rate)  
- Overall Accuracy  
- Confusion Matrices  

---

### Implemented Models

#### Basic:
- K-Nearest Neighbors (KNN)  
- Decision Trees  

#### Advanced:
- Random Forest
- Logistic Regression (with and without L1 regularization)  
- Support Vector Machines (SVM)  

Each model was evaluated both with default parameters and after fine-tuning.

---

### Results

- Analyzed the impact of preprocessing and hyperparameter tuning on performance.
- Compared models based on metrics and training times.
- Selected the best model to generate predictions on new data.
- The final model was saved and its predictions used for external evaluation.

---

### Author

- **Name**: Carlos Bravo Garrán
