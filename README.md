# Student Performance Analysis and Prediction

## Problem Statement

Academic performance in college depends on multiple factors such as daily study hours, lecture attendance, previous academic foundation, and sleep habits. The goal of this project is to build an end-to-end machine learning system that can:
1. Clean and explore student data to identify the habits most associated with high marks.
2. Predict a student's continuous final marks (regression).
3. Predict whether a student will pass or fail and assign their letter grade (classification).
4. Discover natural student behavioral segments to help teachers identify who needs extra support (clustering).
5. Build an automated Scikit-Learn pipeline with cross-validation that handles raw data, addresses class imbalance, and provides an interactive command-line tool for real-time predictions.

## Dataset

This dataset is synthetic and was generated using NumPy random numbers (`random_state=42`) via the script `generate_data.py`. It reflects the typical distributions and challenges found in real-world educational records.

The synthetic data formula for final marks combines previous marks (40% weight), study hours (3.2 multiplier), and attendance (25% weight), with added Gaussian noise to simulate individual exam-day variance. Realistic data imperfections were intentionally added:
- Approximately 5% missing values in `study_hours`, `attendance`, and `parental_education`.
- Exactly 10 duplicate rows.
- A small number of extreme study hour outliers (> 15 hours).
- Class imbalance: failing students account for roughly 12% of the cohort.

### Dataset Schema

| Column Name | Data Type | Description | Value Range / Categories |
| :--- | :--- | :--- | :--- |
| student_id | Integer | Unique identifier for each student | 1001 to 1490 |
| gender | String | Gender of the student | Male, Female |
| study_hours | Float | Daily self-study hours outside class | 0.5 to 10.0 (outliers up to 18.0) |
| attendance | Float | Percentage of lectures attended | 40.0% to 100.0% |
| previous_marks | Float | Score in previous semester exams | 15.0 to 98.0 |
| sleep_hours | Integer | Average sleep duration per night | 4 to 9 hours |
| parental_education | String | Highest education level of parents | School, Graduate, Postgraduate |
| extracurricular | String | Participation in clubs or sports | Yes, No |
| final_marks | Float | Continuous target score | 10.0 to 99.0 |
| result | String | Binary outcome (Pass if marks >= 35) | Pass, Fail |
| grade | String | Four-tier letter grade | A (>=60), B (48-59), C (35-47), D (<35) |

## Project Structure

```
student-performance-ml/
  .gitignore
  README.md
  requirements.txt
  generate_data.py
  data/
    students.csv
    students_clean.csv
  notebooks/
    01_data_handling_and_eda.ipynb
    02_regression_knn_metrics.ipynb
    03_classification.ipynb
    04_clustering_and_pca.ipynb
    05_pipeline_cv_and_saving.ipynb
  models/
    best_model.joblib
  images/
    01_correlation_heatmap.png
    01_grade_boxplot.png
    01_marks_distribution.png
    01_outlier_boxplot.png
    01_result_countplot.png
    01_study_hours_capped_boxplot.png
    01_study_vs_marks.png
    02_bias_variance_polynomial.png
    02_knn_k_vs_error.png
    02_simple_linear_regression.png
    03_confusion_matrix.png
    03_roc_curve.png
    03_sigmoid_curve.png
    04_hierarchical_dendrogram.png
    04_kmeans_elbow.png
    04_pca_clusters.png
    04_silhouette_analysis.png
  src/
    predict.py
```

## Syllabus Topics Covered

| B.Tech AI/ML Syllabus Topic | Notebook Name | Implementation Details |
| :--- | :--- | :--- |
| NumPy Arrays and Memory | 01_data_handling_and_eda.ipynb | Array mean, reshaping, boolean masking, view vs copy memory test |
| Pandas Data Exploration | 01_data_handling_and_eda.ipynb | head, tail, sample, info, describe, missing counts, drop_duplicates, groupby |
| Exploratory Data Analysis | 01_data_handling_and_eda.ipynb | Histograms, boxplots, scatter plots, countplots, correlation heatmaps |
| Data Cleaning and Preprocessing | 01_data_handling_and_eda.ipynb | SimpleImputer (median and mode), IQR outlier capping, ordinal and one-hot encoding, scaling |
| Linear and Regularized Regression | 02_regression_knn_metrics.ipynb | Simple Linear Regression, Multiple Linear Regression, Ridge (L2), Lasso (L1) |
| Non-Linear Modeling and KNN | 02_regression_knn_metrics.ipynb | Polynomial expansion (degree 2), KNN Regressor (tuning k from 1 to 20) |
| Model Complexity and Overfitting | 02_regression_knn_metrics.ipynb | Bias-variance trade-off across polynomial degrees 1, 2, 4, 10 and KNN k=1 vs k=6 |
| Binary Classification | 03_classification.ipynb | Logistic Regression, Sigmoid function, decision boundary, confusion matrix, ROC-AUC |
| Multiclass Classification | 03_classification.ipynb | Softmax Regression for grade prediction, probability verification (sum = 1.0) |
| Imbalanced Data Evaluation | 03_classification.ipynb | Comparing accuracy vs recall on minority class, dummy classifier baseline |
| Unsupervised Clustering | 04_clustering_and_pca.ipynb | k-Means with elbow method and silhouette analysis, cluster profiling into 3 personas |
| Hierarchical Clustering | 04_clustering_and_pca.ipynb | SciPy Ward linkage dendrogram, AgglomerativeClustering, silhouette comparison |
| Dimensionality Reduction | 04_clustering_and_pca.ipynb | PCA 2D projection, explained variance ratio, cluster visualization |
| Cross-Validation Strategies | 05_pipeline_cv_and_saving.ipynb | 5-Fold KFold vs StratifiedKFold cross-validation comparison |
| Imbalanced Class Techniques | 05_pipeline_cv_and_saving.ipynb | class_weight='balanced' vs random oversampling (train set only) |
| Scikit-Learn Pipelines | 05_pipeline_cv_and_saving.ipynb | ColumnTransformer with SimpleImputer, StandardScaler, OneHotEncoder, and GridSearchCV |
| Model Serialization and Deployment | 05_pipeline_cv_and_saving.ipynb | Saving pipeline with joblib and interactive CLI inference script in src/predict.py |

## How to Run

### 1. Set Up Python Virtual Environment
Open a terminal in the project directory:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
# source venv/bin/activate
```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the Dataset

```bash
python generate_data.py
```

This will create `data/students.csv` containing 500 student rows.

### 4. Run the Jupyter Notebooks

Launch Jupyter Notebook and open the notebooks in order:

```bash
jupyter notebook
```

Run notebooks 01 through 05 sequentially:
- `notebooks/01_data_handling_and_eda.ipynb` (generates `data/students_clean.csv`)
- `notebooks/02_regression_knn_metrics.ipynb`
- `notebooks/03_classification.ipynb`
- `notebooks/04_clustering_and_pca.ipynb`
- `notebooks/05_pipeline_cv_and_saving.ipynb` (generates `models/best_model.joblib`)

Alternatively, to execute all notebooks from the terminal:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_data_handling_and_eda.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_regression_knn_metrics.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/03_classification.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/04_clustering_and_pca.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/05_pipeline_cv_and_saving.ipynb
```

### 5. Run the Prediction Tool

After training and saving the pipeline in Notebook 05, test the command-line prediction script:

```bash
python src/predict.py
```

Enter student details (study hours, attendance, previous marks, sleep, gender, parental education, and extracurriculars) when prompted to view the predicted outcome and probabilities.

## Results

### Regression Models (Target: final_marks)

Evaluated on 98 test students:

| Model | MAE | MSE | RMSE | R2 Score |
| :--- | :--- | :--- | :--- | :--- |
| Multiple Linear Regression | 3.355 | 18.844 | 4.341 | 0.838 |
| Ridge Regression (alpha=1.0) | 3.357 | 18.849 | 4.342 | 0.838 |
| Lasso Regression (alpha=0.1) | 3.384 | 18.888 | 4.346 | 0.837 |
| Polynomial Regression (degree 2) | 3.552 | 20.657 | 4.545 | 0.822 |
| KNN Regressor (k=6) | 5.257 | 41.004 | 6.403 | 0.647 |
| Simple Linear Regression (study_hours) | 6.974 | 72.881 | 8.537 | 0.373 |

### Classification Models (Target: result / grade)

| Model | Target | Accuracy | Fail Recall | Fail F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Binary Logistic Regression (Default) | result (Pass/Fail) | 92.86% | 50.00% | 0.632 | 0.9457 |
| Binary Logistic Regression (Balanced) | result (Pass/Fail) | 81.63% | 100.00% | 0.571 | 0.9457 |
| KNN Classifier (k=5) | result (Pass/Fail) | 89.80% | 50.00% | 0.545 | - |
| Softmax Regression | grade (A, B, C, D) | 80.61% | - | - | - |
| Naive Dummy Baseline | result (Guess Pass) | 87.76% | 0.00% | 0.000 | 0.5000 |

### Clustering and Student Segmentation (Unsupervised)

- Optimal Clusters: k = 3 (confirmed by Elbow Method and Silhouette Analysis).
- k-Means Silhouette Score: 0.1340.
- Agglomerative Hierarchical Silhouette Score: 0.1293.
- PCA Explained Variance (2 Components): 44.30% (PC1: 26.65%, PC2: 17.65%).

#### Identified Student Personas:
1. Consistent Achievers: High study hours (~6.3 hrs), high attendance (~88%), high previous marks (~77).
2. Average Students: Moderate study hours (~4.6 hrs), steady attendance (~74%), solid foundation (~59 marks).
3. At-Risk Students: Low study hours (~3.5 hrs), low attendance (~58%), weak previous marks (~38).

### Final Pipeline (Trained on Raw Data)

- Architecture: `ColumnTransformer` (SimpleImputer median + StandardScaler for numerics; SimpleImputer most_frequent + OneHotEncoder for categoricals) followed by `LogisticRegression(class_weight='balanced', C=0.01)`.
- 5-Fold Stratified Cross-Validation Macro F1: 0.7187.
- Test Set Fail Recall: 1.00 (100% of failing students successfully identified).

## What I Learned

1. **Why accuracy is deceptive on imbalanced datasets**: When 88% of students pass, a dummy model that predicts Pass for every student gets 88% accuracy but helps zero at-risk students. Tracking minority class recall is critical.
2. **The bias-variance trade-off in practice**: Increasing polynomial degrees or choosing k=1 in KNN drives training error to zero but causes test error to blow up. Regularization (Ridge/Lasso) and cross-validation help find the right balance.
3. **The power of Scikit-Learn Pipelines**: Hardcoding preprocessing manually can leak test data into training data. Building end-to-end pipelines guarantees that scalers and imputers fit only on training folds and handle raw input during deployment without manual intervention.
4. **Unsupervised clustering reveals actionable insights**: Even without exam labels, clustering grouped students into clear behavioral personas (consistent, average, at-risk) that teachers can use for early academic intervention.

## Future Improvements

1. Collect real student exam records to validate the synthetic findings on actual academic cohorts.
2. Explore tree-based ensemble models like Random Forests and Gradient Boosting for tabular marks prediction.
3. Add a lightweight web interface using Streamlit or Flask so teachers can enter student details directly from a browser.
4. Incorporate time-series attendance tracking over the semester to identify declining attendance trends before final exams.
