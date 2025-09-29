# Credit Card Fraud Detection

This project develops and evaluates advanced machine learning models to accurately detect fraudulent credit card transactions. The primary focus is on handling the severe class imbalance inherent in the dataset by employing sophisticated algorithms and techniques to maximize the identification of fraudulent activities while minimizing false positives.



##  Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Methodology and Models](#-methodology-and-models)
- [Model Evaluation Metrics](#-model-evaluation-metrics)
- [Results](#-results)
- [Conclusion](#-conclusion)
- [Future Work](#-future-work)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

##  Project Overview

This is an end-to-end machine learning project that demonstrates a robust workflow for tackling a real-world imbalanced classification problem. The project moves beyond baseline models to implement and compare powerful gradient boosting algorithms like XGBoost and LightGBM. It explores various techniques for model validation and handling class imbalance to build a highly effective fraud detection system.

##  Problem Statement

The central challenge is to build a model that can accurately distinguish the rare fraudulent transactions from the vast majority of legitimate ones. An effective solution must:
*   **Maximize Fraud Detection**: Correctly identify as many fraudulent transactions as possible (high recall).
*   **Maintain Reliability**: Ensure that legitimate transactions are not frequently flagged as fraudulent (high precision).
*   **Navigate Imbalance**: Overcome the statistical challenges posed by a dataset where fraud represents less than 0.2% of all transactions.

##  Dataset

The project uses the "Credit Card Fraud Detection" dataset available on Kaggle.

*   **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
*   **Description**: The dataset contains transactions made by European cardholders over two days in September 2013. It includes 284,807 transactions, of which only 492 (0.172%) are fraudulent.
*   **Features**: Due to confidentiality, the primary features (`V1` through `V28`) are the result of a PCA transformation. The only features that have not been transformed are `Time` and `Amount`. The target variable is `Class`.

##  Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8 or higher
*   pip (Python package installer)

##  Methodology and Models

This project employed an advanced, iterative approach to model development. Multiple powerful algorithms were trained and evaluated, and various techniques were used to handle the class imbalance and ensure model robustness.

### Machine Learning Models Used

1.  **Random Forest:** An ensemble model that uses multiple decision trees to improve predictive accuracy and control overfitting. It served as a strong baseline to compare against gradient boosting methods.
2.  **LightGBM (Light Gradient Boosting Machine):** A high-performance gradient boosting framework known for its speed and efficiency. It was tested both with and without cross-validation.
3.  **XGBoost (eXtreme Gradient Boosting):** A powerful and highly popular gradient boosting library known for its performance and accuracy. It was the core focus of this project's optimization efforts.

### Techniques for Imbalance and Validation

*   **`scale_pos_weight` (XGBoost):** A parameter used to assign a higher penalty to the misclassification of the minority class, forcing the model to pay more attention to fraudulent transactions.
*   **SMOTE (Synthetic Minority Over-sampling Technique):** An oversampling technique that creates new, synthetic data points for the minority class to balance the dataset.
*   **K-Fold and Stratified K-Fold Cross-Validation (CV):** Validation techniques used to ensure that the model's performance is robust and generalizable by training and testing on different subsets of the data. Stratified K-Fold is particularly useful for imbalanced datasets as it preserves the class distribution in each fold.

##  Model Evaluation Metrics

Given the severe class imbalance, standard accuracy is a misleading metric. This project focused on more informative metrics:

*   **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Measures the model's ability to distinguish between the positive and negative classes. An AUC of 1.0 represents a perfect classifier.
*   **AUPRC (Area Under the Precision-Recall Curve):** This is the most critical metric for this problem. It summarizes the trade-off between precision and recall across different probability thresholds and is highly sensitive to model performance on the minority (fraud) class. A higher AUPRC indicates a better fraud detection model.

##  Results

The performance of the different models and techniques was systematically evaluated. The **XGBoost (No CV)** model, leveraging the `scale_pos_weight` parameter, delivered the most compelling results.

| Model | ROC-AUC | AUPRC | Highlights |
| :--- | :---: | :---: | :--- |
| Random Forest | 0.85 | 0.64 | High precision (91.1%), moderate recall (70.6%); no imbalance handling |
| LightGBM (No CV) | 0.9579 | 0.35 | High ROC-AUC, but poor fraud detection due to low AUPRC |
| **XGBoost (No CV)** | **0.974** | **0.83** | **Excellent precision (93.83%) & recall (76.0%); used `scale_pos_weight`**|
| LightGBM (K-Fold CV) | 0.9703 | 0.51 | CV improved AUPRC (0.35 → 0.51); no class imbalance handling |
| XGBoost (Stratified K-Fold + SMOTE) | 0.984 | 0.8151 ± 0.0587 | Strong AUC; AUPRC slightly below no-CV; SMOTE & CV added robustness |

**Analysis**:
The **XGBoost (No CV)** model stood out with the highest AUPRC of **0.83**, indicating the best performance on the minority fraud class. It also achieved a remarkable balance of **93.83% precision** and **76.0% recall**, making it highly effective at both identifying fraud and avoiding false alarms. While the more complex `XGBoost + SMOTE + CV` approach yielded the highest ROC-AUC and added robustness, its slightly lower AUPRC makes the simpler, `scale_pos_weight`-based XGBoost model the most practical and highest-performing choice.

##  Conclusion

This project successfully implemented and evaluated several advanced machine learning models for credit card fraud detection. The key conclusion is that the **XGBoost algorithm, when combined with the `scale_pos_weight` parameter, provides the most effective solution for this problem.**

This specific model configuration achieved an outstanding **AUPRC of 0.83** and delivered an excellent balance of high precision (93.83%) and high recall (76.0%). This demonstrates its superior ability to accurately flag fraudulent transactions while minimizing the incorrect flagging of legitimate ones.

Based on this comprehensive analysis, the **XGBoost model using `scale_pos_weight` is the definitive recommendation for deployment**, offering a powerful, accurate, and efficient system for fraud detection.

##  Future Work

Potential improvements and future enhancements for this project include:

*   **Hyperparameter Tuning**: Perform extensive hyperparameter tuning on the winning XGBoost model using techniques like Optuna or Bayesian Optimization to further enhance its performance.
*   **Feature Engineering**: Create new, informative features from the existing `Time` and `Amount` columns to potentially provide more signal to the model.
*   **Deep Learning Models**: Explore anomaly detection using deep learning architectures like Autoencoders, which may capture different patterns in the data.
*   **Deployment**: Package the final model and deploy it as a REST API using a framework like FastAPI or Flask to serve real-time predictions.
