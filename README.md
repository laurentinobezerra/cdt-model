# Decision Support System for Fetal Health Triage using a Decision Tree

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


This project presents a decision support system (DSS) based on a **Decision Tree** model to classify fetal health status using data from Cardiotocography (CTG) exams. The primary goal is to provide an initial screening tool that can assist healthcare professionals in interpreting large volumes of data, aiming for the early detection of potential risks to the fetus.

The model was developed as part of a study in Big Data and Artificial Intelligence, demonstrating the practical application of machine learning algorithms in the field of maternal-fetal medicine.

**Quick Links:**
* [Video Presentation](https://youtu.be/i03pcM9l1oY)
* [Project Documentation (PDF in Portuguese)](https://github.com/laurentinobezerra/cdt-model/blob/main/SISTEMA%20DE%20APOIO%20À%20DECISÃO%20PARA%20TRIAGEM%20DE%20SAÚDE%20FETAL%20BESEADO%20EM%20ÁRVORE%20DE%20DECISÃO.pdf)

---

## Table of Contents

* [About The Project](#about-the-project)
* [The Dataset](#the-dataset)
* [Methodology](#methodology)
* [Model Results](#model-results)
    * [Confusion Matrix](#confusion-matrix)
    * [Feature Importance](#feature-importance)
    * [The Interpretable Decision Tree Model](#the-interpretable-decision-tree-model)
* [Running the Project](#running-the-project)
* [Technologies Used](#technologies-used)
* [Next Steps and Future Work](#next-steps-and-future-work)
* [Author](#author)
* [References](#references)

---

## About The Project

The interpretation of Cardiotocography (CTG) exams is crucial for monitoring fetal well-being during pregnancy. However, manual analysis can be subjective and time-consuming. This project leverages machine learning to automate the classification of fetal health into three categories (Normal, Suspect, Pathological) based on 21 features extracted from CTG scans.

The **Decision Tree** was chosen as the primary model due to its high interpretability. This "white-box" characteristic allows the decisions made by the algorithm to be quickly and easily understood by medical experts, which is a critical factor for the adoption of AI technologies in healthcare.

## The Dataset

The dataset was sourced from the Kaggle platform and is based on the work of Ayres de Campos et al. (2000), who developed the SisPorto system for automated analysis of cardiotocograms.

* **Source:** [Fetal Health Classification (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
* **Size:** 2,126 records
* **Features:** 21 numerical features (e.g., accelerations, variability, uterine contractions).
* **Target Variable (`fetal_health`):**
    * `1`: Normal
    * `2`: Suspect
    * `3`: Pathological

The class distribution in the dataset is imbalanced, as shown in the plot below. To address this, the **SMOTE (Synthetic Minority Over-sampling Technique)** was effectively applied.

![Class Distribution Plot](https://github.com/laurentinobezerra/cdt-model/blob/main/results/class-distribution-plot.png)

## Methodology

The project workflow followed the standard stages of a data science lifecycle:

1.  **Data Collection and Exploratory Data Analysis (EDA):** The data was loaded, and an initial analysis was performed to understand the distribution of features and the target variable.
2.  **Preprocessing:**
    * **Zero-Variance Feature Removal:** During the exploratory data analysis, the `severe_decelerations` feature was identified as having zero variance, meaning it contained the same value for all samples. As it held no predictive information, this column was removed from the dataset.
    * The data was split into training and testing sets with a ratio of 80/20 (`train_test_split`).
    * **SMOTE** was applied to the training set to balance the classes, preventing the model from becoming biased towards the majority class.
3.  **Modeling and Optimization:**
    * A `DecisionTreeClassifier` model was trained on the preprocessed training data.
    * **GridSearchCV** was employed to perform an exhaustive search for the optimal hyperparameters (`criterion`, `max_depth`, `min_samples_leaf`), ensuring a robust model with better generalization capabilities.
4.  **Evaluation:** The final model's performance was evaluated on the test set (unseen data) using metrics such as Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix.

## Model Results

The optimized model demonstrated strong performance in classifying fetal health, validating the effectiveness of the chosen approach.

### Confusion Matrix

The confusion matrix below illustrates the model's performance, showing the correct and incorrect predictions for each class. The high number of correct predictions along the main diagonal indicates a high degree of accuracy.

![Confusion Matrix](https://github.com/laurentinobezerra/cdt-model/blob/main/results/confusion-matrix.png)

### Feature Importance

Based on the model's final output, the feature importance analysis reveals a clear and clinically relevant hierarchy driving the classification. The top three most influential features are:

1.  **`histogram_mean`**: The mean of the fetal heart rate (FHR) histogram is the most decisive factor, serving as the primary baseline indicator for the model.
2.  **`mean_value_of_short_term_variability`**: The mean value of short-term variability is the second most influential feature, reflecting the importance of a healthy fetal autonomic nervous system.
3.  **`percentage_of_time_with_abnormal_long_term_variability`**: The percentage of time with abnormal long-term variability rounds out the top three, indicating the impact of more sustained stress.

![Feature Importance Plot](https://github.com/laurentinobezerra/cdt-model/blob/main/results/model-feature-importance-plot.png)

### The Interpretable Decision Tree Model

A key advantage of this project is the interpretability of its final model. The image below shows the optimized decision tree. This visualization provides a clear, flowchart-like representation of the decision-making process.

* **How to Read the Tree:** Start at the root node at the top. Each internal node represents a "question" about a specific feature (e.g., `abnormal_short_term_variability <= 47.5`).
* **Branches:** Following a branch (True or False) leads to the next question.
* **Leaves:** The path from the root to a leaf node represents a classification rule. The leaf node provides the final classification (Normal, Suspect, or Pathological).

This transparency allows medical professionals to audit and understand the model's logic, building trust and facilitating its integration into clinical workflows, unlike "black-box" models where the reasoning is hidden.

![Optimized Decision Tree](https://github.com/laurentinobezerra/cdt-model/blob/main/results/optimized-decision-tree.jpg)

## Running the Project

To run this project locally, follow the steps below:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/laurentinobezerra/cdt-model.git
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd cdt-model
    ```

3.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook CDT_Model_Fetal_Health.ipynb
    ```

## Technologies Used

* **Python:** The primary programming language.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical computation.
* **Scikit-learn:** For machine learning modeling, preprocessing, and evaluation.
* **Imbalanced-learn:** For handling imbalanced datasets with SMOTE.
* **Matplotlib & Seaborn:** For data visualization.
* **Jupyter Notebook:** For interactive code development and presentation.
---

## Next Steps and Future Work

This project establishes a solid foundation for a decision support system, but there are several opportunities for future expansion and improvement. The planned next steps include:

* **Implementation of the Prediction Script:** The most immediate step will be to develop and add the `predict_new_exam.py` script to the repository. This script will allow users to load the trained model (`.joblib`) and easily apply it to new exam data, making the tool operational for practical use as previously discussed.

* **Development of a Graphical User Interface (UI):** Building a simple web interface using **Flask**. The goal is to create a page where a healthcare professional can input the 21 exam values into a form and receive the model's diagnosis and confidence score in real-time, without needing to interact with the code.
---

## Author

* **Gabriel Laurentino**

---

## References

### Dataset & Original Research

* **Ayres de Campos, D., Bernardes, J., Garrido, A., & de Sá, J. P. M. (2000).** SisPorto 2.0: A Program for Automated Analysis of Cardiotocograms. *Journal of Maternal-Fetal Medicine, 9(5)*, 311-318.
* **Kaggle.** Fetal Health Classification Dataset. Retrieved June 27, 2025, from [https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification).

### Methodology & Algorithms

* **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research, 16*, 321-357.

### Software & Libraries

* **[Scikit-learn](https://scikit-learn.org/):** Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.
* **[Imbalanced-learn](https://imbalanced-learn.org/stable/):** Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. *Journal of Machine Learning Research, 18(17)*, 1-5.
* **[NumPy](https://numpy.org/):** Harris, C. R. et al. (2020). Array programming with NumPy. *Nature, 585*, 357-362.
* **[Pandas](https://pandas.pydata.org/):** McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56-61.
* **[Matplotlib](https://matplotlib.org/):** Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering, 9(3)*, 90-95.
* **[Seaborn](https://seaborn.pydata.org/):** Waskom, M. L. (2021). Seaborn: Statistical data visualization. *Journal of Open Source Software, 6(60)*, 3021.