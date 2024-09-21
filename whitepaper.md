# Demo 3
## Business Goals

**Business Question**
The primary business question being addressed is: "What factors contribute to employee attrition in the company, and how can we predict which employees are likely to leave?" This question aims to identify the key drivers of attrition and develop a predictive model to forecast future attrition events. The insights from this model can help the company implement targeted retention strategies, thereby reducing turnover rates and associated costs.

**How the Machine Learning Solution is Expected to Address the Business Question / Goal**

The machine learning solution, specifically a classification model using AutoML Tabular (within Vertex AI), is expected to address the business question in the following ways:
* **Prediction**: By analysing historical data on employee attributes (such as age, business travel frequency, department, distance from home, education, job role, and other relevant features), the model will predict the likelihood of an employee leaving the company.
* **Insight Generation**: The model will help uncover significant patterns and correlations in the data. For instance, it might reveal that employees in certain job roles or those with specific levels of job satisfaction are more likely to leave.
* **Actionable Recommendations**: With the predictions and insights from the model, the HR department can develop targeted strategies to improve employee retention. For example, if the model indicates that lack of career progression is a key factor, the company can focus on providing more growth opportunities for employees.
* **Resource Allocation**: The predictions can help prioritise which employees might need more attention or resources to improve their satisfaction and engagement levels, leading to better resource allocation and more effective retention strategies.

## Data Exploration

**Steps and Decisions:**
1. **Data Loading and Initial Inspection:** Loaded the dataset from Google Cloud Storage, inspected basic statistics, missing values, and data types.
2. **Class Distribution Analysis:** Analysed the distribution of the target variable (Attrition) to identify class imbalance.
3. **Data Cleaning and Preparation**: Standardised categorical values by replacing spaces with underscores and converting to lowercase. Mapped specific categorical values to standard ones.
4. **Dynamic Binning of Numerical Features:** Applied dynamic binning to numerical columns where the range is greater than 5.
5. **Chi-Square Tests:** Performed chi-square tests to determine the independence of Gender and MaritalStatus with respect to Attrition.

```python

# Encoding categorical variables
df['Gender'] = df['Gender']**.**map({'male': 1, 'female': 0})
df['MaritalStatus'] = df['MaritalStatus'].map({'single': 0, 'married': 1, 'divorced': 2})

# Contingency tables
contingency_table_gender = pd.crosstab(df['Gender'], df['Attrition'])
contingency_table_marital = pd.crosstab(df['MaritalStatus'], df['Attrition'])

print("Contingency Table - Gender vs Target:")
print(contingency_table_gender)

print("\nContingency Table - Marital Status vs Target:")
print(contingency_table_marital)

# Chi-Square tests
chi2_gender, p_gender, dof_gender, ex_gender **=** chi2_contingency(contingency_table_gender)
chi2_marital, p_marital, dof_marital, ex_marital **=** chi2_contingency(contingency_table_marital)

print(f"\nChi-Square Test between gender and target: chi2 = {chi2_gender}, p-value = {p_gender}")
print(f"Chi-Square Test between marital_status and target: chi2 = {chi2_marital}, p-value = {p_marital}")

# Interpretation
alpha = 0.05
print("\nInterpretation:")
if p_gender < alpha:
    print("There is a significant association between gender and target (p < 0.05).")
else:
    print("There is no significant association between gender and target (p >= 0.05).")

if p_marital < alpha:
    print("There is a significant association between marital status and target (p < 0.05).")
else:
    print("There is no significant association between marital status and target (p >= 0.05).")

```

**Summary**: Data exploration revealed key characteristics of the dataset, such as class imbalance in Attrition, and informed the need for handling this imbalance. Chi-square tests determined the non-significance of Gender and the significance of MaritalStatus, leading to decisions on feature inclusion and

## Feature Engineering

**Steps and Code Snippets:**
* **Handling Missing Values:** Imputed missing values to ensure completeness of the dataset.
* **Encoding Categorical Variables:** Applied label encoding and one-hot encoding based on the context and analysis.
* **Applying SMOTENC:** Addressed class imbalance by applying SMOTENC on the dataset.
* **Dropping Features:** After examination of first attempt with the AutoML model, Features such as: BusinessTravel, StandardHours, Over18 were removed as they didn’t provide enough predictive power for the model.

```python

df.drop(columns=['Gender','StandardHours', 'Over18', 'EmployeeCount', 'EmployeeNumber', 'BusinessTravel'], inplace=True)`

```

**Summary**:
Feature engineering involved encoding categorical features, handling missing values, and applying SMOTENC to address class imbalance. SMOTENC was specifically chosen over SMOTE due to its ability to handle categorical features directly. These steps ensured that the dataset was in a suitable format for machine learning models and that the model would perform well on minority classes, providing reliable predictions.

## Preprocessing and the data pipeline

**Key Preprocessing Steps include:**
* Handling categorical variables
  * Ordinal encoding
  * Binning continuous features
* Drop irrelevant columns
* Oversampling using SMOTENC

**Summary**:
The data processing pipeline begins by loading a dataset and encoding the target variable (Attrition) and other categorical features using ordinal, one-
hot , and label binarization techniques. Continuous features are binned based on quantiles for better discretisation. Irrelevant columns are dropped, and binary feature like Overtime are mapped to numeric values.

All these steps are wrapped in to a data_transformation function, that is then called by main ml_spec3_pipeline

## Machine Learning model design and selection

**AutoML Product Chosen**: For this implementation, Google Cloud AutoML Tabular (Vertex AI) was chosen to build and evaluate the machine learning model for predicting employee attrition.

**Selection Criteria**:
1. Ease of Use: Google Cloud AutoML Tabular simplifies the process of building machine learning models by automating many of the complex tasks involved in model training and evaluation. This makes it accessible even to those with limited machine learning expertise.
2. Comprehensive Preprocessing: AutoML Tabular handles various preprocessing tasks such as handling missing values, encoding categorical variables, and normalising numerical features, reducing the need for extensive manual preprocessing.
3. Support for Mixed Data Types: AutoML Tabular is capable of handling datasets with both numerical and categorical features, which aligns well with the characteristics of our dataset.
4. Automated Feature Engineering and Selection: AutoML Tabular performs automated feature engineering and selection to optimise model performance.
5. Evaluation Metrics and Insights: AutoML Tabular provides comprehensive evaluation metrics and insights, which are crucial for understanding model performance and making informed decisions

## Machine learning model training and development

**Dataset Sampling and Justification:**
For model training, we used stratified sampling to ensure that the class distribution of the target variable (Attrition) was preserved in both the training and validation/test datasets. This method was chosen to address the imbalance in the dataset and ensure that the model is trained and evaluated on representative samples. 

**Implementation of AutoML-Based Model Training:** Google Cloud AutoML was utilised for model training due to its ability to handle both numerical and categorical features and automatically optimise the model based on the provided data. The following steps were taken to implement AutoML-based model training:

**Initialise the Vertex AI and create the dataset**: The dataset was uploaded to
Google Cloud Storage and then imported into Vertex AI for model training.

**Define and Run the AutoML Training Job**: We specified the optimisation
objective as maximize-au-prcto handle the class imbalance and focus on
improving the precision-recall area under the curve (PR AUC).
Model Evaluation Metric: The chosen evaluation metric was
maximize-au-prc(Area Under the Precision-Recall Curve).
This metric is optimal for the business question being addressed,
which focuses on accurately predicting employee attrition. Given
the class imbalance in the dataset, precision-recall metrics are
more informative than accuracy or ROC-AUC, as they better
capture the model's performance on the minority class (employees
who leave the company).

```python

desired_minority_count = int(0.3 * 1233)
    sampling_strategy = {0: 1233, 1: desired_minority_count}
    X_train = df.drop(columns=['Attrition'])
    y_train = df['Attrition']
    print(X_train.dtypes)
    smotenc = SMOTENC(categorical_features="auto", sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = smotenc.fit_resample(X_train, y_train)

    # Combine resampled features with target
    df = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train.columns), pd.DataFrame(y_train_resampled, columns=['Attrition'])], axis=1) `

```

**Discussion on the Metric:**
* **Precision-Recall Curve (PR AUC)**: This metric evaluates the trade-off between precision (the accuracy of positive predictions) and recall (the ability to identify all positive instances). It is particularly useful for imbalanced datasets where the minority class is of greater interest.
* **Business Goal Alignment:** The business goal is to predict employee attrition accurately. High precision ensures that the predicted attrition cases are likely to be true, reducing false positives. High recall ensures that most actual attrition cases are captured, reducing false negatives.

## Machine learning model evaluation

**Precision-Recall & ROC Curves:**
* High precision and recall across most thresholds, indicating strong model performance.
* ROC curve closely approaches top-left, suggesting excellent discriminative ability.

**Confusion Matrix:**
* High true positive rate and low false positive rate.
* Model shows strong accuracy in classifying both positive and negative cases.

**Feature Importance:**
* Top features impacting model decisions include Monthly Income, Overtime, and Job Level.
* Insights align with key business variables, ensuring relevance in practical applications.
