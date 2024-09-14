# Fraud Detection Model

This project focuses on detecting fraudulent transactions using machine learning. We clean data, handle outliers, and build a predictive model to identify fraud patterns. Metrics like precision and recall evaluate the model, and the project provides recommendations for proactive fraud prevention.

### Technologies Used
- Python 3.x
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook
- 
### Installation
Install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels
```

### Data Cleaning
1. **Handling Missing Values**
   ```python
   import pandas as pd

   # Load dataset
   df = pd.read_csv('your_dataset.csv')

   # Check for missing values
   print(df.isnull().sum())

   # Impute missing values with the mean
   df.fillna(df.mean(), inplace=True)
   ```

2. **Handling Outliers**
   ```python
   # Calculate the first quartile (25th percentile)
   Q1 = df['amount'].quantile(0.25)

   # Calculate the third quartile (75th percentile)
   Q3 = df['amount'].quantile(0.75)

   # Calculate the Interquartile Range (IQR)
   IQR = Q3 - Q1

   # Remove outliers
   df = df[(df['amount'] >= (Q1 - 1.5 * IQR)) & (df['amount'] <= (Q3 + 1.5 * IQR))]
   ```

3. **Check for Multi-Collinearity**
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   # Calculate the correlation matrix
   corr = df.corr()

   # Plot the heatmap of the correlation matrix
   sns.heatmap(corr, annot=True)
   plt.show()
   ```

### Fraud Detection Model
4. **Describe Your Fraud Detection Model**
   - We use a Random Forest Classifier and XGBoost for detecting fraudulent transactions. These models are trained on transaction data to classify transactions as fraudulent or not.

5. **Variable Selection**
   - Variables included: `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`. These features are selected based on their relevance to identifying fraudulent activities.

6. **Model Performance**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report, confusion_matrix

   # Define features and target variable
   X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
   y = df['isFraud']

   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   # Initialize and train the classifier
   clf = RandomForestClassifier()
   clf.fit(X_train, y_train)

   # Predict on the test set
   y_pred = clf.predict(X_test)

   # Print confusion matrix and classification report
   print(confusion_matrix(y_test, y_pred))
   print(classification_report(y_test, y_pred))
   ```

7. **Key Factors Predicting Fraudulent Transactions**
   - Key factors include `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest`. These features are critical for identifying anomalies and suspicious activities.

8. **Evaluation of Key Factors**
   - The factors make sense as they represent critical elements of transaction data. Large discrepancies in balances and high transaction amounts often indicate fraudulent activity.

### Prevention and Infrastructure Updates
- **Recommendations**:
  - Implement real-time fraud detection systems.
  - Use multi-factor authentication (MFA) for high-value transactions.
  - Regularly update fraud detection algorithms.

- **Evaluation of Prevention Measures**:
  - Track fraud rates before and after implementing measures.
  - Monitor false positives and adjust thresholds as needed.
  - Use metrics like precision, recall, and F1-score to assess effectiveness.
