import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
class KNN:
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X if not hasattr(X, 'toarray') else X.toarray()
        self.y_train = y.reset_index(drop=True)

    def predict(self, X):
        predictions = []
        X = X if not hasattr(X, 'toarray') else X.toarray()
        for x in tqdm(X, desc="Predicting", unit="sample"):
            distances = [self.compute_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(majority_vote)
        return predictions

    def compute_distance(self, X1, X2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((X1 - X2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(X1 - X2))
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
def preprocess_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop('Exited', axis=1)
    y_train = train_data['Exited']

    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(test_data)

    return X_train_processed, y_train, X_test_processed
def cross_validate(X, y, knn, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        knn.fit(X_train, y_train)
        y_pred_prob = knn.predict(X_val)
        auc_score = roc_auc_score(y_val, y_pred_prob)
        auc_scores.append(auc_score)
    return np.mean(auc_scores), auc_scores


X, y, X_test = preprocess_data('train.csv', 'test.csv')
knn = KNN(k=42, distance_metric='euclidean')
cv_mean_score, cv_scores = cross_validate(X, y, knn, n_splits=5)
print("Cross-validation scores:", cv_scores)
knn.fit(X, y)
test_predictions = knn.predict(X_test)
submission = pd.DataFrame({'id': pd.read_csv('test.csv')['id'], 'Exited': test_predictions})
submission.to_csv('submissions.csv', index=False)