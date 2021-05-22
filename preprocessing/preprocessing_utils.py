# ------------------------------------------
# Utils functions for classical model tuning
# ------------------------------------------
import numpy as np
import pandas as pd
import os

from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score, roc_curve, auc, \
                        average_precision_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler



def train_and_evaluate_clf(X, y, grid_search_clf, folds, test_size, scaling=True):
	"""
		Function that trains a GridSearchCV across multiple folds and 
		reports testing results as well as the best parameters
	"""
    # Initiate results
    results = {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'auroc': [],
          'fprs': [], 'tprs': [], 'precisions': [], 'recalls': []}
    best_params = []

    for fold in range(folds):
        print(f"Fold {fold}")
        # Split in train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                             stratify=y)
        
        # Scale data
        if scaling:
	        scaler = StandardScaler().fit(X_train)
	        X_train = scaler.transform(X_train)
	        X_test = scaler.transform(X_test)
        
        # Grid search
        grid_search_clf.fit(X_train, y_train)
        best_params.append(grid_search_clf.best_params_)
        
        # Evaluate
        y_true, y_pred = y_test, grid_search_clf.predict(X_test)
        y_score = grid_search_clf.decision_function(X_test)
        assert np.array_equal(y_pred,(y_score>0).astype(int))
        
        # Compute the scores
        results['accuracy'].append(accuracy_score(y_true, y_pred))
        results['average_precision'].append(average_precision_score(y_true, y_score))
        results['balanced_accuracy'].append(balanced_accuracy_score(y_true, y_pred))
        results['auroc'].append(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        results['tprs'].append(tpr)
        results['fprs'].append(fpr)
        pr, rc, _ = precision_recall_curve(y_true, y_score)
        results['precisions'].append(pr)
        # Although tpr and recall are the same thing, they are computed at different thresholds
        results['recalls'].append(rc)
    return results, best_params

def get_xy_from_df(df, date):
    # Returns x and y for a given date, can also be given "all" as a date
    # Select the appropriate columns:
    if date == 'all':
        exclude = ['plot', 'label']
        x = df[df.columns.difference(exclude)].values
    else:
        cols = df.columns[[date in x for x in df.columns]]
        x = df[cols].values
    y = df['label'].values
    return x,y

