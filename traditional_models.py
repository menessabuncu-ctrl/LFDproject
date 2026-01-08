"""
Traditional ML Models Implementation
Learning from Data - Final Project
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time


class TraditionalMLPipeline:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_params = {}

    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """
        Logistic Regression with L2 regularization
        """
        print("\n" + "=" * 60)
        print("Training Logistic Regression")
        print("=" * 60)

        # Hyperparameter tuning
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000]
        }

        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time

        best_model = grid_search.best_estimator_
        self.models['logistic_regression'] = best_model
        self.best_params['logistic_regression'] = grid_search.best_params_

        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)

        # Metrics
        results = self._calculate_metrics(
            y_train, y_train_pred, y_val, y_val_pred,
            'Logistic Regression', train_time
        )
        self.results['logistic_regression'] = results

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Training time: {train_time:.2f}s")

        return best_model, results

    def train_svm_linear(self, X_train, y_train, X_val, y_val):
        """
        Linear SVM
        """
        print("\n" + "=" * 60)
        print("Training Linear SVM")
        print("=" * 60)

        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'max_iter': [1000]
        }

        svm = LinearSVC(random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time

        best_model = grid_search.best_estimator_
        self.models['svm_linear'] = best_model
        self.best_params['svm_linear'] = grid_search.best_params_

        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)

        results = self._calculate_metrics(
            y_train, y_train_pred, y_val, y_val_pred,
            'Linear SVM', train_time
        )
        self.results['svm_linear'] = results

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Training time: {train_time:.2f}s")

        return best_model, results

    def train_svm_rbf(self, X_train, y_train, X_val, y_val):
        """
        SVM with RBF kernel
        """
        print("\n" + "=" * 60)
        print("Training RBF SVM")
        print("=" * 60)

        # Note: RBF kernel can be slow on large datasets
        # Using a subset for faster training
        subset_size = min(2000, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]

        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.1],
            'kernel': ['rbf']
        }

        svm = SVC(random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)

        start_time = time.time()
        grid_search.fit(X_train_subset, y_train_subset)
        train_time = time.time() - start_time

        best_model = grid_search.best_estimator_
        self.models['svm_rbf'] = best_model
        self.best_params['svm_rbf'] = grid_search.best_params_

        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)

        results = self._calculate_metrics(
            y_train, y_train_pred, y_val, y_val_pred,
            'RBF SVM', train_time
        )
        self.results['svm_rbf'] = results

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Training time: {train_time:.2f}s (on {subset_size} samples)")

        return best_model, results

    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """
        Random Forest Classifier
        """
        print("\n" + "=" * 60)
        print("Training Random Forest")
        print("=" * 60)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time

        best_model = grid_search.best_estimator_
        self.models['random_forest'] = best_model
        self.best_params['random_forest'] = grid_search.best_params_

        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)

        results = self._calculate_metrics(
            y_train, y_train_pred, y_val, y_val_pred,
            'Random Forest', train_time
        )
        self.results['random_forest'] = results

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Training time: {train_time:.2f}s")

        return best_model, results

    def train_knn(self, X_train, y_train, X_val, y_val):
        """
        K-Nearest Neighbors
        """
        print("\n" + "=" * 60)
        print("Training K-Nearest Neighbors")
        print("=" * 60)

        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        knn = KNeighborsClassifier(n_jobs=-1)
        grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time

        best_model = grid_search.best_estimator_
        self.models['knn'] = best_model
        self.best_params['knn'] = grid_search.best_params_

        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)

        results = self._calculate_metrics(
            y_train, y_train_pred, y_val, y_val_pred,
            'K-NN', train_time
        )
        self.results['knn'] = results

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Training time: {train_time:.2f}s")

        return best_model, results

    def _calculate_metrics(self, y_train, y_train_pred, y_val, y_val_pred, model_name, train_time):
        """Calculate comprehensive metrics"""
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='weighted'
        )

        results = {
            'model_name': model_name,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': train_time,
            'confusion_matrix': confusion_matrix(y_val, y_val_pred)
        }

        print(f"\nResults for {model_name}:")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        return results

    def plot_confusion_matrices(self, save_path='results/confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f"{results['model_name']}\nAcc: {results['val_accuracy']:.3f}")
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_comparison_table(self):
        """Create comparison table of all models"""
        comparison_data = []

        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': results['model_name'],
                'Train Acc': f"{results['train_accuracy']:.4f}",
                'Val Acc': f"{results['val_accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'Time (s)': f"{results['train_time']:.2f}"
            })

        df = pd.DataFrame(comparison_data)
        print("\n" + "=" * 80)
        print("MODEL COMPARISON TABLE")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        return df

    def save_models(self, filepath='models/traditional_models.pkl'):
        """Save all trained models"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'results': self.results,
                'best_params': self.best_params
            }, f)
        print(f"\nModels saved to {filepath}")


def train_all_traditional_models(data):
    """Train all traditional ML models"""
    X_train = data['features']['tfidf']['X_train']
    X_val = data['features']['tfidf']['X_val']
    y_train = data['labels']['y_train']
    y_val = data['labels']['y_val']

    pipeline = TraditionalMLPipeline()

    # Train all models
    pipeline.train_logistic_regression(X_train, y_train, X_val, y_val)
    pipeline.train_svm_linear(X_train, y_train, X_val, y_val)
    pipeline.train_svm_rbf(X_train.toarray(), y_train, X_val.toarray(), y_val)
    pipeline.train_random_forest(X_train, y_train, X_val, y_val)
    pipeline.train_knn(X_train.toarray(), y_train, X_val.toarray(), y_val)

    # Visualizations and comparisons
    pipeline.plot_confusion_matrices()
    comparison_df = pipeline.create_comparison_table()

    # Save models
    pipeline.save_models()

    return pipeline, comparison_df


if __name__ == "__main__":
    # Load preprocessed data
    with open('data/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Train all models
    pipeline, comparison_df = train_all_traditional_models(data)

    print("\nTraditional ML models training completed!")