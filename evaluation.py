"""
Comprehensive Model Evaluation and Comparison
Learning from Data - Final Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import pickle


class ModelEvaluator:
    def __init__(self):
        self.all_results = {}

    def load_all_models(self):
        """Load all trained models"""
        with open('models/traditional_models.pkl', 'rb') as f:
            traditional = pickle.load(f)

        with open('models/deep_learning_models.pkl', 'rb') as f:
            deep_learning = pickle.load(f)

        # Combine results
        self.all_results = {**traditional['results'], **deep_learning['results']}

        return traditional, deep_learning

    def create_comprehensive_comparison(self):
        """Create comprehensive comparison table"""
        comparison_data = []

        for model_name, results in self.all_results.items():
            comparison_data.append({
                'Model': results['model_name'],
                'Train Accuracy': results['train_accuracy'],
                'Val Accuracy': results['val_accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'Training Time (s)': results['train_time'],
                'Overfitting': results['train_accuracy'] - results['val_accuracy']
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Val Accuracy', ascending=False)

        print("\n" + "=" * 100)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)

        # Save to CSV
        df.to_csv('results/model_comparison.csv', index=False)

        return df

    def plot_model_comparison(self, comparison_df, save_path='results/model_comparison.png'):
        """Create visualization comparing all models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        models = comparison_df['Model']

        # Accuracy comparison
        x = np.arange(len(models))
        width = 0.35

        axes[0, 0].bar(x - width / 2, comparison_df['Train Accuracy'], width, label='Train', alpha=0.8)
        axes[0, 0].bar(x + width / 2, comparison_df['Val Accuracy'], width, label='Validation', alpha=0.8)
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Training vs Validation Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # F1-Score comparison
        axes[0, 1].bar(models, comparison_df['F1-Score'], color='skyblue', alpha=0.8)
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_title('F1-Score Comparison')
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Training time comparison
        axes[1, 0].bar(models, comparison_df['Training Time (s)'], color='coral', alpha=0.8)
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Overfitting analysis
        colors = ['green' if x < 0.05 else 'orange' if x < 0.1 else 'red'
                  for x in comparison_df['Overfitting']]
        axes[1, 1].bar(models, comparison_df['Overfitting'], color=colors, alpha=0.8)
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Train - Val Accuracy')
        axes[1, 1].set_title('Overfitting Analysis')
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].axhline(y=0.05, color='orange', linestyle='--', label='Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_all_confusion_matrices(self, save_path='results/all_confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        n_models = len(self.all_results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for idx, (model_name, results) in enumerate(self.all_results.items()):
            cm = results['confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues',
                        xticklabels=['Negative', 'Neutral', 'Positive'],
                        yticklabels=['Negative', 'Neutral', 'Positive'])

            axes[idx].set_title(f"{results['model_name']}\nF1: {results['f1_score']:.3f}")
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_errors(self, data, model, model_name='Best Model'):
        """Analyze misclassifications"""
        X_test_text = data['raw_text']['X_test']
        y_test = data['labels']['y_test']

        # Get predictions based on model type
        if model_name in ['MLP', 'LSTM', 'CNN']:
            if model_name == 'MLP':
                X_test_features = data['features']['tfidf']['X_test'].toarray()
                y_pred = np.argmax(model.predict(X_test_features), axis=1)
            else:
                # For LSTM/CNN, need to prepare sequences
                from tensorflow.keras.preprocessing.text import Tokenizer
                from tensorflow.keras.preprocessing.sequence import pad_sequences

                with open('models/deep_learning_models.pkl', 'rb') as f:
                    dl_data = pickle.load(f)
                    tokenizer = dl_data['tokenizer']

                X_test_seq = tokenizer.texts_to_sequences(X_test_text)
                X_test_pad = pad_sequences(X_test_seq, maxlen=200, padding='post')
                y_pred = np.argmax(model.predict(X_test_pad), axis=1)
        else:
            X_test_features = data['features']['tfidf']['X_test']
            y_pred = model.predict(X_test_features)

        # Find misclassifications
        misclassified_idx = np.where(y_pred != y_test)[0]

        print(f"\n{'=' * 80}")
        print(f"ERROR ANALYSIS FOR {model_name}")
        print(f"{'=' * 80}")
        print(f"Total test samples: {len(y_test)}")
        print(f"Misclassified: {len(misclassified_idx)} ({len(misclassified_idx) / len(y_test) * 100:.2f}%)")

        # Show examples
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        print("\nSample Misclassifications:")
        print("-" * 80)

        for i in misclassified_idx[:10]:  # Show first 10
            print(f"\nText: {X_test_text[i][:150]}...")
            print(f"True Label: {sentiment_map[y_test[i]]}")
            print(f"Predicted: {sentiment_map[y_pred[i]]}")
            print("-" * 80)

    def create_final_report(self, comparison_df):
        """Create final analysis report"""
        report = []

        report.append("=" * 100)
        report.append("FINAL PROJECT REPORT - LEARNING FROM DATA")
        report.append("Amazon Product Review Sentiment Analysis")
        report.append("=" * 100)

        report.append("\n1. BEST PERFORMING MODEL")
        report.append("-" * 100)
        best_model = comparison_df.iloc[0]
        report.append(f"Model: {best_model['Model']}")
        report.append(f"Validation Accuracy: {best_model['Val Accuracy']:.4f}")
        report.append(f"F1-Score: {best_model['F1-Score']:.4f}")
        report.append(f"Training Time: {best_model['Training Time (s)']:.2f}s")

        report.append("\n2. BIAS-VARIANCE ANALYSIS")
        report.append("-" * 100)
        for _, row in comparison_df.iterrows():
            overfitting = row['Overfitting']
            status = "Good Fit" if overfitting < 0.05 else "Slight Overfit" if overfitting < 0.1 else "Overfitting"
            report.append(f"{row['Model']}: {status} (gap: {overfitting:.4f})")

        report.append("\n3. COMPUTATIONAL EFFICIENCY")
        report.append("-" * 100)
        fastest = comparison_df.loc[comparison_df['Training Time (s)'].idxmin()]
        report.append(f"Fastest Model: {fastest['Model']} ({fastest['Training Time (s)']:.2f}s)")
        slowest = comparison_df.loc[comparison_df['Training Time (s)'].idxmax()]
        report.append(f"Slowest Model: {slowest['Model']} ({slowest['Training Time (s)']:.2f}s)")

        report.append("\n4. RECOMMENDATIONS")
        report.append("-" * 100)
        report.append("- For production deployment: Use Logistic Regression or Linear SVM (fast, accurate)")
        report.append("- For maximum accuracy: Use the best deep learning model")
        report.append("- For interpretability: Use Decision Tree or Logistic Regression")

        report_text = "\n".join(report)
        print(report_text)

        # Save report
        with open('results/final_report.txt', 'w') as f:
            f.write(report_text)

        return report_text


def evaluate_all_models(data):
    """Complete evaluation pipeline"""
    evaluator = ModelEvaluator()

    # Load all models
    traditional, deep_learning = evaluator.load_all_models()

    # Create comparison
    comparison_df = evaluator.create_comprehensive_comparison()

    # Visualizations
    evaluator.plot_model_comparison(comparison_df)
    evaluator.plot_all_confusion_matrices()

    # Error analysis on best model
    best_model_name = comparison_df.iloc[0]['Model']
    if best_model_name in traditional['models']:
        best_model = traditional['models'][list(traditional['models'].keys())[0]]
        evaluator.analyze_errors(data, best_model, best_model_name)

    # Final report
    evaluator.create_final_report(comparison_df)

    return evaluator, comparison_df


if __name__ == "__main__":
    # Load data
    with open('data/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Evaluate all models
    evaluator, comparison_df = evaluate_all_models(data)

    print("\nEvaluation completed! Check 'results/' folder for outputs.")