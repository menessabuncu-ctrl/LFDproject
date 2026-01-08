"""
Main Execution Script - Learning from Data Final Project
Amazon Product Review Sentiment Analysis

Author: [Your Name]
Course: Learning from Data
Instructor: Cumali Türkmenoğlu
Date: December 2025
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def main():
    """Main execution pipeline"""

    print_banner("LEARNING FROM DATA - FINAL PROJECT")
    print_banner("Amazon Product Review Sentiment Analysis")

    print("This pipeline will:")
    print("1. Collect/Load data (3000+ reviews)")
    print("2. Preprocess and extract features")
    print("3. Train 6 ML/DL models (Logistic Regression, SVM, Random Forest, MLP, LSTM, CNN)")
    print("4. Evaluate and compare all models")
    print("5. Generate comprehensive report\n")

    choice = input("Do you want to continue? (y/n): ")
    if choice.lower() != 'y':
        print("Exiting...")
        return

    # Step 1: Data Collection
    print_banner("STEP 1: DATA COLLECTION")
    try:
        from scraper import create_sample_data, AmazonReviewScraper

        print("Generating sample dataset...")
        df = create_sample_data(3000)

        scraper = AmazonReviewScraper()
        df = scraper.create_sentiment_labels(df)
        df.to_csv('data/raw_reviews.csv', index=False)

        print(f"✓ Dataset created: {len(df)} samples")
        print(f"✓ Class distribution:")
        print(df['sentiment_label'].value_counts())

    except Exception as e:
        print(f"✗ Error in data collection: {e}")
        return

    # Step 2: Data Preprocessing
    print_banner("STEP 2: DATA PREPROCESSING & FEATURE EXTRACTION")
    try:
        from preprocessing import prepare_data

        data = prepare_data('data/raw_reviews.csv')

        import pickle
        with open('data/preprocessed_data.pkl', 'wb') as f:
            pickle.dump(data, f)

        print("✓ Text preprocessing completed")
        print("✓ Features extracted: BoW and TF-IDF")
        print("✓ Data split: Train/Val/Test")

    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        return

    # Step 3: Train Traditional ML Models
    print_banner("STEP 3: TRAINING TRADITIONAL ML MODELS")
    try:
        from traditional_models import train_all_traditional_models

        with open('data/preprocessed_data.pkl', 'rb') as f:
            data = pickle.load(f)

        pipeline_trad, comparison_trad = train_all_traditional_models(data)

        print("✓ Logistic Regression trained")
        print("✓ Linear SVM trained")
        print("✓ RBF SVM trained")
        print("✓ Random Forest trained")
        print("✓ K-NN trained")

    except Exception as e:
        print(f"✗ Error in traditional ML training: {e}")
        return

    # Step 4: Train Deep Learning Models
    print_banner("STEP 4: TRAINING DEEP LEARNING MODELS")
    try:
        from deep_learning_models import train_all_deep_learning_models

        pipeline_dl = train_all_deep_learning_models(data)

        print("✓ MLP trained")
        print("✓ LSTM trained")
        print("✓ CNN trained")

    except Exception as e:
        print(f"✗ Error in deep learning training: {e}")
        return

    # Step 5: Comprehensive Evaluation
    print_banner("STEP 5: MODEL EVALUATION & COMPARISON")
    try:
        from evaluation import evaluate_all_models

        evaluator, comparison_df = evaluate_all_models(data)

        print("✓ All models evaluated")
        print("✓ Comparison tables created")
        print("✓ Visualizations generated")
        print("✓ Error analysis completed")

    except Exception as e:
        print(f"✗ Error in evaluation: {e}")
        return

    # Final Summary
    print_banner("PROJECT COMPLETED SUCCESSFULLY!")

    print("Generated Files:")
    print("├── data/")
    print("│   ├── raw_reviews.csv")
    print("│   └── preprocessed_data.pkl")
    print("├── models/")
    print("│   ├── traditional_models.pkl")
    print("│   ├── mlp_model.h5")
    print("│   ├── lstm_model.h5")
    print("│   └── cnn_model.h5")
    print("└── results/")
    print("    ├── data_analysis.png")
    print("    ├── confusion_matrices.png")
    print("    ├── learning_curves.png")
    print("    ├── model_comparison.png")
    print("    ├── model_comparison.csv")
    print("    └── final_report.txt")

    print("\n" + "="*80)
    print("Best Model:")
    best_model = comparison_df.iloc[0]
    print(f"  Model: {best_model['Model']}")
    print(f"  Validation Accuracy: {best_model['Val Accuracy']:.4f}")
    print(f"  F1-Score: {best_model['F1-Score']:.4f}")
    print("="*80)

    print("\nNext Steps:")
    print("1. Review the results/ folder for visualizations")
    print("2. Read final_report.txt for detailed analysis")
    print("3. Check model_comparison.csv for metrics")
    print("4. Use the best model for predictions")

    print("\nThank you!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()