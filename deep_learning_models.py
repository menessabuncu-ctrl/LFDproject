"""
Deep Learning Models Implementation
Learning from Data - Final Project
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time


class DeepLearningPipeline:
    def __init__(self, max_words=10000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        self.models = {}
        self.histories = {}
        self.results = {}

    def prepare_sequences(self, texts_train, texts_val, texts_test):
        """Prepare sequences for deep learning models"""
        print("Preparing sequences...")

        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts_train)

        X_train_seq = self.tokenizer.texts_to_sequences(texts_train)
        X_val_seq = self.tokenizer.texts_to_sequences(texts_val)
        X_test_seq = self.tokenizer.texts_to_sequences(texts_test)

        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len, padding='post', truncating='post')
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_len, padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len, padding='post', truncating='post')

        return X_train_pad, X_val_pad, X_test_pad

    def build_mlp(self, input_dim, num_classes=3):
        """
        Multi-Layer Perceptron
        """
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_lstm(self, num_classes=3):
        """
        LSTM Network
        """
        model = models.Sequential([
            layers.Embedding(self.max_words, 128, input_length=self.max_len),
            layers.SpatialDropout1D(0.2),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_cnn(self, num_classes=3):
        """
        CNN for Text Classification
        """
        model = models.Sequential([
            layers.Embedding(self.max_words, 128, input_length=self.max_len),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_mlp(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train MLP model"""
        print("\n" + "=" * 60)
        print("Training Multi-Layer Perceptron (MLP)")
        print("=" * 60)

        model = self.build_mlp(X_train.shape[1])

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        train_time = time.time() - start_time

        self.models['mlp'] = model
        self.histories['mlp'] = history.history

        # Evaluate
        y_train_pred = np.argmax(model.predict(X_train), axis=1)
        y_val_pred = np.argmax(model.predict(X_val), axis=1)

        results = self._calculate_metrics(
            y_train, y_train_pred, y_val, y_val_pred,
            'MLP', train_time
        )
        self.results['mlp'] = results

        return model, history

    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=64):
        """Train LSTM model"""
        print("\n" + "=" * 60)
        print("Training LSTM")
        print("=" * 60)

        model = self.build_lstm()

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        train_time = time.time() - start_time

        self.models['lstm'] = model
        self.histories['lstm'] = history.history

        y_train_pred = np.argmax(model.predict(X_train), axis=1)
        y_val_pred = np.argmax(model.predict(X_val), axis=1)

        results = self._calculate_metrics(
            y_train, y_train_pred, y_val, y_val_pred,
            'LSTM', train_time
        )
        self.results['lstm'] = results

        return model, history

    def train_cnn(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=64):
        """Train CNN model"""
        print("\n" + "=" * 60)
        print("Training CNN")
        print("=" * 60)

        model = self.build_cnn()

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        train_time = time.time() - start_time

        self.models['cnn'] = model
        self.histories['cnn'] = history.history

        y_train_pred = np.argmax(model.predict(X_train), axis=1)
        y_val_pred = np.argmax(model.predict(X_val), axis=1)

        results = self._calculate_metrics(
            y_train, y_train_pred, y_val, y_val_pred,
            'CNN', train_time
        )
        self.results['cnn'] = results

        return model, history

    def _calculate_metrics(self, y_train, y_train_pred, y_val, y_val_pred, model_name, train_time):
        """Calculate metrics"""
        from sklearn.metrics import precision_recall_fscore_support

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
        print(f"F1-Score: {f1:.4f}")

        return results

    def plot_learning_curves(self, save_path='results/learning_curves.png'):
        """Plot learning curves for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (model_name, history) in enumerate(self.histories.items()):
            axes[idx].plot(history['accuracy'], label='Train Accuracy')
            axes[idx].plot(history['val_accuracy'], label='Val Accuracy')
            axes[idx].plot(history['loss'], label='Train Loss')
            axes[idx].plot(history['val_loss'], label='Val Loss')
            axes[idx].set_title(f'{model_name.upper()} Learning Curves')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Metric')
            axes[idx].legend()
            axes[idx].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_models(self, filepath='models/deep_learning_models.pkl'):
        """Save models"""
        for model_name, model in self.models.items():
            model.save(f'models/{model_name}_model.h5')

        with open(filepath, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'histories': self.histories,
                'tokenizer': self.tokenizer
            }, f)

        print(f"\nModels saved!")


def train_all_deep_learning_models(data):
    """Train all deep learning models"""
    X_train_text = data['raw_text']['X_train']
    X_val_text = data['raw_text']['X_val']
    X_test_text = data['raw_text']['X_test']

    y_train = data['labels']['y_train']
    y_val = data['labels']['y_val']
    y_test = data['labels']['y_test']

    pipeline = DeepLearningPipeline()

    # Prepare sequences for LSTM and CNN
    X_train_seq, X_val_seq, X_test_seq = pipeline.prepare_sequences(
        X_train_text, X_val_text, X_test_text
    )

    # Train MLP on TF-IDF features
    X_train_tfidf = data['features']['tfidf']['X_train'].toarray()
    X_val_tfidf = data['features']['tfidf']['X_val'].toarray()
    pipeline.train_mlp(X_train_tfidf, y_train, X_val_tfidf, y_val, epochs=20)

    # Train LSTM
    pipeline.train_lstm(X_train_seq, y_train, X_val_seq, y_val, epochs=15)

    # Train CNN
    pipeline.train_cnn(X_train_seq, y_train, X_val_seq, y_val, epochs=15)

    # Plot learning curves
    pipeline.plot_learning_curves()

    # Save models
    pipeline.save_models()

    return pipeline


if __name__ == "__main__":
    # Load preprocessed data
    with open('data/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Train all deep learning models
    pipeline = train_all_deep_learning_models(data)

    print("\nDeep learning models training completed!")