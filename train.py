"""
Training module for English chatbot model.
Uses TensorFlow/Keras for the neural network and sklearn for data preprocessing.
"""

import os
import json
import numpy as np
import pickle

# FIX #1: Import TensorFlow/Keras correctly
# Using tensorflow.keras (not legacy tf.keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, 
    LSTM, 
    Dense, 
    Dropout, 
    Bidirectional,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

# FIX #2: Import sklearn correctly for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Import our preprocessing module
from preprocess import (
    preprocess_dataset, 
    create_tokenizer, 
    encode_sequences,
    normalize_text
)


def build_model(vocab_size, max_length, num_classes):
    """
    Build a Bidirectional LSTM model for intent classification.
    
    FIX #3: Correct Keras model architecture
    
    Args:
        vocab_size: Size of the vocabulary
        max_length: Maximum sequence length
        num_classes: Number of intent classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # FIX #3a: Embedding layer for text representation
        # Input dimension is vocab_size + 1 (for OOV token)
        Embedding(input_dim=vocab_size + 1, output_dim=128, input_length=max_length),
        
        # FIX #3b: Bidirectional LSTM for better context understanding
        # return_sequences=False for the last LSTM layer
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        
        # FIX #3c: Batch normalization for training stability
        BatchNormalization(),
        
        # FIX #3d: Dense layers for classification
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # FIX #3e: Output layer - softmax for multi-class
        # Use 'softmax' for multi-class, 'sigmoid' for multi-label
        Dense(num_classes, activation='softmax')
    ])
    
    # FIX #3f: Compile with correct optimizer and loss
    # For multi-class classification: loss='sparse_categorical_crossentropy'
    # (when labels are integers, not one-hot encoded)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',  # FIX: correct loss function
        metrics=['accuracy']
    )
    
    return model


def train_model(training_data, epochs=100, batch_size=16, max_length=50):
    """
    Train the chatbot model.
    
    FIX #4: Proper training with correct data handling
    
    Args:
        training_data: Preprocessed training data dictionary
        epochs: Number of training epochs
        batch_size: Batch size for training
        max_length: Maximum sequence length
        
    Returns:
        Trained model and training history
    """
    # Extract data
    X_train = training_data['X_train']
    X_test = training_data['X_test']
    y_train = training_data['y_train']
    y_test = training_data['y_test']
    
    vocab_size = len(training_data['tokenizer'].word_index)
    num_classes = training_data['num_classes']
    
    print(f"\nTraining configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Max sequence length: {max_length}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Build model
    model = build_model(vocab_size, max_length, num_classes)
    
    # Print model summary
    print("\nModel architecture:")
    model.summary()
    
    # FIX #5: Define callbacks for better training
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            'chatbot_model.keras',  # FIX: use .keras extension (not .h5)
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # FIX #6: Train with proper validation data
    # Using sparse_categorical_crossentropy, so labels should be integers
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, training_data):
    """
    Evaluate the trained model using sklearn metrics.
    
    FIX #7: Proper model evaluation
    
    Args:
        model: Trained Keras model
        training_data: Training data dictionary
        
    Returns:
        Evaluation metrics dictionary
    """
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    
    # Predict on test set
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # FIX #7a: Classification report using sklearn
    target_names = list(training_data['intent_to_idx'].keys())
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=target_names,
        output_dict=True
    )
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return {
        'accuracy': accuracy,
        'classification_report': report
    }


def save_model_artifacts(model, training_data):
    """
    Save model and all necessary artifacts for inference.
    
    FIX #8: Proper model saving
    
    Args:
        model: Trained Keras model
        training_data: Training data dictionary
    """
    # Save the Keras model
    model.save('chatbot_model.keras')  # FIX: use .keras format
    print("\nModel saved to: chatbot_model.keras")
    
    # Save tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(training_data['tokenizer'], f)
    print("Tokenizer saved to: tokenizer.pkl")
    
    # Save intent mappings and config
    config = {
        'max_length': training_data['max_length'],
        'num_classes': training_data['num_classes'],
        'intent_to_idx': training_data['intent_to_idx'],
        'idx_to_intent': training_data['idx_to_intent'],
        'responses': training_data['responses']
    }
    
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("Config saved to: config.json")
    
    print("\nAll artifacts saved successfully!")


def main():
    """
    Main training pipeline.
    """
    print("="*60)
    print("ENGLISH CHATBOT MODEL TRAINING")
    print("="*60)
    
    # Step 1: Preprocess the dataset
    print("\n[1/4] Loading and preprocessing dataset...")
    data = preprocess_dataset('dataset.json')
    print(f"  Loaded {len(data['questions'])} questions")
    print(f"  Found {data['num_classes']} unique intents")
    
    # Step 2: Prepare training data
    print("\n[2/4] Preparing training data...")
    max_length = 50
    training_data = prepare_training_data(data, max_length=max_length)
    print(f"  Training set: {len(training_data['X_train'])} samples")
    print(f"  Test set: {len(training_data['X_test'])} samples")
    
    # Step 3: Train the model
    print("\n[3/4] Training model...")
    model, history = train_model(
        training_data, 
        epochs=100, 
        batch_size=16,
        max_length=max_length
    )
    
    # Step 4: Evaluate and save
    print("\n[4/4] Evaluating and saving model...")
    evaluation = evaluate_model(model, training_data)
    save_model_artifacts(model, training_data)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {evaluation['accuracy']:.4f}")
    print("="*60)


if __name__ == '__main__':
    # Import the prepare_training_data function
    from preprocess import prepare_training_data
    
    main()
