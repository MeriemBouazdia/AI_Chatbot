"""
Test script to verify model, tokenizer, and config loading.
Run this to confirm all files are properly loaded.
"""
import tensorflow as tf
import pickle
import json

print("=" * 50)
print("Testing Model Loading")
print("=" * 50)

# Load model
print("\nLoading chatbot_model.keras...")
model = tf.keras.models.load_model("chatbot_model.keras")
print("* Model loaded successfully!")

# Load tokenizer
print("\nLoading tokenizer.pkl...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("* Tokenizer loaded successfully!")
print(f"  - Vocabulary size: {len(tokenizer.word_index)}")

# Load config (replaces label_encoder)
print("\nLoading config.json...")
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
print("* Config loaded successfully!")
print(f"  - Number of intents: {len(config['intent_to_idx'])}")
print(f"  - Max sequence length: {config['max_length']}")
print(f"  - Available intents: {list(config['intent_to_idx'].keys())}")

print("\n" + "=" * 50)
print("All files loaded successfully!")
print("=" * 50)
