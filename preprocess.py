import json
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# -----------------------------
def normalize_text(text):
    """Normalize text: lowercase, remove punctuation, extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# -----------------------------
def load_dataset(filepath='dataset.json'):
    """Load dataset JSON and extract questions, intents, and responses."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []
    intents = []
    responses = []

    for intent in data.get('intents', []):
        for question in intent.get('questions', []):
            questions.append(question)
            intents.append(intent['id'])
        intent_responses = [resp['answer'] for resp in intent.get('responses', [])]
        responses.append(intent_responses)

    return questions, intents, responses

# -----------------------------
def preprocess_dataset(filepath='dataset.json'):
    """Normalize questions and create mappings for intents."""
    questions, intents, responses = load_dataset(filepath)
    normalized_questions = [normalize_text(q) for q in questions]

    unique_intents = sorted(list(set(intents)))
    intent_to_idx = {intent: idx for idx, intent in enumerate(unique_intents)}
    idx_to_intent = {idx: intent for intent, idx in intent_to_idx.items()}

    intent_indices = [intent_to_idx[intent] for intent in intents]

    return {
        'questions': normalized_questions,
        'intents': intents,
        'intent_indices': intent_indices,
        'responses': responses,
        'intent_to_idx': intent_to_idx,
        'idx_to_intent': idx_to_intent,
        'num_classes': len(unique_intents)
    }

# -----------------------------
def create_tokenizer(questions, num_words=5000):
    """Fit a tokenizer on the questions."""
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(questions)
    return tokenizer

# -----------------------------
def encode_sequences(tokenizer, questions, max_length=50):
    """Convert text questions to padded sequences of integers."""
    sequences = tokenizer.texts_to_sequences(questions)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded

# -----------------------------
def prepare_training_data(data, max_length=50, test_size=0.2, random_state=42):
    """Prepare training and test datasets for model training."""
    tokenizer = create_tokenizer(data['questions'])
    X = encode_sequences(tokenizer, data['questions'], max_length)
    y = np.array(data['intent_indices'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'tokenizer': tokenizer,
        'max_length': max_length,
        'num_classes': data['num_classes'],
        'intent_to_idx': data['intent_to_idx'],
        'idx_to_intent': data['idx_to_intent'],
        'responses': data['responses']
    }

# -----------------------------
# Testing
if __name__ == '__main__':
    test_texts = [
        "Hello, how are you?",
        "مرحبا كيف حالك؟",
        "What's the weather like today?",
        "كيف الطقس اليوم؟"
    ]
    
    print("Testing normalization:")
    for text in test_texts:
        normalized = normalize_text(text)
        print(f"  {text} → {normalized}")

    print("\nLoading and preprocessing dataset...")
    data = preprocess_dataset('dataset.json')
    print(f"  Total questions: {len(data['questions'])}")
    print(f"  Unique intents: {data['num_classes']}")
    print(f"  Intent mapping: {data['intent_to_idx']}")