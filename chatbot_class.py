import os
import json
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import detect_language
from preprocess import normalize_text

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    print("Translation library not available, fallback to English only")
    GoogleTranslator = None
    TRANSLATOR_AVAILABLE = False

# تحميل dataset
with open("dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

class SmartChatbot:
    def __init__(self, model_path='chatbot_model.keras',
                 tokenizer_path='tokenizer.pkl',
                 config_path='config.json'):
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.max_length = self.config['max_length']
        self.intent_to_idx = self.config['intent_to_idx']
        self.idx_to_intent = {str(k): v for k, v in self.config['idx_to_intent'].items()}
        # حذف enumerate، استخدم dataset مباشرة
        self.responses = self.config['responses']
        self.confidence_threshold = 0.6

        print(f"Chatbot initialized. Intents: {len(self.intent_to_idx)}, Max length: {self.max_length}")

    def preprocess_input(self, user_input):
        normalized = normalize_text(user_input)
        sequence = self.tokenizer.texts_to_sequences([normalized])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
        return padded

    def predict_intent(self, user_input):
        processed_input = self.preprocess_input(user_input)
        predictions = self.model.predict(processed_input, verbose=0)[0]
        intent_idx = int(np.argmax(predictions))
        confidence = float(predictions[intent_idx])
        intent = self.idx_to_intent[str(intent_idx)]
        return intent, confidence, intent_idx

    def get_response(self, intent_idx, **sensor_data):
        # اسم الـ intent الحقيقي
        intent_name = self.idx_to_intent[str(intent_idx)]
        intent = next((i for i in data['intents'] if i['id'] == intent_name), None)

        if intent:
            response = random.choice(intent['responses'])
            text = response['answer']
            # استبدال أي placeholder بالقيم الحية
            for key, value in sensor_data.items():
                placeholder = f"{{{key}}}"
                text = text.replace(placeholder, str(value))
            return text

        # fallback لو ما تعرفش
        fallback = next(intent for intent in data['intents'] if intent['id'] == "fallback")
        return random.choice(fallback['responses'])['answer']

    def chat(self, user_input, **sensor_data):
        intent, confidence, intent_idx = self.predict_intent(user_input)

        if confidence < self.confidence_threshold:
            fallback_idx = self.intent_to_idx.get('fallback', 0)
            response = self.get_response(fallback_idx, **sensor_data)
        else:
            response = self.get_response(intent_idx, **sensor_data)

        user_lang = detect_language(user_input)[:2]
        if user_lang not in ['en', 'ar', 'fr']:
            user_lang = 'en'

        if user_lang != 'en':
            try:
                response = GoogleTranslator(source='en', target=user_lang).translate(response)
            except Exception as e:
                print(f"Translation error: {e}")

        return response, intent, confidence, user_lang

    def interactive_chat(self):
        print("Smart Greenhouse Chatbot (type 'quit' to exit)")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            if not user_input:
                continue
            response, intent, confidence, _ = self.chat(user_input, **simulate_sensor_data())
            print(f"Bot: {response} (Intent: {intent}, Confidence: {confidence:.2f})")


def simulate_sensor_data():
    return {
        'temperature': random.randint(18, 32),
        'humidity': random.randint(40, 90),
        'light': random.randint(5000, 35000),
        'soil': random.randint(30, 70),
    }


def main():
    files = ['chatbot_model.keras', 'tokenizer.pkl', 'config.json']
    if not all(os.path.exists(f) for f in files):
        print("ERROR: Model files not found!")
        return

    chatbot = SmartChatbot()
    test_inputs = ["Hello", "What's the temperature?", "Comment ça va?", "متى نسقي النباتات؟"]

    for inp in test_inputs:
        response, intent, conf, _ = chatbot.chat(inp, **simulate_sensor_data())
        print(f"Input: {inp}\nResponse: {response}\nIntent: {intent} (Confidence: {conf:.2f})\n{'-'*40}")

    chatbot.interactive_chat()


if __name__ == '__main__':
    main()