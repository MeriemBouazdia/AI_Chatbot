import os
import random
from flask import Flask, request, jsonify
from chatbot_class import SmartChatbot, simulate_sensor_data
from utils import detect_language

# Translation setup
try:
    from deep_translator import GoogleTranslator
    print("Using deep_translator for translations")
    TRANSLATOR_AVAILABLE = True
except ImportError:
    print("Translation library not available - fallback to English only")
    GoogleTranslator = None
    TRANSLATOR_AVAILABLE = False


def translate_to_english(text: str) -> str:
    if GoogleTranslator:
        try:
            translator = GoogleTranslator(source='auto', target='en')
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
    return text


def translate_from_english(text: str, target_lang: str) -> str:
    if target_lang == 'en' or not GoogleTranslator:
        return text
    try:
        translator = GoogleTranslator(source='en', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# Initialize Flask & Chatbot
app = Flask(__name__)

chatbot = SmartChatbot(
    model_path='chatbot_model.keras',
    tokenizer_path='tokenizer.pkl',
    config_path='config.json'
)


# API route
@app.route("/chat", methods=["POST"])
def chat_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        user_input = (data.get("message") or data.get("user_input") or "").strip()
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        sensor_data = data.get("sensor_data", simulate_sensor_data())

        # y3rf user language
        detected_lang = (detect_language(user_input) or 'en')[:2]
        if detected_lang not in ['en', 'fr', 'ar']:
            detected_lang = 'en'

       
        english_input = translate_to_english(user_input) if detected_lang != 'en' else user_input

      
        response, intent, confidence, _ = chatbot.chat(english_input, **sensor_data)

        
        final_response = translate_from_english(response, detected_lang) if detected_lang != 'en' else response

        return jsonify({
            "response": final_response,
            "intent": intent,
            "confidence": confidence,
            "detected_language": detected_lang
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Smart Greenhouse Chatbot is running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)