from flask import Flask, request, jsonify
from qa_engine import QAEngine

app = Flask(__name__)
qa_engine = QAEngine()

@app.route('/health')
def health():
    return jsonify({"ok": True, "mode": "api"})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    answer, conf = qa_engine.answer_question(question)
    return jsonify({"answer": answer, "confidence": conf})

@app.route('/reload', methods=['POST'])
def reload():
    qa_engine.reload_products()
    return jsonify({"ok": True})
