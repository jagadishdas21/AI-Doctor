from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import chatbot_response  # Import your chatbot function

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    user_message = request.args.get('msg')
    print(f"User message received: {user_message}")  # Debugging line
    response = chatbot_response(user_message)
    print(f"Bot response: {response}")  # Debugging line
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode