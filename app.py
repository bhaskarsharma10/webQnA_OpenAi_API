from flask import Flask,request,render_template,jsonify
from newChat_prototype import answer_question, df

import openai

app = Flask(__name__)



conversation = []

@app.route('/')
def index():
    return render_template('chat.html', conversation=conversation)

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    conversation.append(f"You: {user_input}")
    
    if user_input.lower() == 'exit':
        conversation.append("Chatbot: Goodbye!")
    else:
        response = answer_question(df, question=user_input)
        chatbot_reply = response
        conversation.append(f"Chatbot: {chatbot_reply}")
    
    return render_template('chat.html', conversation=conversation)

if __name__ == '__main__':
    app.run(debug=True)
