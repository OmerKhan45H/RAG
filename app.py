from flask import Flask, render_template, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import os

app = Flask(__name__)

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Groq(model="llama3-70b-8192", api_key="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")

# Initialize the RAG system
try:
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
except Exception as e:
    print(f"Error initializing RAG system: {str(e)}")
    # Create a fallback query engine that returns a default message
    class FallbackQueryEngine:
        def query(self, text):
            return "I apologize, but I'm having trouble accessing the knowledge base. Please try again later."
    query_engine = FallbackQueryEngine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'response': 'Please enter a message'})
        
        # Get response from the query engine
        response = query_engine.query(user_message)
        return jsonify({'response': str(response)})
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'response': 'I apologize, but something went wrong. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 