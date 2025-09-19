from flask import Flask, request, jsonify
from keybert import KeyBERT
from flask_cors import CORS
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
CORS(app)

kw_model = KeyBERT('paraphrase-MiniLM-L6-v2')
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Same model as KeyBERT

@app.route('/process_texts', methods=['POST'])
def process_texts():
    data = request.get_json()
    texts = data.get('texts', [])
    num_clusters = data.get('num_clusters', None)  # Optional parameter

    # Compute embeddings
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)

    # Convert embeddings to numpy array
    embeddings_np = embeddings.cpu().detach().numpy()

    # Determine the number of clusters if not provided
    if not num_clusters:
        num_clusters = max(2, len(texts) // 5)  # Adjust as needed

    # Perform clustering
    clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = clustering_model.fit_predict(embeddings_np)

    # Extract keywords
    keywords_list = []
    for text in texts:
        if not text:
            keywords_list.append([])
            continue
        keywords = kw_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=5
        )
        keywords_only = [kw for kw, score in keywords]
        keywords_list.append(keywords_only)

    return jsonify({'keywords_list': keywords_list, 'cluster_labels': cluster_labels.tolist()})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
