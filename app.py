from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict, Set, Tuple
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import nltk
import re

app = Flask(__name__)
cors = CORS(app)

# for sentences embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("nhs_health_symptoms.csv")
diagnosis_embeddings_dict = dict()


for index, row in df.iterrows():
    # symptom sentences
    text = row["symptoms"]
    sentences = nltk.tokenize.sent_tokenize(text.replace("\n", " "))
    # print(sentences)

    # sentences embeddings
    embeddings = model.encode(sentences)
    diagnosis_embeddings_dict[row["diagnosis"].split(",")[0]] = embeddings


@app.route("/receiver", methods=["POST"])
def postME():
    data = request.get_json()
    query, response = data.split("@@")
    
    diagnosis = most_similar_diag(query, diagnosis_embeddings_dict)
    prompt = ""
    if diagnosis == "No diagnosis":
        prompt = "Rephrase this sentence in a formal way: {} Please describe more about your symptoms if you want any further diagnosis.".format(response)
    else:
        prompt = "Provide some information and give some advises about {}.".format(diagnosis)

    return jsonify(prompt)


def most_similar_diag(query: str, diagnosis_embeddings_dict: Dict[str, NDArray[NDArray[float]]]) -> str:
    """
    :param query: string of the symptoms of the user
    :param diagnosis_embeddings_dict: dict that maps each diagnosis to the sentence embeddings of the corresponding symptoms
    :return: most possible (similar symptoms) diagnosis
    """
    query_sentences = sentences = nltk.tokenize.sent_tokenize(query.replace("\n", " "))
    # print(query_sentences)
    
    # query sentences embeddings
    query_embeddings = model.encode(query_sentences)
    
    max_top_k_sims = -1
    prob_diagnosis = ""
    
    for diagnosis, diagnosis_embeddings in diagnosis_embeddings_dict.items():
        sims = []
        
        # pair-wise cosine similarity of each sentence embeddings pair
        for diagnosis_embedding in diagnosis_embeddings:
            for query_embedding in query_embeddings:
                # cosine similarities
                sim = np.dot(diagnosis_embedding, query_embedding) / (np.linalg.norm(diagnosis_embedding) * np.linalg.norm(query_embedding))
                sims.append(sim)
        
        # average similarity of top 5 sentences
        top_k = max(5, len(sims))
        top_k_sims = np.mean(np.sort(sims)[:top_k])
        # print(top_k_sims)
        if top_k_sims > max_top_k_sims:
            max_top_k_sims = top_k_sims
            prob_diagnosis = diagnosis
                
    return prob_diagnosis


if __name__ == "__main__":
    app.run(debug=True)
