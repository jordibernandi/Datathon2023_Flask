from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import List, Union, Dict, Set, Tuple
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import nltk
import re

app = Flask(__name__)
cors = CORS(app)


@app.route("/receiver", methods=["POST"])
def postME():
    data = request.get_json()
    result = re.search('.*(\[.*\]).*', data)
    symptoms = []
    if result != None:
        symptoms = [x[1:-1] for x in result.group(1)[1:-1].split(', ')]
    
    nltk.download("punkt")
    nltk.download('wordnet')

    df = pd.read_csv("nhs_health_symptoms.csv")

    word2index, doc2index, index2doc, doc2word_vector = process_df(df)
    td_matrix = term_document_matrix(doc2word_vector, doc2index, word2index)
    idx = most_similar_docs(symptoms, word2index, td_matrix, 1)
    
    return "You have {}. Further information are {}".format(df.loc[df.index[idx]]["diagnosis"].to_list()[0], df.loc[df.index[idx]]["information"].to_list()[0])


def process_df(df: pd.DataFrame) -> (Dict[str, int], Dict[str, int], Dict[int, str], Dict[str, List[int]]):
    """
    :params docs: df that maps each document (diagnosis) to the symptoms
    :returns:
        - word2index: dict that maps each word to a unique id
        - doc2index: dict that maps each diagnosis to a unique id
        - index2doc: dict that maps ids to their associated diagnosis
        - doc2word_vector: dict that maps each document name to a list of word ids that appear in it
    """
    word2index = {}
    doc2index = {}
    index2doc = {}
    doc2word_vector = {}
    
    digit = re.compile(r"\d+")
    
    for index, row in df.iterrows():
        doc2index[row["diagnosis"]] = index
        index2doc[index] = row["diagnosis"]
        doc2word_vector[row["diagnosis"]] = []
        
        for word in nltk.tokenize.word_tokenize(row["symptoms"]):
            # remove non-characters
            word = re.sub(r'[^\w\s]', '', word.lower())
            if not word:
                continue
            # remove digits
            if re.fullmatch(digit, word):
                continue
            # lemmatization
            lemmmatizer = nltk.stem.WordNetLemmatizer()
            word = lemmmatizer.lemmatize(word)
            
            if word not in word2index:
                word2index[word] = len(word2index)
            doc2word_vector[row["diagnosis"]].append(word2index[word])
    
    return word2index, doc2index, index2doc, doc2word_vector


def term_document_matrix(doc2word_vector: Dict[str, List[int]], doc2index: Dict[str, int], word2index: Dict[str, int]) -> NDArray[NDArray[int]]:
    """
    :param doc_word_v: dict that maps each document to the list of word ids that appear in it
    :param doc2index: dict that maps each document name to a unique id
    :param word2index: dict that maps each word to a unique id
    :return: term-document matrix (each word is a row, each document is a column) that indicates word count
    """
    td_matrix = np.zeros((len(word2index), len(doc2index)), dtype=int)
    for key, val in doc2word_vector.items():
        for idx in val:
            td_matrix[idx, doc2index[key]] += 1
    return td_matrix


def to_tf_idf_matrix(td_matrix: NDArray[NDArray[int]]) -> NDArray[NDArray[float]]:
    """
    :param td_matrix: term-document matrix 
    :return: matrix with tf-idf values for each word-document pair 
    """
    rows, cols = td_matrix.shape
    tf_idf_matrix = np.zeros((rows, cols))
    for x in range(rows):
        for y in range(cols):
            if td_matrix[x, y] > 0:
                tf_idf_matrix[x, y] = 1 + np.log10(td_matrix[x, y])
    
    idf = np.log10(cols / np.sum(td_matrix > 0, axis=1))
    tf_idf_matrix = tf_idf_matrix * idf[:, None]
    return tf_idf_matrix


def most_similar_docs(query: List[str], word2index: Dict[str, int], td_matrix: NDArray[NDArray[float]], top_k: int) -> List[int]:
    """
    :param query: list of query tokens
    :param word2index: dict that maps each word to a unique id
    :param td_matrix: term-document matrix 
    :param top_k: number of top-k similar documents to return
    :return: list with names of the top-k most similar documents
    """
    # lemmatization
    lemmmatizer = nltk.stem.WordNetLemmatizer()
    query = [lemmmatizer.lemmatize(token) for token in query]
    
    # create a vector that has an entry for each vocabulary word
    representation = np.zeros(len(word2index))
    for token in query:
        if token in word2index:
            representation[word2index[token]] += 1
    
    representation = [1 + np.log10(x) if x > 0 else 0 for x in representation]
    idf_val = np.log10(td_matrix.shape[1] / np.sum(td_matrix > 0, axis=1))
    representation = representation * idf_val
    
    tf_idf_matrix = to_tf_idf_matrix(td_matrix)
    
    # cosine similarities
    sims = np.dot(representation, tf_idf_matrix) / ((np.linalg.norm(representation, ord=2) * np.linalg.norm(tf_idf_matrix, ord=2, axis=0)))
    # sort in descending order
    idx = np.argsort(-sims, )
    return idx[:top_k]


if __name__ == "__main__":
    app.run(debug=True)
