import os
import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Import stop_words from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer


# Preprocess text
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords using scikit-learn's stop_words
    stop_words = set(ENGLISH_STOP_WORDS)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Stemming
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


# Read documents from a folder
def read_documents(folder_path):
    documents = []
    current_document = None

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()

                for line in lines:
                    if line.startswith("Document"):
                        if current_document is not None:
                            documents.append(current_document)
                        current_document = {'title': '', 'content': '', 'number': int(re.search(r'\d+', line).group())}
                    elif line.startswith('*' * 40):
                        continue  # Skip separator lines
                    else:
                        current_document['content'] += line.strip()

    return documents


# TF-IDF Weighted Term Document Incident Matrix
def compute_tfidf_matrix(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc['title'] + ' ' + doc['content'] for doc in documents])
    return tfidf_matrix


# Write Inverted Index to file
def write_inverted_index(inverted_index, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for term, doc_ids in inverted_index.items():
            file.write(f"{term}: {', '.join(map(str, doc_ids))}\n")


# Build Inverted Index
def build_inverted_index(documents):
    inverted_index = {}
    for doc in documents:
        terms = set(preprocess_text(doc['title'] + ' ' + doc['content']).split())
        for term in terms:
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].append(doc['number'])
    return inverted_index


# Main function for pre-processing
def preprocess_main(input_folder, output_tfidf_file, output_index_file):
    # Read documents
    documents = read_documents(input_folder)

    # Preprocess and compute TF-IDF matrix
    tfidf_matrix = compute_tfidf_matrix(documents)

    # Save TF-IDF matrix to file
    with open(output_tfidf_file, 'w', encoding='utf-8') as file:
        for row in tfidf_matrix.toarray():
            file.write(','.join(map(str, row)) + '\n')

    # Build and save Inverted Index
    inverted_index = build_inverted_index(documents)
    write_inverted_index(inverted_index, output_index_file)


# Run pre-processing
if __name__ == "__main__":
    input_folder = "/Users/rohinmehra/Desktop/Assignment1/files/"
    output_tfidf_file = "tfidf_matrix.txt"
    output_index_file = "inverted_index.txt"
    preprocess_main(input_folder, output_tfidf_file, output_index_file)
