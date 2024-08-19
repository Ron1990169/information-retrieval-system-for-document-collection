import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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


# Read queries from a file
def read_queries(file_path):
    queries = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        current_query = {'number': None, 'content': ''}
        for line in lines:
            if line.strip().isdigit():
                if current_query['number'] is not None:
                    queries.append(current_query)
                current_query = {'number': int(line.strip()), 'content': ''}
            else:
                current_query['content'] += line.strip()
        if current_query['number'] is not None:
            queries.append(current_query)
    return queries


# TF-IDF Vectorize Queries
def vectorize_queries(queries):
    vectorizer = TfidfVectorizer()
    query_matrix = vectorizer.fit_transform([query['content'] for query in queries])
    return query_matrix


# Write Query Vector to file
def write_query_vector(query_matrix, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for row in query_matrix.toarray():
            file.write(','.join(map(str, row)) + '\n')


# Main function for query processing
def query_main(query_file, tfidf_matrix_file, inverted_index_file, output_file):
    # Read queries
    queries = read_queries(query_file)

    # Vectorize queries
    query_matrix = vectorize_queries(queries)

    # Write query vectors to file
    write_query_vector(query_matrix, output_file)


# Run query processing
if __name__ == "__main__":
    query_file = "/Users/rohinmehra/Desktop/Assignment1/files/IR1_Queries.txt"
    tfidf_matrix_file = "tfidf_matrix.txt"
    inverted_index_file = "inverted_index.txt"
    output_file = "query_vector.txt"
    query_main(query_file, tfidf_matrix_file, inverted_index_file, output_file)
