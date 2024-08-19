import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Function to read inverted index from file
def read_inverted_index(file_path):
    inverted_index = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            term, doc_ids = line.strip().split(': ')
            inverted_index[term] = list(map(int, doc_ids.split(', ')))
    return inverted_index


# Function to read query vector from file
def read_query_vector(file_path):
    query_vectors = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            vector = list(map(float, line.split(',')))
            query_vectors.append(vector)
    return query_vectors


# Function to display information for each query
def display_query_info(query_id, query_vector, inverted_index):
    print(f"\n{'=' * 50}")
    print(f"Query {query_id}:")

    # Display Query Vector
    print("\nQuery Vector:")
    print(query_vector)

    # Display Top terms and their weights in the query
    sorted_indices = np.argsort(query_vector)[::-1]
    print("\nTop terms and their TF-IDF weights in the query:")
    for index in sorted_indices[:5]:  # Display top 5 terms
        term = list(inverted_index.keys())[index]
        weight = query_vector[index]
        print(f"  {term}: {weight:.4f}")

    # Display Documents related to the query based on Inverted Index
    print("\nRelated Documents (based on Inverted Index):")
    for term, weight in zip(inverted_index.keys(), query_vector):
        if weight > 0:
            print(f"  {term}: {', '.join(map(str, inverted_index[term]))}")


# Main function
def main():
    base_path = "/Users/rohinmehra/Desktop/Assignment1"

    # Read files
    inverted_index = read_inverted_index(os.path.join(base_path, "inverted_index.txt"))
    query_vectors = read_query_vector(os.path.join(base_path, "query_vector.txt"))

    # Display information for each query
    for i, query_vector in enumerate(query_vectors):
        display_query_info(i + 1, query_vector, inverted_index)


if __name__ == "__main__":
    main()
