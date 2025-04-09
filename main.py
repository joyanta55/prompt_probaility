import spacy
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
class KeywordSimilarity:
    def __init__(self, predefined_keywords, threshold=0.5):
        # Load the spaCy medium-sized model (which includes word vectors)
        self.nlp = spacy.load("en_core_web_md")
        self.predefined_keywords = predefined_keywords
        self.threshold = threshold
        
        # Convert predefined keywords to vectors
        self.keyword_vectors = [self.nlp(keyword).vector for keyword in predefined_keywords]

    def get_similarity(self, input_text):
        # Check if the input contains "docker image" or a programming language
        if not self.is_valid_prompt(input_text):
            return "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc."

        # Convert input text to vector
        input_vector = self.nlp(input_text).vector

        # Compute cosine similarity between input vector and predefined keyword vectors
        similarities = cosine_similarity([input_vector], self.keyword_vectors)
        
        # Get the indices of the keywords sorted by similarity score in descending order
        sorted_similarities_idx = similarities.argsort()[0][::-1]
        
        # Get the top keywords and their similarity scores
        ranked_keywords = [(self.predefined_keywords[idx], similarities[0][idx]) for idx in sorted_similarities_idx]
        
        # Ensure that the most relevant keywords (like 'python' and 'cpp') have a minimum similarity score
        if not self.meets_confidence_level(ranked_keywords):
            return "Not a valid prompt. Insufficient similarity score for required keywords like 'python' or 'cpp'."
        
        return ranked_keywords

    def is_valid_prompt(self, input_text):
        # Check if input text contains "docker image" or a programming language (e.g., python, cpp)
        docker_image_check = "docker image" in input_text.lower()
        language_check = bool(re.search(r'\b(python|cpp|java|go|flask|tensorflow|dockerfile)\b', input_text, re.IGNORECASE))
        
        return docker_image_check or language_check

    def meets_confidence_level(self, ranked_keywords):
        # Check if the similarity score for 'python' or 'cpp' meets the required threshold
        for keyword, score in ranked_keywords:
            if keyword.lower() in ['python', 'cpp'] and score >= self.threshold:
                return True
        return False

def main():
    # Define predefined keywords (you can expand this list as needed)
    data = None
    with open('config.json', 'r') as file:
        data = json.load(file)
    # print(data['ml_tools'])
    # print(len(data))
    predefined_keywords = data['ml_tools']+data['Language']+data['container']

    # Create an instance of KeywordSimilarity with a threshold of 0.5 for Python/CPP confidence level
    keyword_similarity = KeywordSimilarity(predefined_keywords, threshold=0.0)

    # Get input from the user
    input_text = input("Enter the text for keyword extraction: ")

    # Get the most similar keywords
    similar_keywords = keyword_similarity.get_similarity(input_text)

    # Display the result
    category_map = [0] * len(data)
    if isinstance(similar_keywords, str):
        print(similar_keywords)  # If it's an error message, print it
    else:
        print("\nMost Similar Keywords to Input Text (Ranked by Confidence):")
        for keyword, score in similar_keywords:
            if keyword in data['container'] :
                category_map[0] = category_map[0]+1
            if keyword in data['Language'] :
                category_map[1] = category_map[0]+1

            print(f"Keyword: {keyword}, Similarity Score: {score}")

if __name__ == "__main__":
    main()
