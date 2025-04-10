import spacy
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import numpy as np


class BayesianKeywordSimilarity:
    def __init__(self, predefined_keywords, weights, threshold=0.5, boost_factor=0.02):
        # Load the spaCy medium-sized model (which includes word vectors)
        self.nlp = spacy.load("en_core_web_md")
        self.predefined_keywords = predefined_keywords
        self.weights = weights  # Weights for each category
        self.threshold = threshold
        self.boost_factor = boost_factor  # Configurable boost factor

        # Convert predefined keywords to vectors for all categories
        self.keyword_vectors = {
            category: [self.nlp(keyword).vector for keyword in keywords]
            for category, keywords in self.predefined_keywords.items()
        }

    def get_similarity(self, input_text):
        # Check if the input contains "docker image" or a programming language
        if not self.is_valid_prompt(input_text):
            # print("Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc.")
            return (
                "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc.",
                None,
            )

        # Convert input text to vector
        input_vector = self.nlp(input_text).vector

        # Initialize dictionaries for storing similarities and posterior probabilities
        category_similarities = {}
        category_posteriors = {}

        for category, keywords in self.predefined_keywords.items():
            keyword_vectors = self.keyword_vectors[category]
            similarities = cosine_similarity([input_vector], keyword_vectors)

            # Rescale cosine similarity to be in the range [0, 1]
            similarities = (
                similarities + 1
            ) / 2  # Transform similarity from [-1, 1] to [0, 1]

            # Sort by similarity score in descending order
            sorted_similarities_idx = similarities.argsort()[0][::-1]
            ranked_keywords = [
                (keywords[idx], similarities[0][idx]) for idx in sorted_similarities_idx
            ]

            # Boost similarity for exact matches (container and language terms)
            if category == "python" or category == "container":
                for idx, (keyword, score) in enumerate(ranked_keywords):
                    if keyword.lower() in input_text.lower():
                        ranked_keywords[idx] = (
                            keyword,
                            score + self.boost_factor,
                        )  # Boost score for exact match

            # Apply weight for the category based on the config file
            weight = self.weights.get(
                category, 1.0
            )  # Default to 1.0 if no specific weight is found
            weighted_ranked_keywords = [
                (keyword, score * weight) for keyword, score in ranked_keywords
            ]

            category_similarities[category] = weighted_ranked_keywords

            # Apply Bayesian Update for each keyword and sort by posterior probability
            posterior_probabilities = []
            for keyword, similarity_score in weighted_ranked_keywords:
                # Compute the likelihood (which is the similarity score)
                likelihood = similarity_score

                # Retrieve the prior probability (default to uniform distribution if not provided)
                prior = 1 / len(self.predefined_keywords)  # Default prior if not found

                # Bayesian Update: Posterior = (Likelihood * Prior)
                posterior = likelihood * prior

                # Store the keyword and its computed posterior probability
                posterior_probabilities.append((keyword, posterior))

            # Sort by the posterior probability in descending order
            sorted_posterior = sorted(
                posterior_probabilities, key=lambda x: x[1], reverse=True
            )
            category_posteriors[category] = sorted_posterior

        # Now calculate the OR probability (combined probability for each category)
        combined_probabilities = {}
        for category, ranked_keywords in category_posteriors.items():
            # Calculate the combined probability using the OR rule
            prob_not_occurring = 1
            for _, score in ranked_keywords:
                prob_not_occurring *= 1 - score
            combined_prob = 1 - prob_not_occurring  # OR of all probabilities

            combined_probabilities[category] = combined_prob

        # If there is no valid result, return None for both
        if not category_posteriors or not combined_probabilities:
            return "No relevant keywords found", None

        return category_posteriors, combined_probabilities

    def is_valid_prompt(self, input_text):
        # Check if input text contains "docker image" or a programming language (e.g., python, cpp)
        docker_image_check = "docker image" in input_text.lower()
        language_check = bool(
            re.search(
                r"\b(python|cpp|java|go|flask|tensorflow|dockerfile)\b",
                input_text,
                re.IGNORECASE,
            )
        )

        return docker_image_check or language_check

    def meets_confidence_level(self, ranked_keywords):
        # Check if the similarity score for 'python' or 'cpp' meets the required threshold
        for keyword, score in ranked_keywords:
            if keyword.lower() in ["python", "cpp"] and score >= self.threshold:
                return True
        return False


class BayesianKeywordSimilarityStat:
    def __init__(self, result=None):
        # Load the spaCy medium-sized model (which includes word vectors)
        self.result = result

    def display(self):

        category_posteriors, combined_probabilities = self.result

        if combined_probabilities is None:
            print(
                "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc."
            )
            return

        # Display the result by category
        print(
            "\nMost Similar Keywords to Input Text (Ranked by Posterior Probability):"
        )
        for category, sorted_keywords in category_posteriors.items():
            print(f"\nCategory: {category.capitalize()}")
            for keyword, score in sorted_keywords:
                print(f"  Keyword: {keyword}, Posterior Probability: {score}")

        print("\nCombined Probability of Category Occurrence (OR of All Keywords):")
        for category, combined_prob in combined_probabilities.items():
            print(
                f"  Category: {category.capitalize()}, Combined Probability: {combined_prob:.4f}"
            )

    def return_higher_probability(self, key1="", key2=""):
        _, combined_probabilities = self.result

        if combined_probabilities is None:
            print(
                "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc."
            )
            return

        key1_weight = 0

        key2_weight = 0

        for category, combined_prob in combined_probabilities.items():
            if category == key1:
                key1_weight = combined_prob
            if category == key2:
                key2_weight = combined_prob

        if key1_weight == 0.0 or key1_weight == 0:
            print(key1 + " Or " + key2 + " not in defined category in config.json")
            return None
        elif key1_weight >= key2_weight:
            return key1

        else:
            return key2


def load_config(file_path="config.json"):
    """
    Loads configuration from a JSON file. The config includes predefined keywords
    and their weights.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    predefined_keywords = {
        "ml_tools": data["ml_tools"]["positives"],
        "cpp": data["cpp"]["positives"],
        "python": data["python"]["positives"],
        "container": data["container"]["positives"],
    }

    predefined_keywords_negative = {
        "ml_tools": data["ml_tools"]["negatives"],
        "cpp": data["cpp"]["negatives"],
        "python": data["python"]["negatives"],
        "container": data["container"]["negatives"],
    }

    weights = data["weights"]  # Extracting weights for each category

    return predefined_keywords, weights, predefined_keywords_negative


def main():
    # Load predefined keywords and weights from config.json
    predefined_keywords, weights , _ = load_config("config.json")

    keyword_similarity = BayesianKeywordSimilarity(predefined_keywords, weights)

    print("Bayesian Keyword Similarity Analysis")
    print("Type 'exit' to quit the program.\n")

    while True:
        # Get input from the user
        input_text = input("Enter the text for keyword extraction: ")

        # Exit condition
        if input_text.lower() == "exit":
            print("Exiting program.")
            break

        # Get the most similar keywords using the Bayesian approach
        result = keyword_similarity.get_similarity(input_text)

        # Check if the result is an error message (a string)
        if isinstance(result, str):
            print(result)  # Print the error message
            continue  # Continue the loop to get the next input

        # If result is valid (a tuple containing category_posteriors and combined_probabilities)
        stat = BayesianKeywordSimilarityStat(result)
        stat.display()
        print(stat.return_higher_probability("cpp", "python"))


if __name__ == "__main__":
    main()
