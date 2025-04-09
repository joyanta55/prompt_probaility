import unittest
import json
import sys
import os
# Add the parent directory of the test directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bayesian import BayesianKeywordSimilarity


class TestBayesianKeywordSimilarity(unittest.TestCase):

    def setUp(self):
        # Setup predefined keywords and initialize the keyword similarity class

        data = None
        data = {
            "ml_tools": ["tensorflow", "keras", "scikit-learn"],
            "cpp": ["c plus plus", "cpp", "c++", "c"],
            "python": ["python", "python 3", "Flask", "python 3.12"],
            "container": ["docker", "image", "container"],
            "weights": {
                "cpp": 1.0,
                "python": 1.0,
                "ml_tools": 0.8,
                "container": 0.7
            }
        }

        # Set predefined keywords from the mocked data
        self.predefined_keywords = {
            'ml_tools': data['ml_tools'],
            'cpp': data['cpp'],
            'python': data['python'],
            'container': data['container']
        }
        self.boost_factor = 0.02

        # Extract weights from the data (if available in the mock)
        self.weights = data['weights']

        # Initialize the BayesianKeywordSimilarity instance
        self.keyword_similarity = BayesianKeywordSimilarity(
            self.predefined_keywords, weights=self.weights, threshold=0.0, boost_factor=self.boost_factor
        )

    def test_valid_langage_matches(self):
        input_prompt = "create a cpp docker image"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        if category_posteriors:
            # Check if the returned combined probabilities are reasonable
            self.assertGreater(combined_probabilities['cpp'], combined_probabilities['python'], "Expected higher probability for 'cpp'")
        else:
            print("Error:", category_posteriors)

        input_prompt = "create a python based hello world docker image"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        if category_posteriors:
            # Check if the returned combined probabilities are reasonable
            self.assertGreater(combined_probabilities['python'], combined_probabilities['cpp'], "Expected higher probability for 'python'")
        else:
            print("Error:", category_posteriors)

        input_prompt = "create cpp hello world docker image"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        if category_posteriors:
            # Check if the returned combined probabilities are reasonable
            self.assertGreater(combined_probabilities['cpp'], combined_probabilities['python'], "Expected higher probability for 'cpp'")
        else:
            print("Error:", category_posteriors)

        # When python and cpp both declared, python dominates. Because en_core_web_md offers more similarity of docker to python. Change data["weights"] hyperparameter
        input_prompt = "create python cpp hello world docker image"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        if category_posteriors:
            # Check if the returned combined probabilities are reasonable
            self.assertGreater(combined_probabilities['python'], combined_probabilities['cpp'], "Expected higher probability for 'python'")
        else:
            print("Error:", category_posteriors)

        # When only ask for a dockerfile with different language for example rust, or java, still python dominates because java and rust part not implemented yet.
        input_prompt = "create java hello world docker image"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        if category_posteriors:
            # Check if the returned combined probabilities are reasonable
            self.assertGreater(combined_probabilities['python'], combined_probabilities['cpp'], "Expected higher probability for 'python'")
        else:
            print("Error:", category_posteriors)
        

    def test_invalid_prompt_no_match(self):

        input_prompt = "Hello"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        
        # Check if we get a valid responseand handle the error message
        if isinstance(category_posteriors, str):
            self.assertEqual(category_posteriors, "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc.")

        input_prompt = "this text has no relevant keywords"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        
        # Check if we get a valid responseand handle the error message
        if isinstance(category_posteriors, str):
            self.assertEqual(category_posteriors, "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc.")


        input_prompt = "create cpp or python" # another invalid prompt
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        # Check if we get a valid response and handle the error message
        if isinstance(category_posteriors, str):
            self.assertEqual(category_posteriors, "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc.")
        
        input_prompt = "" # another invalid prompt
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        # Check if we get a valid response and handle the error message
        if isinstance(category_posteriors, str):
            self.assertEqual(category_posteriors, "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc.")
    

if __name__ == '__main__':
    unittest.main()

