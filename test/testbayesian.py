import unittest
import json
import sys
import os
# Add the parent directory of the test directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bayesian import BayesianKeywordSimilarity

class TestBayesianKeywordSimilarity(unittest.TestCase):

    def setUp(self):
        """
        Setup method for initializing test environment with dynamic configurations.
        """
        # Default configuration data (can be changed per test case)
        self.data = {
            "ml_tools": ["tensorflow", "keras", "scikit-learn"],
            "cpp": ["c plus plus", "cpp", "c++", "c"],
            "python": ["python", "python 3", "Flask", "python 3.12"],
            "container": ["docker", "image", "container"],
            "weights": {
                "cpp": 1.0,
                "python": 0.9,
                "ml_tools": 0.0,
                "container": 0.0
            }
        }

        # You can modify this in each test case to change configurations
        self.predefined_keywords = {
            'ml_tools': self.data['ml_tools'],
            'cpp': self.data['cpp'],
            'python': self.data['python'],
            'container': self.data['container']
        }
        
        self.boost_factor = 0.02
        self.weights = self.data['weights']

        # Initialize the BayesianKeywordSimilarity instance
        self.keyword_similarity = BayesianKeywordSimilarity(
            self.predefined_keywords, weights=self.weights, threshold=0.0, boost_factor=self.boost_factor
        )

    def test_valid_language_matches_cpp(self):
        # Modify the configuration for this specific test case if needed
        self.data['weights']['cpp'] = 1.5  # Boost cpp weight for this test case
        self.setUp()  # Reinitialize with modified configuration

        input_prompt = "create a cpp docker image"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        self.assertGreater(combined_probabilities['cpp'], combined_probabilities['python'], "Expected higher probability for 'cpp'")

    def test_valid_language_matches_python(self):
        # Modify the configuration for this specific test case if needed
        self.data['weights']['python'] = 1.5  # Boost python weight for this test case
        self.setUp()  # Reinitialize with modified configuration

        input_prompt = "create a python based hello world docker image"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        self.assertGreater(combined_probabilities['python'], combined_probabilities['cpp'], "Expected higher probability for 'python'")

    def test_valid_language_matches_both(self):
        # Modify the configuration for this specific test case if needed
        self.data['weights']['cpp'] = 1.0  # Default cpp weight for this test case
        self.data['weights']['python'] = 1.0  # Default python weight for this test case
        self.setUp()  # Reinitialize with modified configuration

        input_prompt = "create python cpp hello world docker image"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        self.assertGreater(combined_probabilities['python'], combined_probabilities['cpp'], "Expected higher probability for 'python'")

    

    def test_invalid_prompt_no_match(self):
        # Test cases for invalid prompts
        input_prompt = "Hello"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        
        # Check if we get a valid response and handle the error message
        if isinstance(category_posteriors, str):
            self.assertEqual(category_posteriors, "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc.")

        input_prompt = "this text has no relevant keywords"
        category_posteriors, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        
        # Check if we get a valid response and handle the error message
        if isinstance(category_posteriors, str):
            self.assertEqual(category_posteriors, "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc.")


if __name__ == '__main__':
    unittest.main()
