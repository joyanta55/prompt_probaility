import unittest
import json
import sys
import os
# Add the parent directory of the test directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bayesian import BayesianKeywordSimilarity

class TestBayesianKeywordSimilarity(unittest.TestCase):

    def setUp(self,cpp_weight = 1.0, python_weight=1.0, threshold=0.0, boost_factor=0.0):
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
                "cpp": cpp_weight,
                "python": python_weight,
                "ml_tools": 1.0,
                "container": 1.0
            }
        }

        # You can modify this in each test case to change configurations
        self.predefined_keywords = {
            'ml_tools': self.data['ml_tools'],
            'cpp': self.data['cpp'],
            'python': self.data['python'],
            'container': self.data['container']
        }
        
        self.boost_factor = boost_factor
        self.weights = self.data['weights']
        self.threshold = threshold

        # Initialize the BayesianKeywordSimilarity instance
        self.keyword_similarity = BayesianKeywordSimilarity(
            self.predefined_keywords, weights=self.weights, threshold=self.threshold, boost_factor=self.boost_factor
        )

    def test_valid_language_matches_cpp(self):
        # Modify the configuration for this specific test case if needed

        self.setUp()  # Reinitialize with modified configuration
        input_prompt = "create a cpp docker image"
        _, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        self.assertGreater(combined_probabilities['cpp'], combined_probabilities['python'], "Expected higher probability for 'cpp'")
    

    def test_valid_language_matches_python(self):
        # Modify the configuration for this specific test case if needed

        self.setUp()  # Reinitialize with modified configuration
        input_prompt = "create a python cpp based hello world docker image"
        _, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        self.assertGreater(combined_probabilities['python'], combined_probabilities['cpp'], "Expected higher probability for 'python'")


    def test_valid_language_matches_both(self):
        # If both cpp and python are used in prompt, keyword python would get higher probability due to more relevance with docker

        self.setUp()  # Reinitialize with modified configuration
        input_prompt = "create python cpp hello world docker image"
        _, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        self.assertGreater(combined_probabilities['python'], combined_probabilities['cpp'], "Expected higher probability for 'python'")
        

    # Set weight values to get cpp related keyword to get more weight, to override the base spacy vector.
    def test_valid_language_matches_both_no_mention_cpp(self):
        # If no mention of language and you want more priority/weight on cpp.

        self.setUp(cpp_weight=1.0, python_weight=0.5)  # Reinitialize with modified configuration
        input_prompt = "create docker image"
        _, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        self.assertGreater(combined_probabilities['cpp'], combined_probabilities['python'], "Expected higher probability for 'cpp'")

    def test_valid_language_matches_both_both_mention_cpp(self):
        # If both of cpp and python are put as prompt, but you want more priority on cpp

        self.setUp(cpp_weight=1.0, python_weight=0.5)  # Reinitialize with modified configuration
        input_prompt = "create a cpp python docker image"
        _, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        self.assertGreater(combined_probabilities['cpp'], combined_probabilities['python'], "Expected higher probability for 'cpp'")

    
     # Set threshold in consine similarity so that overall probability would be less.
    def test_boost_factor(self):
        # If you want to put such priority on the exact key match. boost_factor is the key.

        self.setUp(cpp_weight=1.0, python_weight=1.0)  # Cosine similarity threshold, if below 0.4 dont consider the similarity. Also if exact keyword match then increase the similarity by 0.2.
        input_prompt = "create cpp docker image that has hello world web socket implemented like tornado. expose the program to the outer world" #here tornado is a python framework
        _, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        self.assertGreater(combined_probabilities['python'], combined_probabilities['cpp'], "Expected higher probability for 'cpp'")

        self.setUp(cpp_weight=1.0, python_weight=1.0, boost_factor=0.3)  # boost factor, since "cpp" is a category in out config.json, so there would be exact match and tornado will be less prioritize.  
        input_prompt = "create cpp docker image that has hello world web socket implemented like tornado. expose the program to the outer world" #here tornado is a python framework, but cpp is a exact match
        _, combined_probabilities = self.keyword_similarity.get_similarity(input_prompt)
        self.assertGreater(combined_probabilities['cpp'], combined_probabilities['python'], "Expected higher probability for 'cpp'")

    

    def test_invalid_prompt_no_match(self):
        # Test cases for invalid prompts
        input_prompt = "Hello"
        category_posteriors, _ = self.keyword_similarity.get_similarity(input_prompt)
        
        # Check if we get a valid response and handle the error message
        if isinstance(category_posteriors, str):
            self.assertEqual(category_posteriors, "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc.")

        input_prompt = "this text has no relevant keywords"
        category_posteriors, _ = self.keyword_similarity.get_similarity(input_prompt)
        
        # Check if we get a valid response and handle the error message
        if isinstance(category_posteriors, str):
            self.assertEqual(category_posteriors, "Not a valid prompt. Please include 'docker image' or a programming language like 'python', 'cpp', etc.")


if __name__ == '__main__':
    unittest.main()
