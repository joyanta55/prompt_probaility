# prompt_probaility
A way to figure out the similarity or any given prompt, based on different keywords.

## Steps
Create a python 3.12 virtual environment, and activate it.
```
python3.12 -m venv myenv promptprobability

source promptprobability/bin/activate
```


Install the pre-requisite
```
pip install -r requirements.txt
```

Run the code 

```
python bayesian.py
```
### Docker Steps
If you don't want the complexity of creating venv, installing dependencies, Follow the Docker approach. If you have `docker` in your machine. First run

```
docker build -t propmtprobability .
```
Follwed by 
```
docker run -it propmtprobability
```
### What it does
This code evaluates the relevance of predefined keyword lists (e.g., `cpp, python`) in a given prompt. It uses the `en_core_web_md` word embedding model from `spaCy` to compute similarity between the input and the keyword list. The model outputs `posterior probabilitie`s (derived from Bayesian inference) for each keyword, where the probability reflects the `maximum likelihood` of a match between the input and the predefined categories (e.g., `cpp, python`).


## Example prompt with output
Current design works on very few set of keywords (i.e. "docker", "cpp", "python"), if your prompt input contains any other words, please change in the `config.json` file.
let's assume this is the input from user on the console when you run the code.
```
create a cpp docker image
```
Here's the output you would get with current setup.

| **Category**   | **Combined Probability** |
|----------------|--------------------------|
| **ML Tools**   | 0.2979                   |
| **C++**        | 0.4829                   |
| **Python**     | 0.4671                   |
| **Container**  | 0.3278                   |

### Observations:
You might question why **Python** has a near-equal probability to **C++** even though the prompt doesn't mention `python`. This outcome arises from the nature of the **`en_core_web_md`** model in **spaCy** [https://spacy.io/models/en#en_core_web_md]. In general, when discussing **Docker**-based solutions, there is a strong association with **Python** due to its widespread use in **web frameworks** and **machine learning libraries**.
The model reflects this contextual connection, which results in a higher probability for **Python** despite the lack of explicit mentions. This showcases how the model uses **word embeddings** and **Bayesian inference** to capture underlying relationships between words and concepts in the text, even when they aren't directly stated.

Lets work with another prompt
```
create python cpp hello world docker image
```
Here's the result.

| **Category**   | **Combined Probability** |
|----------------|--------------------------|
| **ML Tools**   | 0.3059                  |
| **C++**        | 0.4941                  |
| **Python**     | 0.5406                  |
| **Container**  | 0.3189                  |

Now, based on the input prompt, you can probably guess what the output will be. **Python** should definitely receive a higher probability compared to **C++**, for the same reasons discussed.

Please check the `config.json` file to check the pre classified categories (i.e. *ML Tools*,*C++*, *Python* , *Container*), also the assciated `weight`, while making probabilistic decisions. Make changes to the `weight` based on your use case, also you can enrich each categories. Checkout the `test` directory for more samples.  



