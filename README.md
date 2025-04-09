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

