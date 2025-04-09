import json

# Load the data from the JSON file
with open('config.json', 'r') as file:
    data = json.load(file)

# Print the parsed data
print(data['ml_tools'])
