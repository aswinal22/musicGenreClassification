import requests
import os

# Define the URL of your Flask app
url = 'http://127.0.0.1:5000/predict'

# Provide the file path of the song to test
# Replace this with the actual path of a song file you'd like to test
song_file_path = os.path.join(os.getcwd(), "C:\\Users\\HP\\Downloads\\trap-beat-loop-ken-carson-drums_152bpm.wav")

# Prepare the data payload
data = {
    'file_path': song_file_path
}

# Send a POST request to the Flask app
response = requests.post(url, json=data)

# Print the response (either the predicted genre or the error message)
print('Response:', response.json())
