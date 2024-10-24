from flask import Flask, request, jsonify
import pickle
import os

# Initialize the Flask app
app = Flask(__name__)


# Get the current working directory (where your Python script is running)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative paths to your .pkl files
model_path = os.path.join(current_dir, 'models', 'music_genre_classifier.pkl')  # Adjust the path as needed
scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')

# Load the scaler used for feature scaling
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the trained model from the .pkl file
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Assuming you have a feature extraction function that works with the song file
import librosa
import numpy as np

def extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, duration=30)  # Load 30 seconds of the audio
    
    # Extract features using librosa
    features = []
    
    # 1. Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    features.append(zcr)
    
    # 2. Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features.append(spectral_centroid)
    
    # 3. Spectral Bandwidth
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features.append(spectral_bandwidth)
    
    # 4. Spectral Contrast
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    features.append(spectral_contrast)
    
    # 5. Spectral Rolloff
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features.append(spectral_rolloff)
    
    # 6. RMSE (Root Mean Square Energy)
    rmse = np.mean(librosa.feature.rms(y=y))
    features.append(rmse)
    
    # 7. Chroma Feature (average over 12 bins)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    features.extend(chroma_stft.tolist())  # Ensure chroma_stft is flattened into a list
    
    # 8. MFCC (average over 13 coefficients)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    features.extend(mfcc.tolist())  # Ensure mfcc is flattened into a list
    
    # Ensure we have exactly 36 features
    if len(features) > 36:
        features = features[:36]  # Trim if there are more than 36
    elif len(features) < 36:
        # Pad with zeros to get 36 features
        features.extend([0] * (36 - len(features)))
    
    return features

# Flask route to handle genre prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data sent by the Express server
        data = request.json
        print(data)
        if 'filePath' in data:
            file_path = data['filePath']  # Access 'filePath' correctly
            print('File path:', file_path)  # Print the file path
        else:
            print('filePath not found in the request data')
        # Ensure the file path exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        # Extract features from the file
        features = extract_features(file_path)
        print(features)
        # Scale the features using the loaded scaler
        features_scaled = scaler.transform([features])

        # Make a prediction using the pre-trained model
        predicted_genre = model.predict(features_scaled)[0]

        # Return the predicted genre as a JSON response
        return jsonify({'predicted_genre': predicted_genre})

    except Exception as e:
        # Handle any exception and return an error message
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5005)  # Flask app runs on port 5000
