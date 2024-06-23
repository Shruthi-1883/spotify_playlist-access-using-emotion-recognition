import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pygame
import requests
import io
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN

# Set your Spotify client ID and client secret
client_id = '870351b4cce94f7f971f62860ae5c077'
client_secret = '1f03baa6b12c4d62be1a83476a8cdb44'

# Set up the Spotipy client with your Spotify credentials
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Initialize Pygame mixer
pygame.mixer.init()

# Example: Map emotions to playlist URIs
emotion_playlists = {
    'Happy': 'https://open.spotify.com/playlist/37i9dQZF1DX0TyiNWW7uUQ?si=0d1b81090de64952',
    'Sad': 'https://open.spotify.com/playlist/6NH9fMwCF0H9Nu0ZgHG5CB?si=42c35fa359dc4b98',
    'Neutral': 'https://open.spotify.com/playlist/5QFICJp45xjZwnTVOpw0Aa?si=d3416e8c354247f4',
    'Angry': 'https://open.spotify.com/playlist/4AFJCW0ZlyM41NWCxc4ptV?si=6a03001c00de455a',
    'Surprise': 'https://open.spotify.com/playlist/1gwT9lJwIaUXDJteuWudAr?si=fee29dd7977d40f0',
}

# Function to play a track
def play_track(track_url):
    print(f"Playing: {track_url}")
    pygame.mixer.music.load(io.BytesIO(requests.get(track_url).content))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Load the trained emotion recognition model
model = load_model('emotion_recognition_model.h5')

# Define the emotions
emotions = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Create a VideoCapture object to capture video from the webcam
cap = cv2.VideoCapture(0)

# Create an MTCNN detector
detector = MTCNN()

while True:
    # Capture each frame
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    for face in faces:
        # Extract face coordinates
        x, y, w, h = face['box']

        # Extract the face region
        face_region = frame[y:y + h, x:x + w]

        # Preprocess the face region for the model
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        # Predict emotion using the model
        prediction = model.predict(reshaped)
        emotion_label = emotions[np.argmax(prediction)]

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the emotion prediction on the frame
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Retrieve playlist information based on detected emotion
        playlist_uri = emotion_playlists.get(emotion_label, None)

        if playlist_uri:
            playlist_id = playlist_uri.split('/')[-1].split('?')[0]
            playlist = sp.playlist(playlist_id)

            # Play each track in the playlist
            for track in playlist['tracks']['items']:
                track_info = track['track']
                track_name = track_info['name']
                track_artists = ', '.join([artist['name'] for artist in track_info['artists']])
                print(f"Track: {track_name} - Artists: {track_artists}")

                # Check if a preview URL is available
                if track_info['preview_url']:
                    print(emotion_label)
                    play_track(track_info['preview_url'])
                else:
                    print(f"No preview available for '{track_name}'")

            # Stop playing after 30 seconds (adjust as needed)
            pygame.mixer.music.fadeout(30000)
        else:
            print("No playlist found for the detected emotion.")

    # Display the frame
    cv2.imshow('Live Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

