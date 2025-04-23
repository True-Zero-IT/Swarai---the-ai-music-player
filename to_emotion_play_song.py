import cv2
from fer import FER
import pygame
import time

# Emotion to song mapping
emotion_to_song = {
   'happy': 'D:/swarai/Music/happy_song.mp3',
    'sad': 'D:/swarai/Music/sad_song.mp3',
    'angry': 'D:/swarai/Music/angry_song.mp3',
    'neutral': 'D:/swarai/Music/chill_song.mp3'
}

# Initialize emotion detector
detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

print("Detecting emotion...")

emotion_detected = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect_emotions(frame)

    if result:
        emotion, score = detector.top_emotion(frame)
        emotion_detected = emotion
        print(f"Detected Emotion: {emotion}")
        break

# Stop the camera
cap.release()
cv2.destroyAllWindows()

# Play song using pygame
if emotion_detected and emotion_detected in emotion_to_song:
    song = emotion_to_song[emotion_detected]
    print(f"Playing: {song}")

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(song)
    pygame.mixer.music.play()

    # Wait until song finishes playing
    while pygame.mixer.music.get_busy():
        time.sleep(1)

    pygame.quit()
else:
    print("No matching emotion detected.")
