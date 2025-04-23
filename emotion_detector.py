import cv2
from fer import FER

# Initialize webcam and emotion detector
cap = cv2.VideoCapture(0)
detector = FER(mtcnn=True)  # uses MTCNN for better face detection

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions in the frame
    result = detector.detect_emotions(frame)

    if result:
        # Get dominant emotion of the first face
        emotions = result[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)

        # Display it on screen
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("FER Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
