import cv2

class FaceRecognition:
    def __init__(self):
        # Load pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_mask(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            mask_region = frame[y:y+h, x:x+w]
            return mask_region

        return None

def recognize_faces():
    return FaceRecognition()
