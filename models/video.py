import cv2

class VideoCapture:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to capture frame from video source.")
        return frame

    def release(self):
        self.cap.release()

    def display_frame(self, window_name, frame):
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

def video_stream():
    return VideoCapture()
