from models.face_recognition import recognize_faces
from models.inpaint_model import InpaintModel
from training.gan_train import train_gan
import cv2

def main():
    # Train the GAN first if needed
    train_gan()

    # Initialize face recognition and inpainting model
    face_recognizer = recognize_faces()
    inpaint_model = InpaintModel()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face and mask region
        mask_region = face_recognizer.detect_mask(frame)

        # Inpaint the masked area
        if mask_region:
            inpainted_face = inpaint_model.inpaint(mask_region)

        # Display the result
        cv2.imshow("Inpainted Face", inpainted_face)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
