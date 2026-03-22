import cv2
import os

def create_video(output_path, image_folder, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, _ = frame.shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    cv2.destroyAllWindows()

def create_inpainted_video():
    create_video('output/inpainted_faces.mp4', './data/inpainted', fps=30)
