## UnmaskX

UnmaskX is a project that implements a neural network system to unmask and inpaint faces from images and video streams. This project utilizes Generative Adversarial Networks (GANs) for the inpainting task, providing high-quality face restoration.

# Features
- **Face Mask Detection:** Detects faces in images and video streams.
- **Image Inpainting:** Fills in masked regions of detected faces using GANs.
- **Video Processing:** Processes real-time video feeds and generates inpainted outputs.
- **Performance Metrics:** Evaluates the quality of inpainted images using MSE and PSNR.



# Run the setup script to install dependencies:
bash scripts/setup.sh

# Usage
1) Setting Up the Environment
-> After setting up the environment using the setup.sh script, activate the virtual environment:
- conda activate unmaskx-env

# Training the Model
2) To train the GAN model, run:
- python training/experiment.py

- This will initiate the training process with specified configurations. You can modify the experiment parameters in experiment.py.

# Running Face Unmasking
3) To run the face unmasking on a video stream, execute:
- python models/video.py

- This will start the video capture, detect faces, and apply inpainting on masked regions.

# Creating Output Video
4) To create a video from the inpainted frames, run:
- python utils/create_video.py

- This will compile the inpainted frames into a video file.

# Requirements
5) The project dependencies are listed in requirements.txt. To install them, ensure you are in the virtual environment and run:
- pip install -r requirements.txt

# Evaluation Metrics
6) The project uses the following metrics to evaluate the quality of the generated inpainting:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- These metrics are implemented in utils/metrics.py.

