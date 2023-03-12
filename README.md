# Face Mask Detection Web App

This is a web application that uses machine learning to detect whether people in an image are wearing face masks. The app is built using Streamlit.

<a href="https://face-mask-detection-gyn1.onrender.com/"><img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" /></a>

## Features

- Allows users to upload an image and receive a prediction of whether people in the image are wearing face masks
- Employs transfer learning and uses a pretrained model named MobilenetV2 which is retrained with Simulated Face Mask Dataset to make predictions.
- Provides feedback to the user with a message indicating whether face masks were detected or not

## How to Use

2. Click the "Choose File" button and select an image file that you would like to analyze.
3. Click the "Upload" button to submit the image.
4. Wait for the prediction to be generated (this should only take a few seconds).
5. View the prediction result, which will indicate whether face masks were detected or not.

## Installation

To run the web app on your local machine, follow these steps:

1. Clone this repository to your local machine.
2. Create a virtual environment `python3.10.9 -m venv .venv`
3. Install the required Python packages by running `pip install -r requirements.txt` in the project directory.
4. Start the web app by running `streamlit run webapp.py` in the project directory.
5. Visit [localhost:8501](http://localhost:8501) in your web browser to access the app.
