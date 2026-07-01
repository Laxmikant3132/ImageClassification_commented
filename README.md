# Cat vs Dog Image Classification

This project is a deep learning-based image classification system designed to distinguish between images of cats and dogs. It includes a Convolutional Neural Network (CNN) built with TensorFlow/Keras, a Jupyter Notebook detailing the data preparation and training process, and two web application implementations (FastAPI and Flask) to serve the trained model as an API.

## Project Structure

- **`ImageClassification_commented.ipynb`**: The main Jupyter Notebook containing the code for loading the dataset, preprocessing images, defining the CNN architecture, training the model, evaluating its performance, and visualizing feature maps.
- **`app_fastAPI.py`**: A FastAPI application that provides a RESTful API endpoint (`/predict`) for image classification. It accepts an image upload and returns the predicted class (dog or cat) along with the prediction probability. It also utilizes `pyngrok` to expose the local server to the public internet.
- **`app_flask.py`**: A Flask application providing similar functionality to the FastAPI app, serving the model predictions.
- **`cat_vs_dog_model (1).h5` & `best_model (1).h5`**: The trained Keras model files containing the weights and architecture of the CNN.
- **`*.png` (e.g., `confusion_matrix.png`, `training_history.png`, `conv2d_feature_maps.png`)**: Various visualizations generated during the model evaluation process, including training curves, confusion matrix, and visualizations of the convolutional layers' feature maps.

## Requirements

To run this project, you will need the following Python libraries installed:

- TensorFlow / Keras
- FastAPI
- Flask
- Uvicorn
- python-multipart
- numpy
- Pillow (PIL)
- pyngrok
- nest_asyncio
- matplotlib (for the notebook)
- scikit-learn (for the notebook)

You can install the dependencies using pip (you may need to create a `requirements.txt` file based on your exact environment):

```bash
pip install tensorflow fastapi flask uvicorn python-multipart numpy pillow pyngrok nest_asyncio
```

## How to Run the API (FastAPI)

1.  Ensure you have your ngrok auth token set in `app_fastAPI.py` if you intend to use ngrok for a public URL.
2.  Run the FastAPI application:

    ```bash
    python app_fastAPI.py
    ```

3.  The application will start a local server (typically at `http://localhost:8000`) and print a public ngrok URL to the console.
4.  You can send a POST request to the `/predict` endpoint (e.g., using Postman or `curl`) with a file attached under the key `file`.

    Example using `curl`:
    ```bash
    curl -X POST -F "file=@path/to/your/image.jpg" <YOUR_NGROK_URL>/predict
    ```

## Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of multiple `Conv2D` and `MaxPooling2D` layers for feature extraction, followed by a `Flatten` layer and `Dense` layers for classification. The final output layer uses a sigmoid activation function for binary classification (cat vs. dog).
