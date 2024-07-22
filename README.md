# Human Emotion Detection App

This project is a human emotion detection application built using a Convolutional Neural Network (CNN). The model is designed to classify images into three emotion categories.

## Model Architecture

The model is built using the Sequential API from TensorFlow's Keras library and consists of the following layers:

```python
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(2, 2),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## Installation

To get started with the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/alihassanml/human-emotion-detection-app.git
    cd human-emotion-detection-app
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset

Ensure you have a dataset of images categorized into the three emotion classes. Update the code to point to your dataset directory.

## Training the Model

To train the model, use the following command:
```sh
python train.py
```

This script will load the dataset, preprocess the images, and train the model. The trained model will be saved to a file for later use.

## Evaluating the Model

After training, evaluate the model's performance using:
```sh
python evaluate.py
```

This will print out the accuracy and loss of the model on the test dataset.

## Using the Model

You can use the trained model to make predictions on new images:
```sh
python predict.py path_to_image
```

This script will load the trained model and output the predicted emotion for the given image.

## Streamlit App

To run the Streamlit app for human emotion detection, use:
```sh
streamlit run app.py
```

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License.
```

This README file provides an overview of your project, instructions for installation, training, evaluation, and usage, as well as details on how to run the Streamlit app. Feel free to modify it to better fit your specific implementation and project details.
