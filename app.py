import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import streamlit as st

st.title('Human Emotion Detection')

from keras.optimizers import Adam

model = load_model('model.h5', compile=False)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


CLASS_NAME = ['angry','happy','sad']
uploaded_file = st.file_uploader('Upload File', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    img = image.load_img(uploaded_file, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  

    predictions = model.predict(x)
    predicted_class = np.argmax(predictions, axis=1)

    st.write(f"Predictions: {predictions[0]}")
    st.write(f"Predicted class -: {CLASS_NAME[predicted_class[0]]}")
