import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Create a Streamlit app
def main():
    st.title("Plant Disease Detection")
    st.write("Upload an image of a plant to detect diseases.")

    # Allow the user to upload an image file
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classify the image
        result = classify_image(image)
        st.write("Classification Results:")
        for res in result:
            st.write(f"{res[1]}: {res[2] * 100:.2f}%")

def classify_image(image):
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)

    # Make predictions using the pre-trained model
    preds = model.predict(processed_img)
    decoded_preds = decode_predictions(preds, top=3)[0]

    return decoded_preds

if __name__ == "__main__":
    main()
