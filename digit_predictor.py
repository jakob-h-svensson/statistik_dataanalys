import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import joblib
import cv2
import numpy as np
from scipy.ndimage import center_of_mass


# -------------------------------------------------------------


@st.cache_resource
def load_models():
    """Importing the model and scaler"""
    model = joblib.load("extra_trees.joblib")
    scaler = joblib.load("digit_scaler.joblib")
    return model, scaler

model, scaler = load_models()


@st.cache_resource
def preprocess_and_predict(img_input, source):
    """Preprocessing image from canvas or uploaded, returning predictions and probabilities for each class"""
    if source == "canvas":
        img = img_input[:, :, 0].astype(np.uint8)

        # THRESH_BINARY converts the image to black (0) and white (255) based on a threshold
        thresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

        # Finding the contours around the digit. Returning the actual contours and the relationship between the contours (ignored with _)
        # RETR_EXTERNAL is used to only get the outer contours and CHAIN_APPROX_SIMPLE to only get the important pixels
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Finding the "largest" contour area in the list of contours (ignoring small contours - like dots etc).
        chosen_contour = max(contours, key=cv2.contourArea)

        # Creating a box around the largest contour - x,y is upper left corner, w,h is width and height
        x, y, w, h = cv2.boundingRect(chosen_contour)

        # Saving the image with only the box
        img = img[y:y + h, x:x + w]

        # Finding height and width for the img. IF img is higher than wide the height is set to 20 px, and the width is calculated to keep proportions
        # ELSE is the other way around
        rows, cols = img.shape
        if rows > cols:
            factor = 20.0 / rows
            rows_new = 20
            cols_new = max(1, int(round(cols * factor)))
        else:
            factor = 20.0 / cols
            cols_new = 20
            rows_new = max(1, int(round(rows * factor)))

        # Applying the height and widht on our img, INTER_AREA will make the downscaling softer
        img = cv2.resize(img, (cols_new, rows_new), interpolation=cv2.INTER_AREA)

        # Calculation correct padding and centering our image in a 28x28px empty grayscale image
        padded_img = np.zeros((28, 28), dtype=np.uint8)
        pad_y = (28 - rows_new) // 2
        pad_x = (28 - cols_new) // 2
        padded_img[pad_y:pad_y + rows_new, pad_x:pad_x + cols_new] = img

        # Finding x and y center of mass (where the center of the actual digit is) to center the padded image
        cy, cx = center_of_mass(padded_img)

        # The center is 14 in both x and y, this is calculating how many "steps" from the center cy and cx is
        move_x = int(np.round(14 - cx))
        move_y = int(np.round(14 - cy))

        # M is collected for warpAffine. [1,0,move_x] means keep x unchanged (1), dont let y affect x (0)
        # and just move x in the direction collected in move_x. The same principle is
        # applied to y in [0, 1, move_y]. warpAffine is then applying the matrix on the padded image.
        M = np.float32([[1, 0, move_x], [0, 1, move_y]])
        padded_img = cv2.warpAffine(padded_img, M, (28, 28))

        # The 2x2 kernel is used in the erode-method. erode is moving the kernel over the whole image one time,
        # and shrinking the lighter areas, making edges of the digits thinner
        kernel = np.ones((2, 2), np.uint8)
        padded_img = cv2.erode(padded_img, kernel, iterations=1)

        # Flattening and reshaping the image to a numpy array with 784 columns (for the model)
        flat_img = padded_img.flatten().astype(np.float64)
        img_2d = flat_img.reshape(1, -1)

        # Scaling with the loaded scaler
        scaled_img = scaler.transform(img_2d)

        prediction = model.predict(scaled_img)
        proba = model.predict_proba(scaled_img)

        return prediction, proba


    if source == "uploaded":
        file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8)

        # Loading the image in grayscale
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        # Inverting the colors (dark digit on white paper)
        img = cv2.bitwise_not(img)

        # Using THRESH_BINARY to tell the method: IF over threshold value THEN make it super white (255), else BLACK (0)
        # Using THRESH_OTSU to ignore threshold argument (128) and calculates the optimal threshold
        thresh, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Finding the contours around the digit. Returning the actual contours and the relationship between the contours (ignored with _)
        # RETR_EXTERNAL is used to only get the outer contours and CHAIN_APPROX_SIMPLE to only get the imortant pixels
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Finding the "largest" contour area in the list of contours (ignoring small contours - like dots etc).
        chosen_contour = max(contours, key=cv2.contourArea)

        # Creating a box around the largest contour - x,y is upper left corner, w,h is width and height
        x, y, w, h = cv2.boundingRect(chosen_contour)

        # Saving the image with only the box
        img = img[y:y+h, x:x+w]

        # Finding height and width for the img. IF img is higher than wide the height is set to 20 px, and the width is calculated to keep proportions
        # ELSE is the other way around
        rows, cols = img.shape
        if rows > cols:
            factor = 20.0 / rows
            rows_new = 20
            cols_new = max(1, int(round(cols * factor)))
        else:
            factor = 20.0 / cols
            cols_new = 20
            rows_new = max(1, int(round(rows * factor)))

        # Applying the height and widht on our img, INTER_AREA will make the downscaling softer
        img = cv2.resize(img, (cols_new, rows_new), interpolation=cv2.INTER_AREA)

        # Calculation correct padding and centering our image in a 28x28px empty grayscale image
        pad_y = (28 - rows_new) // 2
        pad_x = (28 - cols_new) // 2
        padded_img = np.zeros((28, 28), dtype=np.uint8)
        padded_img[pad_y:pad_y+rows_new, pad_x:pad_x+cols_new] = img

        # Flattening and reshaping the image to a numpy array with 784 columns (for the model)
        flat_img = padded_img.flatten().astype(np.float64)
        img_2d = flat_img.reshape(1, -1)

        # Scaling with scaler from training data
        scaled_img = scaler.transform(img_2d)

        # Prediction
        prediction = model.predict(scaled_img)
        proba = model.predict_proba(scaled_img)

        return prediction, proba

# -------------------------------------------------------------

source = ""

st.header('Digit Predictor')
st.subheader('Write a digit (0-9) in the canvas below, or upload a picture of a digit')

col1, col2 = st.columns(2)

with col1:
    tab1, tab2 = st.tabs(["Write", "Upload"])

    with tab1:
        st.write("Write a digit in the canvas.")
        canvas_result = st_canvas(
            fill_color="black",
            background_color="black",
            stroke_width=20,
            stroke_color="white",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        if st.button("Predict"):
            source = "canvas"
            if canvas_result.image_data is not None:
                if np.sum(canvas_result.image_data) == 0:
                    st.warning("Vänligen rita en siffra först!")
                else:
                    source = "canvas"

    with tab2:
        uploaded_file = ""
        if uploaded_file == "":
            st.write("Upload a picture of a digit.")
            uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            st.image(uploaded_file)
            if st.button("Predict uploaded"):
                source = "uploaded"

with col2:
    st.write("")
    st.write("")

    if source == "uploaded":
        prediction, confidence = preprocess_and_predict(uploaded_file, "uploaded")
        probs = confidence[0]
        st.subheader(f"Predicted digit: {prediction[0]}")
        df = pd.DataFrame({
            "Digit": list(range(10)),
            "Probability": probs
        })

        st.bar_chart(df, x="Digit", y="Probability")

    if source == "canvas":
        prediction, confidence = preprocess_and_predict(canvas_result.image_data, "canvas")
        probs = confidence[0]
        st.subheader(f"Predicted digit: {prediction[0]}")

        df = pd.DataFrame({
            "Digit": list(range(10)),
            "Probability": probs
        })

        st.bar_chart(df, x="Digit", y="Probability")




