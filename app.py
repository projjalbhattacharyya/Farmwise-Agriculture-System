# STREAMLIT APPLICATION
import streamlit as st
import numpy as np
import joblib
from PIL import Image
import cv2
import os
from skimage.feature import hog  # For HOG feature extraction
import altair as alt
import pandas as pd

# Load pre-trained models
MODEL_DIR = "models/"
crop_model_path = os.path.join(MODEL_DIR, "crop_recommend.pkl")
crop_recommend_minmax = joblib.load('models/crop_recommend_minmax.pkl')
crop_recommend_scaler = joblib.load('models/crop_recommend_standscaler.pkl')

yield_model_path = os.path.join(MODEL_DIR, "crop_yield_dtr.pkl")
yield_preprocessor_path = os.path.join(MODEL_DIR, "crop_yield_preprocessor.pkl")

potato_model_path = os.path.join(MODEL_DIR, "potato_leaf_modelSVM.pkl")

# Initialize models
crop_model, yield_model, potato_model, yield_preprocessor = None, None, None, None

# Inject Custom CSS for Full-Width Fixed Sidebar Header
st.sidebar.markdown(
    """
    <style>
        /* Push sidebar content down to prevent overlap */
        section[data-testid="stSidebar"] .css-1d391kg {
            margin-top: 20px !important; /* Adjust content to not overlap with the header */
        }

        /* Full-width fixed sidebar header */
        .sidebar-header {
            position: static;
            top: 0px;
            left: 0px;
            width: 100%;
            height: 70px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #262730;  /* Matching Streamlit theme */
            border: 2px solid #FF4B4B;
            color: white;
            font-size: 27px;
            font-weight: bold;
            font-family: Arial, sans-serif;
            z-index: 1000;
            margin-bottom:20px;
            border-radius:10px;
        }
        .large-letter {
            font-size: 35px;
            font-weight: bold;
            color: #f39c12;
        }
    </style>
    
    <div class="sidebar-header">
        <span class="large-letter">F</span>IELDWISE
    </div>
    """,
    unsafe_allow_html=True
)

task = st.sidebar.radio("🔍 Choose a Task:", ["Crop Recommendation", "Crop Yield Prediction", "Potato Disease Classification"])

# Load models with error handling
try:
    crop_model = joblib.load(crop_model_path)
    st.sidebar.success("✅ Crop Model Loaded")
except:
    st.sidebar.warning("⚠️ Crop Model Not Available")

try:
    yield_model = joblib.load(yield_model_path)
    yield_preprocessor = joblib.load(yield_preprocessor_path)
    st.sidebar.success("✅ Yield Model Loaded")
except:
    st.sidebar.warning("⚠️ Yield Model Not Available")

try:
    potato_model = joblib.load(potato_model_path)
    st.sidebar.success("✅ Potato Disease Model Loaded")
except:
    st.sidebar.warning("⚠️ Potato Disease Model Not Available")


# Crop Recommendation Function
def crop_recommendation():
    st.subheader("🌱 Crop Recommendation System")
    st.write("Enter soil and climate parameters to get the best crop recommendation.")

    # Crop dictionary mapping prediction output to crop names
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
        7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
        12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
        21: "Chickpea", 22: "Coffee"
    }

    # User inputs
    N = st.number_input("Nitrogen (N)", value=0.0, step=0.1)
    P = st.number_input("Phosphorus (P)", value=0.0, step=0.1)
    K = st.number_input("Potassium (K)", value=0.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", value=0.0, step=0.1)
    humidity = st.number_input("Humidity (%)", value=0.0, step=0.1)
    ph = st.number_input("pH Level", value=0.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", value=0.0, step=0.1)

    if st.button("🌾 Recommend Crop"):
        if crop_model:
            try:
                # Prepare the input features
                features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

                # Apply MinMax Scaling and Standard Scaling
                features_minmax = crop_recommend_minmax.transform(features)
                features_scaled = crop_recommend_scaler.transform(features_minmax)

                # Make the prediction
                prediction = crop_model.predict(features_scaled).reshape(1, -1)

                # Map prediction to crop name
                predicted_crop = crop_dict.get(prediction[0][0], "Unknown Crop")

                st.success(f"✅ Recommended Crop: **{predicted_crop}**")

                # Display the visualization
                crop_recommendation_chart(predicted_crop, crop_dict)
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
        else:
            st.error("❌ Crop model not available!")

# Crop Recommendation - Show distribution of recommended crops
def crop_recommendation_chart(predicted_crop, crop_dict):
    crops = list(crop_dict.values())  # Ensure we use all crop names
    counts = [1 if crop == predicted_crop else 0.1 for crop in crops]  # Avoid zero-height bars

    df = pd.DataFrame({"Crop": crops, "Recommendation": counts})
    df["Crop"] = df["Crop"].astype("category")  # Explicitly set as category

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Crop", sort="-y", title="Crops"),
            y=alt.Y("Recommendation", title="Prediction Score"),
            color="Crop"
        )
        .properties(title="Crop Recommendation Distribution")
    )

    st.altair_chart(chart, use_container_width=True)

# Crop Yield Prediction Function
def crop_yield_prediction():
    st.subheader("🌾 Crop Yield Prediction")
    st.write("Predict the estimated crop yield based on past data.")

    # Numeric Inputs
    avg_rainfall = st.number_input("Average Rainfall (mm/year)", value=0.0, step=100.0)
    pesticides = st.number_input("Pesticides Used (tonnes)", value=0.0, step=0.1)
    avg_temp = st.number_input("Average Temperature (°C)", value=0.0, step=0.1)
    
    # Categorical Inputs
    country = st.selectbox("Select Country", [
        "Albania", "Algeria", "Angola", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
        "Bahamas", "Bahrain", "Bangladesh", "Belarus", "Belgium", "Botswana", "Brazil", "Bulgaria",
        "Burkina Faso", "Burundi", "Cameroon", "Canada", "Central African Republic", "Chile", "Colombia",
        "Croatia", "Denmark", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Eritrea", "Estonia",
        "Finland", "France", "Germany", "Ghana", "Greece", "Guatemala", "Guinea", "Guyana", "Haiti", "Honduras",
        "Hungary", "India", "Indonesia", "Iraq", "Ireland", "Italy", "Jamaica", "Japan", "Kazakhstan", "Kenya",
        "Latvia", "Lebanon", "Lesotho", "Libya", "Lithuania", "Madagascar", "Malawi", "Malaysia", "Mali",
        "Mauritania", "Mauritius", "Mexico", "Montenegro", "Morocco", "Mozambique", "Namibia", "Nepal",
        "Netherlands", "New Zealand", "Nicaragua", "Niger", "Norway", "Pakistan", "Papua New Guinea",
        "Peru", "Poland", "Portugal", "Qatar", "Romania", "Rwanda", "Saudi Arabia", "Senegal", "Slovenia",
        "South Africa", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Tajikistan",
        "Thailand", "Tunisia", "Turkey", "Uganda", "Ukraine", "United Kingdom", "Uruguay", "Zambia", "Zimbabwe"
    ])

    crop = st.selectbox("Select Crop", [
        "Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Wheat", "Cassava",
        "Sweet potatoes", "Plantains and others", "Yams"
    ])

    if st.button("📈 Predict Yield"):
        if yield_model and yield_preprocessor:
            try:
                # Prepare input data
                input_data = np.array([[avg_rainfall, pesticides, avg_temp, country, crop]])

                # Apply preprocessing
                transformed_data = yield_preprocessor.transform(input_data)

                # Make prediction
                predicted_yield = yield_model.predict(transformed_data)[0]

                st.success(f"📊 Predicted Yield for **{crop}** in **{country}**: **{predicted_yield:.2f} tons/hectare**")
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
        else:
            st.error("❌ Yield model or preprocessor not available!")


# Function for HOG Feature Extraction
def extract_hog_features(image, size=(128, 128)):
    """
    Extract HOG features from an image.

    Parameters:
        image (numpy array): Input image.
        size (tuple): Target image size before feature extraction.

    Returns:
        numpy array: Reshaped HOG feature vector.
    """
    # Convert PIL image to NumPy array (if needed)
    if isinstance(image, np.ndarray):
        img_array = image
    else:
        img_array = np.array(image)

    # Resize the image to ensure uniformity
    img_resized = cv2.resize(img_array, size)

    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Extract HOG features
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    return features.reshape(1, -1)  # Reshape for model input

# Potato Disease Classification Function
def potato_disease_classification():
    st.subheader("🥔 Potato Disease Classification")
    st.write("Upload an image of a potato leaf to classify the disease.")

    uploaded_file = st.file_uploader("📤 Upload Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict Disease"):
            if "potato_model" in globals():
                try:
                    # Convert image to NumPy array
                    img = np.array(image)

                    # Convert RGB to BGR (if needed)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    # Resize image before feature extraction
                    img = cv2.resize(img, (128, 128))

                    # Extract HOG features
                    hog_features = extract_hog_features(img)

                    # Make prediction
                    prediction = potato_model.predict(hog_features)

                    # Define labels
                    labels = ["Early Blight", "Healthy", "Late Blight"]

                    # Show prediction
                    st.success(f"🩺 **Predicted Disease:** {labels[prediction[0]]}")

                    # Show probability chart (if the model supports probabilities)
                    if hasattr(potato_model, "predict_proba"):
                        probabilities = potato_model.predict_proba(hog_features)[0]
                        potato_disease_chart(probabilities)

                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
            else:
                st.error("❌ Potato disease model not available!")

# Potato Disease Classification - Show class probabilities
def potato_disease_chart(predictions):
    labels = ["Early Blight", "Healthy", "Late Blight"]

    if isinstance(predictions, np.ndarray) and predictions.ndim == 1:
        probabilities = predictions.tolist()
    else:
        probabilities = [1/3] * len(labels)  # Assign equal probabilities as fallback

    df = pd.DataFrame({"Disease": labels, "Probability": probabilities})
    df["Disease"] = df["Disease"].astype("category")

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Disease", title="Potato Disease"),
            y=alt.Y("Probability", title="Prediction Probability"),
            color="Disease"
        )
        .properties(title="Potato Disease Classification Probabilities")
    )

    st.altair_chart(chart, use_container_width=True)


# Sidebar Navigation
st.title("🌿 Smart Agriculture System")
st.write("🚀 Predict crop recommendations, yield, and potato diseases using AI.")

if task == "Crop Recommendation":
    crop_recommendation()
elif task == "Crop Yield Prediction":
    crop_yield_prediction()
elif task == "Potato Disease Classification":
    potato_disease_classification()
