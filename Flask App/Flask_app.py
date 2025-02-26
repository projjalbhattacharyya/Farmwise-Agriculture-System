# FLASK APPLICATION
# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import joblib
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# import cv2
# from skimage.feature import hog
# from skimage.color import rgb2gray
# #import tensorflow as tf
# #from tensorflow.keras.preprocessing import image as tf_image
# #from keras.preprocessing.image import load_img, img_to_array
# #from tensorflow.keras.applications.vgg16 import preprocess_input

# app = Flask(__name__)

# # Define the upload folder
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the models and preprocessors
# try:
#     crop_yield_preprocessor = joblib.load('models/crop_yield_preprocessor.pkl')
#     crop_yield_model = joblib.load('models/crop_yield_dtr.pkl')

#     #potato_disease_model = joblib.load('models/model_potatoedisease.pkl')
#     #class_names = ['Potato___Late_blight', 'Potato___Early_blight','Potato___healthy']
#     # Load potato disease classification model (SVM)
#     potato_disease_model = joblib.load('models/potato_leaf_modelSVM.pkl')
    
#     # Mapping of class labels (if the model returns numbers)
#     class_mapping = {0: 'Potato___Early_blight', 1: 'Potato___healthy', 2: 'Potato___Late_blight'}

#     crop_recommend_model = joblib.load('models/crop_recommend.pkl')
#     crop_recommend_minmax = joblib.load('models/crop_recommend_minmax.pkl')
#     crop_recommend_scaler = joblib.load('models/crop_recommend_standscaler.pkl')
    
# except Exception as e:
#     print(f"Error loading models: {e}")
#     exit(1)

# # Function to preprocess image and extract HOG features
# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (128, 128))

#     # Ensure image is grayscale
#     if len(image.shape) == 3:  # If RGB, convert to grayscale
#         gray_image = rgb2gray(image)
#     else:
#         gray_image = image  # Already grayscale

#     # Extract HOG features
#     features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
#     return np.array(features).reshape(1, -1)

# @app.route('/')
# def index():
#     return render_template('homepage.html')

# @app.route('/potato_disease', methods=['GET', 'POST'])
# def potato_disease():
#     if request.method == 'POST':
#         try:
#             if 'image' not in request.files:
#                 return jsonify({'error': 'No file uploaded'})

#             file = request.files['image']
#             if file.filename == '':
#                 return jsonify({'error': 'No selected file'})

#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_path)

#             # Preprocess image and extract features
#             features = preprocess_image(file_path)

#             # Ensure feature shape matches model input
#             if features.shape[1] != potato_disease_model.n_features_in_:
#                 return jsonify({'error': 'Feature size mismatch with trained model'})

#             # Make prediction
#             prediction = potato_disease_model.predict(features)[0]

#             # Map prediction to class name
#             if isinstance(prediction, (np.integer, int)):  # Check if the prediction is a number
#                 predicted_label = class_mapping[prediction]  # Correct way to get the label
#             else:
#                 predicted_label = prediction  # If model already returns labels

#             return render_template('potato_disease.html', result=f'Predicted Disease: {predicted_label}')
#         except Exception as e:
#             return render_template('potato_disease.html', result=f'Error: {str(e)}')

#     return render_template('potato_disease.html', result=None)

# @app.route('/crop_yield', methods=['GET', 'POST'])
# def crop_yield():
#     if request.method == 'POST':
#         try:
#             # Get data from the form
#             average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
#             pesticides_tonnes = float(request.form['pesticides_tonnes'])
#             avg_temp = float(request.form['avg_temp'])
#             Area = request.form['Area']
#             Item = request.form['Item']

#             # Convert categorical variables to numerical if needed
#             features = np.array([[average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])

#             # Preprocess the input
#             transformed_features = crop_yield_preprocessor.transform(features)

#             # Make the prediction
#             predicted_yield = crop_yield_model.predict(transformed_features)
            
#             return render_template('crop_yield.html', result=f'Predicted Yield: {predicted_yield[0]:.2f}')
#         except Exception as e:
#             return render_template('crop_yield.html', result=f'Error: {str(e)}')

#     return render_template('crop_yield.html', result=None)

# @app.route('/crop_recommend', methods=['GET', 'POST'])
# def crop_recommend():
#     if request.method == 'POST':
#         try:
#             # Get data from the form
#             N = float(request.form['N'])
#             P = float(request.form['P'])
#             K = float(request.form['K'])
#             temperature = float(request.form['temperature'])
#             humidity = float(request.form['humidity'])
#             ph = float(request.form['ph'])
#             rainfall = float(request.form['rainfall'])

#             # Prepare the input features
#             features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

#             # Preprocess the input
#             features_minmax = crop_recommend_minmax.transform(features)
#             features_scaled = crop_recommend_scaler.transform(features_minmax)

#             # Make the prediction
#             prediction = crop_recommend_model.predict(features_scaled).reshape(1, -1)

#             # Map the prediction to crop names
#             crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
#                          8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
#                          14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
#                          19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

#             predicted_crop = crop_dict.get(prediction[0][0], "Unknown crop")

#             return render_template('crop_recommend.html', result=f'{predicted_crop}')
#         except Exception as e:
#             return render_template('crop_recommend.html', result=f'Error: {str(e)}')

#     return render_template('crop_recommend.html', result=None)

# # @app.route('/potato_disease', methods=['GET', 'POST'])
# # def potato_disease():
# #     if request.method == 'POST':
# #         try:
# #             # Handle file upload
# #             if 'image' not in request.files:
# #                 return jsonify({'error': 'No file part'})

# #             file = request.files['image']
# #             if file.filename == '':
# #                 return jsonify({'error': 'No selected file'})

# #             if file:
# #                 # Save the file to the static folder
# #                 filename = file.filename
# #                 file_path = os.path.join(UPLOAD_FOLDER, filename)
# #                 file.save(file_path)

# #                 # Load and preprocess the image
# #                 img = tf_image.load_img(file_path, target_size=(256, 256))
# #                 img_array = img_to_array(img)/ 255.0
# #                 img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# #                 # Make the prediction
# #                 predictions = potato_disease_model.predict(img_array)
# #                 predicted_class = class_names[np.argmax(predictions)]

# #                 # Get the confidence of the prediction
# #                 confidence = round(100 * np.max(predictions), 2)

# #                 return render_template('potato_disease.html', 
# #                                        result=f'Predicted Disease: {predicted_class} with confidence {confidence}%')
# #         except Exception as e:
# #             return render_template('potato_disease.html', result=f'Error: {str(e)}')

# #     return render_template('potato_disease.html', result=None)

# if __name__ == '__main__':
#     app.run(debug=True)