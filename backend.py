from flask import Flask, request
import tempfile
import rasterio
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pickle
from sklearn.preprocessing import StandardScaler
import firebase_admin
from firebase_admin import credentials, storage, firestore
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB

# Load the pre-trained random forest model
RF_PATH = 'NDVI_RF_V1'
FM_PATH = 'kmeans_model.pkl'
model_rf = tf.saved_model.load(RF_PATH)

# Load the pre-trained KMeans model
KM_PATH = 'kmeans_model.pkl'
with open(KM_PATH, 'rb') as model_file:
    kmeans_model = pickle.load(model_file)

# Initialize Firebase
BUCKET_URL = os.getenv('BUCKET_URL')
cred = credentials.Certificate('capstone-a4973-firebase-adminsdk-7w72c-1086b07d41.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': BUCKET_URL
})
db = firestore.client()
bucket = storage.bucket()

# Define constants
FEATURES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
            'EVI', 'NBR', 'NDMI', 'NDWI', 'NDBI', 'NDBaI', 'NDVI', 'elevation']
ALL_FEATURES = FEATURES
CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

@app.route('/process', methods=['POST'])
def process_tif():
    # Receive the file names from the request
    if 'tif_filename' not in request.json or 'csv_filename' not in request.json or 'city_name' not in request.json:
        return 'Parameters are missing', 400

    tif_filename = request.json.get('tif_filename')
    csv_filename = request.json.get('csv_filename')
    city_name = request.json.get('city_name')

    if not tif_filename or not csv_filename or not city_name:
        return 'Invalid parameters', 400

    # Define the paths in Firebase Storage
    tif_file_path = f"RawFiles/{tif_filename}"
    csv_file_path = f"RawFiles/{csv_filename}"

    # Download the TIF file from Firebase Storage
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    tif_blob = bucket.blob(tif_file_path)
    tif_blob.download_to_filename(temp_input.name)

    # Download the CSV file from Firebase Storage
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    csv_blob = bucket.blob(csv_file_path)
    csv_blob.download_to_filename(temp_csv.name)

    # Load image
    image = rasterio.open(temp_input.name)
    # Collect valid features
    used_image_feature = [i + 1 for i in range(len(ALL_FEATURES)) if image.descriptions[i] in FEATURES]
    image_data = []
    for i in used_image_feature:
        band_data = image.read(i).flatten()
        image_data.append(band_data)

    # Create DataFrame for image data
    image_df = pd.DataFrame(np.array(image_data).T, columns=FEATURES)
    # Replace invalid values
    image_df['elevation'] = image_df['elevation'].replace([np.inf, -np.inf, np.nan], 0).astype(np.int64)
    # Identify valid pixels
    valid_pixels = ~((image_df == 0).all(axis=1) | image_df.isnull().any(axis=1))
    valid_data = image_df[valid_pixels]
    
    # Calculate average NDVI
    average_ndvi = image_df['NDVI'].mean()
    
    # Load CSV data from the uploaded file
    csv_data = pd.read_csv(temp_csv.name)
    
    # Calculate average Carbon Stock
    csv_features = csv_data[['CA', 'CB', 'CS']].values
    average_carbon_stock = (csv_features[:, 0] + csv_features[:, 1] + csv_features[:, 2]).mean()
    
    # Predict on valid data
    prediction = model_rf(dict(valid_data))
    valid_predicted_classes = np.argmax(prediction, axis=1)
    # Prepare the output array
    image_predicted_classes = np.full(image_df.shape[0], np.nan)
    image_predicted_classes[valid_pixels] = valid_predicted_classes
    # Reshape to original image shape
    shape = (image.height, image.width)
    image_predicted_classes = image_predicted_classes.reshape(shape)

    # Save the Random Forest output to a temporary file
    temp_rf_output = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    with rasterio.open(
        temp_rf_output.name,
        'w',
        driver='GTiff',
        height=image_predicted_classes.shape[0],
        width=image_predicted_classes.shape[1],
        count=1,
        dtype=image_predicted_classes.dtype.name,
        crs=image.crs,
        transform=image.transform,
    ) as dst:
        dst.write(image_predicted_classes, 1)

    # Process the Random Forest output with KMeans clustering
    rf_image = rasterio.open(temp_rf_output.name)
    rf_data = rf_image.read()
    height, width = rf_image.height, rf_image.width
    num_bands = rf_data.shape[0]

    # Flatten the image data
    rf_data_flat = rf_data.reshape((-1, num_bands))

    # Exclude NaN pixels
    nan_mask = np.isnan(rf_data_flat).any(axis=1)
    valid_rf_data = rf_data_flat[~nan_mask]

    # Load CSV data from the uploaded file
    csv_data = pd.read_csv(temp_csv.name)
    csv_features = csv_data[['CA', 'CB', 'CS']].values

   # Downsample the valid TIF pixels to match the number of CSV rows
    num_csv_rows = csv_features.shape[0]
    downsampled_indices = np.linspace(0, valid_rf_data.shape[0] - 1, num=num_csv_rows, dtype=int)
    downsampled_image_data = valid_rf_data[downsampled_indices]

    # Normalize and apply weights
    image_weight = 1
    csv_weight = 1

    scaler = StandardScaler()
    valid_rf_data_normalized = scaler.fit_transform(downsampled_image_data)
    csv_features_normalized = scaler.fit_transform(csv_features)

    weighted_rf_data = valid_rf_data_normalized * image_weight
    weighted_csv_features = csv_features_normalized * csv_weight

    # Combine the weighted data
    combined_features = np.hstack((weighted_rf_data, weighted_csv_features))

    # Apply KMeans clustering
    cluster_labels = kmeans_model.predict(combined_features)

    # Create a full array for the clustered data
    # Start with an empty array (NaN for invalid pixels)
    clustered_full = np.full(valid_rf_data.shape[0], np.nan)

    # Assign labels to the valid pixels in the downsampled indices
    for i, idx in enumerate(downsampled_indices):
        clustered_full[idx] = cluster_labels[i]

    # Map clustered_full back to the full array, including NaN pixels
    full_clustered_image = np.full(rf_data_flat.shape[0], np.nan)
    full_clustered_image[~nan_mask] = clustered_full

    #  Reshape the clustered data back to the original TIF dimensions
    clustered_image = full_clustered_image.reshape(height, width)

    # Save the KMeans output to a temporary file
    temp_kmeans_output = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    with rasterio.open(
        temp_kmeans_output.name,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs=rf_image.crs,
        transform=rf_image.transform,
    ) as dst:
        dst.write(clustered_image.astype(np.float32), 1)

    # Define file names and paths
    rf_filename = f"{city_name}/random_forest.tif"
    kmeans_filename = f"{city_name}/final_model.tif"
    
    # Upload Random Forest output to Firebase Storage
    rf_blob = bucket.blob(rf_filename)
    rf_blob.upload_from_filename(temp_rf_output.name)
    rf_blob.make_public()
    rf_url = rf_blob.public_url

    # Upload KMeans output to Firebase Storage
    kmeans_blob = bucket.blob(kmeans_filename)
    kmeans_blob.upload_from_filename(temp_kmeans_output.name)
    kmeans_blob.make_public()
    kmeans_url = kmeans_blob.public_url

    # Store URLs and averages in Firestore
    doc_ref = db.collection('data').document(city_name)
    doc_ref.set({
        'city_name': city_name,
        'url_ndvi': rf_url,
        'url_final': kmeans_url,
        'average_ndvi': float(average_ndvi),
        'average_carbon_stock': float(average_carbon_stock)
    })

    return 'Upload successful', 200

if __name__ == '__main__':
    app.run(debug=True)