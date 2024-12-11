from flask import Flask, request, send_file
import tempfile
import rasterio
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pickle  # For loading the KMeans model
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom
import zipfile  # For packaging multiple files

app = Flask(__name__)

# Load the pre-trained random forest model
RF_PATH = 'NDVI_RF_V1'
FM_PATH = 'kmeans_model.pkl'
model_rf = tf.saved_model.load(RF_PATH)

# Load the pre-trained KMeans model
KM_PATH = 'kmeans_model.pkl'
with open(KM_PATH, 'rb') as model_file:
    kmeans_model = pickle.load(model_file)

# Define constants
FEATURES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
            'EVI', 'NBR', 'NDMI', 'NDWI', 'NDBI', 'NDBaI', 'NDVI', 'elevation']
ALL_FEATURES = FEATURES
CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

@app.route('/process', methods=['POST'])
def process_tif():
    # Receive the files from the request
    if 'file' not in request.files or 'csv_file' not in request.files:
        return 'Files are missing', 400
    tif_file = request.files['file']
    csv_file = request.files['csv_file']
    if tif_file.filename == '' or csv_file.filename == '':
        return 'No selected files', 400

    # Save the uploaded TIF file to a temporary location
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    tif_file.save(temp_input.name)

    # Save the uploaded CSV file to a temporary location
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    csv_file.save(temp_csv.name)

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

    # Upsample CSV data to match the number of valid pixels
    num_valid_pixels = valid_rf_data.shape[0]
    upsampled_csv_features = zoom(csv_features, (num_valid_pixels / csv_features.shape[0], 1), order=1)

    # Normalize and apply weights
    image_weight = 1
    csv_weight = 1

    scaler = StandardScaler()
    valid_rf_data_normalized = scaler.fit_transform(valid_rf_data)
    csv_features_normalized = scaler.fit_transform(upsampled_csv_features)

    weighted_rf_data = valid_rf_data_normalized * image_weight
    weighted_csv_features = csv_features_normalized * csv_weight

    # Combine the weighted data
    combined_features = np.hstack((weighted_rf_data, weighted_csv_features))

    # Apply KMeans clustering
    cluster_labels = kmeans_model.predict(combined_features)

    # Create a full array for the clustered data
    clustered_full = np.full(rf_data_flat.shape[0], np.nan)
    clustered_full[~nan_mask] = cluster_labels

    # Reshape to original image shape
    clustered_image = clustered_full.reshape(height, width)

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

    # Package both output files into a ZIP file
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        zipf.write(temp_rf_output.name, arcname='random_forest.tif')
        zipf.write(temp_kmeans_output.name, arcname='kmeans_output.tif')

    # Return the ZIP file
    return send_file(temp_zip.name, as_attachment=True, download_name='outputs.zip')

if __name__ == '__main__':
    app.run(debug=True)
