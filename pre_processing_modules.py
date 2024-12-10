from kfp.v2.dsl import component, Input, Output, Dataset
import configparser
config = configparser.ConfigParser()
config.read("../config.ini")

@component(
    packages_to_install=["pandas", "gcsfs", "scikit-learn", "numpy", "google-cloud-storage"],
    output_component_file="output_files/basic_preprocessing.yaml",
    base_image="python:3.11",
)
def basic_preprocessing(
    BUCKET_URI: str,
    FILE1: str,
    FILE2: str,
    dataset: Output[Dataset],
):
    import pandas as pd
    from google.cloud import storage

    # Step 1: Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(BUCKET_URI.replace('gs://', '').split('/')[0])

    # Step 2: Define categories we want to keep
    clothing_items_top_10 = [
        "Topwear", "Bottomwear", "Innerwear", "Dress", 
        "Loungewear and Nightwear", "Saree", "Headwear", 
        "Ties", "Scarves", "Apparel Set"
    ]

    # Step 3: Read both CSV files from GCS
    df_uri1 = "".join([BUCKET_URI, FILE1])  # images.csv
    df_uri2 = "".join([BUCKET_URI, FILE2])  # styles.csv
    images_df = pd.read_csv(df_uri1, on_bad_lines='skip')
    styles_df = pd.read_csv(df_uri2, on_bad_lines='skip')

    # Step 4: Process images_df to get ID from filename
    images_df['id'] = images_df['filename'].str.replace('.jpg', '').astype(int)

    # Step 5: Merge datasets on ID
    df_merged = pd.merge(images_df, styles_df, on='id', how='inner')

    # Step 6: Remove unnecessary columns
    df_merged.drop(columns=[
        'link', 'gender', 'masterCategory', 'articleType', 
        'baseColour', 'season', 'year', 'usage', 'productDisplayName'
    ], inplace=True)

    # Step 7: Get list of actual images in GCS
    blobs = bucket.list_blobs(prefix='raw_images/')
    image_files = {blob.name.split('/')[-1] for blob in blobs if blob.name.endswith('.jpg')}

    # Step 8: Filter dataset to only include:
    # - Images that exist in GCS
    # - Images in our top 10 categories
    df_filtered = df_merged[
        (df_merged['filename'].isin(image_files)) & 
        (df_merged['subCategory'].isin(clothing_items_top_10))
    ].copy()

    # Step 9: Cap each category at 1000 images
    capped_dataset = pd.DataFrame()
    for category in clothing_items_top_10:
        subset = df_filtered[df_filtered['subCategory'] == category]
        if len(subset) > 1000:
            subset = subset.sample(n=1000, random_state=42)
        capped_dataset = pd.concat([capped_dataset, subset])

    # Step 10: Save the processed dataset
    capped_dataset.to_csv(dataset.path + ".csv", index=False)

@component(
    packages_to_install=["pandas", "gcsfs", "pillow", "tensorflow", "numpy", "google-cloud-storage"],
    output_component_file="output_files/feature_engineering.yaml",
    base_image="python:3.11",
)
def data_transformation(
    df: Input[Dataset],
    BUCKET_URI: str,
    dataset: Output[Dataset],
    
):
    import pandas as pd
    import tensorflow as tf
    from google.cloud import storage
    import os
    import tempfile

    # Read the input dataframe
    df = pd.read_csv(df.path + ".csv")
    
    # Initialize GCS client
    client = storage.Client()
    bucket_name = BUCKET_URI.replace('gs://', '').split('/')[0]
    bucket = client.bucket(bucket_name)

    # Get the first filename to verify the path structure
    first_filename = df['filename'].iloc[0]
    print(f"First filename: {first_filename}")  # Add this for debugging

    # Set up data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    def augment_and_save_image(blob, label, new_filename):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download image from GCS
            local_path = os.path.join(temp_dir, "temp_image.jpg")
            blob.download_to_filename(local_path)
            
            # Load and augment image
            img = tf.keras.preprocessing.image.load_img(local_path)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            augmented_img = data_augmentation(img_array)
            augmented_img = tf.keras.preprocessing.image.array_to_img(augmented_img[0])
            
            # Save augmented image locally
            augmented_path = os.path.join(temp_dir, new_filename)
            augmented_img.save(augmented_path)
            
            # Upload augmented image to GCS
            new_blob = bucket.blob(f"processed_images/{label}/{new_filename}")
            new_blob.upload_from_filename(augmented_path)
            return f"gs://{bucket_name}/processed_images/{label}/{new_filename}"

    # Process each category
    balanced_data = []
    for category in df['subCategory'].unique():
        category_df = df[df['subCategory'] == category]
        current_count = len(category_df)
        
        # Add existing images to processed folder
        for _, row in category_df.iterrows():
            # Get the source blob using the full path from raw_images
            source_blob = bucket.blob(f"raw_images/{row['filename']}")
            dest_blob = bucket.blob(f"processed_images/{category}/{row['filename']}")
            
            try:
                dest_blob.rewrite(source_blob)
                balanced_data.append({
                    'filename': row['filename'],
                    'subCategory': category,
                    'gcs_file_path': f"gs://{bucket_name}/processed_images/{category}/{row['filename']}"
                })
            except Exception as e:
                print(f"Error processing {row['filename']}: {str(e)}")
                continue
        
        # Augment if needed
        if current_count < 1000:
            needed = 1000 - current_count
            for i in range(needed):
                source_row = category_df.sample(n=1).iloc[0]
                new_filename = f"aug_{category}_{i}.jpg"
                source_blob = bucket.blob(f"raw_images/{source_row['filename']}")
                
                try:
                    gcs_path = augment_and_save_image(source_blob, category, new_filename)
                    balanced_data.append({
                        'filename': new_filename,
                        'subCategory': category,
                        'gcs_file_path': gcs_path
                    })
                except Exception as e:
                    print(f"Error augmenting image {new_filename}: {str(e)}")
                    continue

    # Create final balanced dataframe
    df_balanced = pd.DataFrame(balanced_data)

    # Create final dataframe with only required columns
    final_df = pd.DataFrame({
        'gcs_file_path': df_balanced['gcs_file_path'],
        'label': df_balanced['subCategory'],
    })

    print(final_df.head())
    # Save the balanced dataset
    final_df.to_csv(dataset.path + ".csv", index=False)

