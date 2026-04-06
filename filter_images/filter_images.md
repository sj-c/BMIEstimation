1. build metadata_csv: build_metadata_csv.py


2: Filter out images
Version 1: 
extract CV features (bbox + keypoints): build_posture_training_csvs.py
train clustering: run_posture_training.py
Inspect clusters:  run_posture_training.py

Run Order: 
python BMIEstimation/csvs/build_metadata_csv.py
python -m csvs.build_posture_training_csvs
python filter_images/scripts/run_posture_training.py
python filter_images/scripts/export_cluster_samples.py
-choose clusters and update predict posture index
python BMIEstimation/filter_images/scripts/predict_posture.py

Version 2: (Used this)
python filter_images/filter_images.py
Zero keypoints — removes images where the model couldn't detect a joint (outputs 0,0 as a placeholder)
NaN keypoints — removes any rows with missing data
Front-facing — shoulder width > 15% of bbox width (side-on people have very narrow shoulders), and shoulders roughly centered
Upright — shoulders above hips, removes lying down / inverted images
Full body — ankles near bottom of bbox, torso spans enough height — removes cropped/partial body shots
Confidence — bbox confidence > 0.9, removes cluttered scenes with multiple people or poor detections

3: Crop Images based on bounding boxes:
python filter_images/crop_images.py
**saved images in filtered_images

pip install -U scikit-learn
