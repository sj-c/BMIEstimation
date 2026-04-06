pip install -U scikit-learn
1. build metadata_csv: build_metadata_csv.py

2: Filter out images
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


