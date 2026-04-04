1. build metadata_csv: build_metadata_csv.py
2. extract CV features (bbox + keypoints): build_posture_training_csvs.py
3. train clustering: run_posture_training.py
4. Inspect clusters:  run_posture_training.py

Run Order: 
python build_metadata_csv.py
python build_posture_training_csvs.py
python run_posture_training.py
python export_cluster_samples.py