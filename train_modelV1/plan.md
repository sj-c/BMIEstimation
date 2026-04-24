Dataset source
csvs/filtered_images_cleaned.csv columns: source,image_path,bmi
for images: filtered_and_cropped_images/<source>/<filename>
2DImage2BMI → filtered_and_cropped_images/2DImage2BMI/<filename>
Celeb-FBI → filtered_and_cropped_images/Celeb-FBI/<filename>
visual_body_to_BMI → filtered_and_cropped_images/visual_body_to_BMI/<filename>

Train: 3550
visual_body_to_BMI    2404
2DImage2BMI           1060
Celeb-FBI               86

Val: 444
visual_body_to_BMI    300
2DImage2BMI           133
Celeb-FBI              11

Test: 445
visual_body_to_BMI    301
2DImage2BMI           133
Celeb-FBI              11


pre-training:
(1) split within each source so one dataset does not dominate one split.
train: 80%
val: 10%
test: 10%
csvs/train.csv
csvs/val.csv
csvs/test.csv

training step: 

cd train_model
python train.py

(1) Original image
→ (2) Augment
→ (3) DINOv2 ViT-B/14 distilled (with registers) (frozen)
→ (4) Extract CLS embedding: ViT-B: CLS ∈ ℝ^768
→ (5) Regression head (only train head)
→ (6) BMI

2: Augmentation 
Resize (preserve aspect ratio, long side = 224)
Pad to square (224×224)
RandomHorizontalFlip(p=0.5)
RandomRotation(±5 degrees)
ColorJitter(brightness/contrast/saturation mild)
RandomApply([
    ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.02
    )
], p=0.3)
Normalize using DINOv2 processor stats
"normalize": {
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225]
}

3: DINOv2
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

5: Regression Head
Image (224×224)
→ CLS (768-dim)
→ Linear(768, 256)
→ ReLU
→ Dropout(0.2)
→ Linear(256, 1)
→ BMI
Loss Func: MSELoss 
optimizer: Adam
learning rate: 1e-3
epochs: 50 (max)
early stopping: patience=5
batch size: 32 (if GPU allows)

Save: 
.pth 
loss graph (keep a csv of the loss changes-MAE and MSE)
Predicted vs Actual BMI (val set and test set)  

results:
Epoch 17: Train 18.8282 | Val 29.4934

test set:
Test MSE:  27.6163
Test RMSE: 5.2551
Test MAE:  3.7892

aim (waybed):2.56

Epoch 0: Train 144.2740 | Val 58.6781 
Epoch 1: Train 40.5613 | Val 47.3999 
Epoch 2: Train 34.7304 | Val 41.0713 
Epoch 3: Train 30.1186 | Val 35.8425 
Epoch 4: Train 28.1036 | Val 33.9611 
Epoch 5: Train 26.8266 | Val 35.2067 
Epoch 6: Train 24.6567 | Val 30.0393 
Epoch 7: Train 24.0490 | Val 30.2853 
Epoch 8: Train 23.4830 | Val 29.8700 
Epoch 9: Train 22.8103 | Val 32.4541 
Epoch 10: Train 22.0400 | Val 29.2847 
Epoch 11: Train 21.2294 | Val 28.5677 
Epoch 12: Train 20.3163 | Val 28.0868 
Epoch 13: Train 19.4833 | Val 28.6874 
Epoch 14: Train 20.4745 | Val 29.2968 
Epoch 15: Train 19.3734 | Val 28.5145 
Epoch 16: Train 18.5312 | Val 32.4436 
Epoch 17: Train 18.8282 | Val 29.4934