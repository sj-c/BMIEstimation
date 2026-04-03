import re

file_name = "1_5.7h_67w_male_33a.png"

pattern = r"\d+_(\d+)\.(\d+)h_(\d+)w_(male|female)_\d+a\.(png|jpg)"
ret = re.match(pattern, file_name)

# Extract raw values
feet = int(ret.group(1))
inches = int(ret.group(2))
weight_kg = float(ret.group(3))
gender_str = ret.group(4)

# Convert height to meters
height_m = (feet * 12 + inches) * 0.0254

# Encode gender (0 = female, 1 = male)
gender = 1 if gender_str == "male" else 0

print("Height (m):", height_m)
print("Weight (kg):", weight_kg)
print("Gender (0=female):", gender)