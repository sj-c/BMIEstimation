import re

file_name = "1a9089_a3eWh9O_138_69_false.jpg"

pattern = r"([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(\d+)_(\d+)_(true|false)\.jpg"
ret = re.match(pattern, file_name)

weight_lb = float(ret.group(3))
height_in = float(ret.group(4))
gender_str = ret.group(5)

# Convert
weight_kg = weight_lb * 0.4536
height_m = height_in * 0.0254

# Sex encoding: 0 if female
sex = 0 if gender_str == "true" else 1

print("Height (m):", height_m)
print("Weight (kg):", weight_kg)
print("Sex (0=female):", sex)