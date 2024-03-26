import re
import csv

# Probabilities for each classifier
classifier_files = [
    "C:/Users/AJAY/Desktop/melanoma-detection-main/probabilities/bagging_clf_prob.csv",
    "C:/Users/AJAY/Desktop/melanoma-detection-main/probabilities/clf_gini_prob.csv",
    "C:/Users/AJAY/Desktop/melanoma-detection-main/probabilities/logreg_prob.csv",
    "C:/Users/AJAY/Desktop/melanoma-detection-main/probabilities/random_forest_clf_prob.csv",
    "C:/Users/AJAY/Desktop/melanoma-detection-main/probabilities/svc_prob.csv",
    "C:/Users/AJAY/Desktop/melanoma-detection-main/probabilities/xgb_clf_prob.csv",
]

# Lists to store probabilities for class 0 and class 1
class_0_probabilities = []
class_1_probabilities = []

# Iterate over the probabilities of each classifier
for file_path in classifier_files:
    with open(file_path, 'r') as file:
        probabilities = file.read().splitlines()
        print(file_path)
        for prob in probabilities:
            prob = prob.strip("[]")
            # print(prob)
            numbers = prob.split()
            numbers = [float(num) for num in numbers]
            class_0_probabilities.append(numbers[0])
            class_1_probabilities.append(numbers[1])

# Calculate the average probabilities for each class
avg_class_0_probability = sum(class_0_probabilities) / len(class_0_probabilities)
avg_class_1_probability = sum(class_1_probabilities) / len(class_1_probabilities)

# Print the average probabilities
print("Average Probability for Class 0:", avg_class_0_probability)
print("Average Probability for Class 1:", avg_class_1_probability)

# Create a list to store the average probabilities for each image
avg_probabilities = []

# Iterate over the probabilities of each image
for i in range(189):
    image_probabilities = []
    for file_path in classifier_files:
        with open(file_path, 'r') as file:
            probabilities = file.read().splitlines()
            prob = probabilities[i].strip("[]")
            numbers = prob.split()
            numbers = [float(num) for num in numbers]
            image_probabilities.append(numbers)
    
    # Calculate the average probabilities for the image
    avg_image_probability = [sum(p) / len(p) for p in zip(*image_probabilities)]
    avg_probabilities.append(avg_image_probability)

# Write the average probabilities to a new file
output_file = "C:/Users/AJAY/Desktop/melanoma-detection-main/probabilities/avg_probabilities.csv"
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(avg_probabilities)

print("Average probabilities for each image have been written to", output_file)