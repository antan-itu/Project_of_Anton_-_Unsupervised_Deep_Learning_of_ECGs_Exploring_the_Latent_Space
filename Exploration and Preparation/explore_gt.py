import pandas as pd

# Define the output file name
OUTPUT_CSV = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Preparing MIMIC/training_metadata_sample.csv"

FILE_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Data/Full training dataset/training_dataset.h5"

print("Extracting Pandas DataFrame from 'GT'...\n")
df_ground_truth = pd.read_hdf(FILE_PATH, key='GT')

# Save the first 1000 rows to CSV
# index=False ensures the row numbers (index) are not saved as a column
df_ground_truth.head(100).to_csv(OUTPUT_CSV, index=False)

print(f"Success! The first 1000 rows have been saved to: {OUTPUT_CSV}")