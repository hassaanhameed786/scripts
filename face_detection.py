import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


def make_histogram(image):
    # Split the image into RGB channels
    r, g, b = cv2.split(image)

    # Create histograms for each channel
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])

    # Flatten histograms
    r_hist_flat = r_hist.flatten()
    g_hist_flat = g_hist.flatten()
    b_hist_flat = b_hist.flatten()

    # Concatenate the histograms into a single row
    hist_row = np.concatenate([r_hist_flat, g_hist_flat, b_hist_flat])

    return hist_row


def process_dataset(dataset_path):
    data = []
    celeb_names = os.listdir(dataset_path)

    for celeb_name in celeb_names:
        celeb_path = os.path.join(dataset_path, celeb_name)
        if not os.path.isdir(celeb_path):  # Skip if not a directory
            continue
        print("Celebrity Path:", celeb_path)  # Add this line to check the path
        for img_name in os.listdir(celeb_path):
            if img_name.startswith('.'):  # Skip files starting with '.'
                continue
            img_path = os.path.join(celeb_path, img_name)
            image = cv2.imread(img_path)


            # Crop face from the image (you may need to adjust this part)
            # (Assuming you have a function to crop face like in the original code)
            # cropped_image = crop_face(image)

            # Calculate histogram for the cropped face image
            histogram = make_histogram(image)

            # Append histogram to data list along with celebrity name
            data.append([celeb_name] + histogram.tolist())

    # Create DataFrame from the data
    columns = ['Celebrity'] + [f'pixel{i}' for i in range(len(data[0]) - 1)]
    df = pd.DataFrame(data, columns=columns)

    csv_file_path = "/Users/muhammadhassaan/Downloads/celebrity_histograms/csv/histogram_data.csv"
    df.to_csv(csv_file_path, index=False)  # Set index=False to exclude row numbers from the CSV

    # Provide download link
    print("CSV file saved successfully.")
    print("Download link:", csv_file_path)
    return df


# Example usage:
dataset_path = "/Users/muhammadhassaan/Downloads/celebrity_histograms/Celebrity Faces Dataset"
histogram_df = process_dataset(dataset_path)



print(histogram_df.head(15))

