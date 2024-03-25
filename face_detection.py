import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

def detect_faces(image):
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    results = face_detection.process(image_rgb)

    return results

def crop_face(image, detection):
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = image.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                 int(bboxC.width * iw), int(bboxC.height * ih)
    
    # Crop the image using the top-left and bottom-right corners
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

def compute_histograms(image):
    # Calculate histograms for each channel
    r_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    b_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

    # Flatten histograms
    r_hist_flat = r_hist.flatten()
    g_hist_flat = g_hist.flatten()
    b_hist_flat = b_hist.flatten()

    # Concatenate the histograms into a single row
    hist_row = np.concatenate([r_hist_flat, g_hist_flat, b_hist_flat])
    hist_row = hist_row.reshape(1, -1)

    # Convert to pandas DataFrame
    hist_df = pd.DataFrame(hist_row)

    return hist_df

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

            # Detect faces in the image
            results = detect_faces(image)

            if results.detections:
                for detection in results.detections:
                    # Crop the face from the image
                    cropped_image = crop_face(image, detection)

                    # Compute histograms for the cropped face image
                    histogram = compute_histograms(cropped_image)

                    # Append histogram to data list along with celebrity name
                    data.append([celeb_name] + histogram.values.tolist())

    # Create DataFrame from the data
    columns = ['Celebrity'] + [f'pixel{i}' for i in range(len(data[0]) - 1)]
    df = pd.DataFrame(data, columns=columns)
    print(df.head(10))
    return df


def main():
    dataset_path = "/Users/muhammadhassan/Downloads/Celebrity Faces Dataset"
    histograms = process_dataset(dataset_path)
    # print(histograms)

if __name__ == "__main__":
    main()
