import os
import shutil
from PIL import Image
import imagehash 

def calculate_image_hash(image_path):
    '''
    calculate the hash xyz value from imgs and compare which are duplicates and unquie
    '''
    image = Image.open(image_path)
    hash_value = imagehash.average_hash(image)
    return hash_value

def move_duplicates_and_unique_images(input_dir, output_dir_duplicates, output_dir_unique, threshold=5):
    # Create output directories if they don't exist
    os.makedirs(output_dir_duplicates, exist_ok=True)
    os.makedirs(output_dir_unique, exist_ok=True)

    # Dictionary to store hash values and corresponding file paths
    hash_dict = {}

    # Iterate through images in the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            image_path = os.path.join(root, file)

            # Calculate hash for the current image
            hash_value = calculate_image_hash(image_path)

            # Check if the hash is already in the dictionary
            if hash_value in hash_dict:
                # Move the duplicate image to the duplicates directory
                shutil.move(image_path, os.path.join(output_dir_duplicates, file))
            else:
                # Add the hash and file path to the dictionary
                hash_dict[hash_value] = image_path
                # Move the unique image to the unique directory
                shutil.move(image_path, os.path.join(output_dir_unique, file))

# Example usage
input_directory = '/home/muhammadhassan/Downloads/jpgs/'
output_directory_duplicates = '/home/muhammadhassan/Downloads/dupp'
output_directory_unique = '/home/muhammadhassan/Downloads/uique'
move_duplicates_and_unique_images(input_directory, output_directory_duplicates, output_directory_unique)
