# import os
# import csv
# import math
#
# def list_filenames(directory, train_output_file, test_output_file, train_ratio=0.7):
#     # Get all files in the specified directory
#     filenames = os.listdir(directory)
#
#     # Filter out only files (ignoring directories)
#     filenames = [f for f in filenames if os.path.isfile(os.path.join(directory, f))]
#     #
#     # # Sort the filenames
#     # filenames.sort()
#
#     # Calculate the split index for training and testing
#     total_files = len(filenames)
#     train_size = math.floor(total_files * train_ratio)
#
#     # Split filenames into training and testing sets
#     train_filenames = filenames[:train_size]
#     test_filenames = filenames[train_size:]
#
#     # Write the training filenames to the output CSV file (overwrite if exists)
#     with open(train_output_file, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         for filename in train_filenames:
#             writer.writerow([filename])  # Write each filename in a new row
#
#     # Write the testing filenames to the output CSV file (overwrite if exists)
#     with open(test_output_file, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         for filename in test_filenames:
#             writer.writerow([filename])  # Write each filename in a new row
#
# def main():
#     # Specify the directory and output files
#     directory = '/home/nan/GANimation/dataset/imgs'
#     train_output_file = '/home/nan/GANimation/dataset/train_ids.csv'
#     test_output_file = '/home/nan/GANimation/dataset/test_ids.csv'
#
#     list_filenames(directory, train_output_file, test_output_file)
#     print(f"Training filenames have been written to {train_output_file}")
#     print(f"Testing filenames have been written to {test_output_file}")
#
# if __name__ == '__main__':
#     main()
import os
import shutil
import csv
import math
import glob

def copy_and_rename_files(source_folder, dest_folder):
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Get all subfolders in the source folder
    subfolders = glob.glob(os.path.join(source_folder, 'IMG_*_aligned'))

    for subfolder in subfolders:
        # Extract the ID from the subfolder name
        folder_name = os.path.basename(subfolder)
        id_part = folder_name.split('_')[1]  # Get the ID part (****)

        # Iterate over all files in the subfolder
        for file in os.listdir(subfolder):
            if os.path.isfile(os.path.join(subfolder, file)):
                # Create new file name
                new_name = f'IMG_{id_part}_{file}'
                # Copy and rename the file
                shutil.copy(os.path.join(subfolder, file), os.path.join(dest_folder, new_name))

def list_filenames(dest_folder, train_output_file, test_output_file, train_ratio=0.7):
    # Get all files in the specified destination folder
    filenames = os.listdir(dest_folder)

    # Filter out only files (ignoring directories)
    filenames = [f for f in filenames if os.path.isfile(os.path.join(dest_folder, f))]

    # Calculate the split index for training and testing
    total_files = len(filenames)
    train_size = math.floor(total_files * train_ratio)

    # Split filenames into training and testing sets
    train_filenames = filenames[:train_size]
    test_filenames = filenames[train_size:]

    # Write the training filenames to the output CSV file (overwrite if exists)
    with open(train_output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for filename in train_filenames:
            writer.writerow([filename])  # Write each filename in a new row

    # Write the testing filenames to the output CSV file (overwrite if exists)
    with open(test_output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for filename in test_filenames:
            writer.writerow([filename])  # Write each filename in a new row

def main():
    # Specify the source folder and destination folder
    source_folder = '/home/nan/syracuse_pain_videos/openface_frame'  # Update with actual path
    dest_folder = '/home/nan/GANimation/dataset/imgs'  # Update with actual path
    train_output_file = '/home/nan/GANimation/dataset/train_ids.csv'  # Update with actual path
    test_output_file = '/home/nan/GANimation/dataset/test_ids.csv'  # Update with actual path

    # Copy and rename files from source to destination
    copy_and_rename_files(source_folder, dest_folder)

    # List filenames and create train/test splits
    list_filenames(dest_folder, train_output_file, test_output_file)

    print(f"Training filenames have been written to {train_output_file}")
    print(f"Testing filenames have been written to {test_output_file}")

if __name__ == '__main__':
    main()
