import numpy as np
import os
from tqdm import tqdm
import argparse
import glob
import re
import pickle

# parser = argparse.ArgumentParser()
# parser.add_argument('-ia', '--input_aus_filesdir', type=str, help='Dir with imgs aus files')
# parser.add_argument('-op', '--output_path', type=str, help='Output path')
# args = parser.parse_args()
#
# def get_data(filepath):
#     data = dict()
#     content = np.loadtxt(filepath, delimiter=',', skiprows=1)  # Load entire file, skipping the header
#
#     # Loop through each row and create keys in the format "frame_0001"
#     for i in range(content.shape[0]):  # Iterate through each row
#         key = f'frame_{i + 1:04}'  # Create key as "frame_0001"
#         data[key] = content[i, 1:18]  # Store data from columns 1 to 17
#
#     return data
#
# def save_dict(data, name):
#     with open(name + '.pkl', 'wb') as f:
#         pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#
# def main():
#     filepaths = glob.glob(os.path.join(args.input_aus_filesdir, '*.csv'))
#     filepaths.sort()
#
#     # create aus file
#     data = get_data(filepaths)
#
#     if not os.path.isdir(args.output_path):
#         os.makedirs(args.output_path)
#     save_dict(data, os.path.join(args.output_path, "aus"))
#
#
# if __name__ == '__main__':
#     main()
# import numpy as np
# import os
# from tqdm import tqdm
# import argparse
# import pickle
#
# # Argument parser setup
# parser = argparse.ArgumentParser()
# parser.add_argument('-ia', '--input_aus_file', type=str, help='Path to the input AUS CSV file')
# parser.add_argument('-op', '--output_path', type=str, help='Output path')
# args = parser.parse_args()
#
#
# def get_data(filepath):
#     data = dict()
#     content = np.loadtxt(filepath, delimiter=',', skiprows=1)  # Load entire file, skipping the header
#
#     # Loop through each row and create keys in the format "frame_0001"
#     for i in range(content.shape[0]):  # Iterate through each row
#         key = f'frame_det_00_00{i:04}'  # Create key as "frame_0001"
#         data[key] = content[i, 5:22]  # Store data from columns 1 to 17
#
#     return data
#
#
# def save_dict(data, name):
#     with open(name + '.pkl', 'wb') as f:
#         pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#
#
# def main():
#     # Use the single input file path provided
#     filepath = args.input_aus_file
#
#     # Ensure the input file exists
#     if not os.path.isfile(filepath):
#         print(f"Error: The file {filepath} does not exist.")
#         return
#
#     # Create aus file from the input file
#     data = get_data(filepath)
#     print(data)
#
#     # Ensure the output directory exists
#     if not os.path.isdir(args.output_path):
#         os.makedirs(args.output_path)
#
#     # Save the data
#     save_dict(data, os.path.join(args.output_path, "aus"))
#
#
# if __name__ == '__main__':
#     main()
import numpy as np
import os
import glob
import argparse
import pickle

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('-ia', '--input_folder', type=str, help='Path to the input folder containing CSV files')
parser.add_argument('-op', '--output_path', type=str, help='Output path')
args = parser.parse_args()


def get_data(filepath):
    data = dict()
    content = np.loadtxt(filepath, delimiter=',', skiprows=1)  # Load entire file, skipping the header

    # Extract the filename without the extension
    filename = os.path.basename(filepath).replace('.csv', '')

    # Loop through each row and create keys in the format "<filename>_frame_0001"
    for i in range(content.shape[0]):  # Iterate through each row
        key = f'{filename}_frame_det_00_00{i:04}'  # Create key as "<filename>_frame_det_00_000001"
        data[key] = content[i, 5:22]  # Store data from columns 1 to 17

    return data


def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def main():
    # Use the input folder path provided
    input_folder = args.input_folder

    # Ensure the input folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: The folder {input_folder} does not exist.")
        return

    # Collect all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in the folder: {input_folder}")
        return

    # Initialize a dictionary to store data from all files
    all_data = {}

    # Process each CSV file
    for filepath in csv_files:
        print(f"Processing file: {filepath}")
        file_data = get_data(filepath)
        all_data.update(file_data)  # Combine data from each file

    # Print out the collected data
    # print(all_data)

    # Ensure the output directory exists
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # Save the combined data
    save_dict(all_data, os.path.join(args.output_path, "aus_openface"))

if __name__ == '__main__':
    main()
