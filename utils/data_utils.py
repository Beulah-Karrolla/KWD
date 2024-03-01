import csv
import sys
import os
import pandas as pd
import random
import pydub
from pydub import AudioSegment
import torchaudio

dataset_root = "/home/karrolla.1/KWD/data/speech_commands"

def create_csv_from_text_file(csv_file, text_file):
    with open(text_file, 'r') as txt_file:
        lines = txt_file.readlines()
    with open(csv_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['AudioPath', 'Label', 'Class'])
        for line in lines:
            csv_writer.writerow([(os.path.join(dataset_root, line.strip())), (line.strip().split('/'))[0], 0])


def create_full_csv_from_dataset_directory(csv_file):
    with open(csv_file, mode ="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['AudioPath', 'Label', 'Class'])
        for class_name in os.listdir(dataset_root):
            class_folder = os.path.join(dataset_root, class_name)
            if os.path.isdir(class_folder):
                for filename in os.listdir(class_folder):
                    file_path = os.path.join(class_folder, filename)
                    writer.writerow([str(file_path), class_name, 0])

def make_happy_1_class(file_path):
    df = pd.read_csv(file_path)
    condition = df['Label'] == 'happy'
    df.loc[condition, 'Class'] = 1
    df.to_csv(file_path, index=False)

def full_minus_test_plus_validation(self):
    full_csv = "/home/karrolla.1/KWD/full_data.csv"
    test_csv = "/home/karrolla.1/KWD/test_data.csv"
    validation_csv = "/home/karrolla.1/KWD/validation_data.csv"

    df_full = pd.read_csv(full_csv)
    df_test = pd.read_csv(test_csv)
    df_validation = pd.read_csv(validation_csv)
    
    merge_list_full = df_full.merge(df_test, on=list(df_full.columns), how ='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    merge_list_full = merge_list_full.merge(df_validation, on=list(merge_list_full.columns), how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

    merge_list_full.to_csv('/home/karrolla.1/KWD/train_data.csv', index=False)

def make_background_as_invalid():
    file_path = "/home/karrolla.1/KWD1/KWD/data/csv_files/st_validation_data_resampled.csv"
    df = pd.read_csv(file_path)
    condition = df['Label'] == '_background_noise_'
    df.loc[condition, 'Label'] = '-'
    df.to_csv(file_path, index=False)

def create_speech_text_csv():
    #dataset_root = "/homes/2/karrolla.1/KWD/data/speech_commands"
    new_csv_file = "/home/karrolla.1/KWD1/KWD/data/csv_files/st_validation_data.csv"
    old_csv_file = "/home/karrolla.1/KWD1/KWD/data/csv_files/validation_data.csv"

    folders = ['right', 'up', 'go', 'bird', 'five', 'wow', 'forward', 'follow', 
                'visual', 'down', 'marvin', '_background_noise_', 'left', 
                'sheila', 'learn', 'backward', 'dog', 'four', 'zero', 'tree', 
                'nine', 'happy', 'cat', 'two', 'on', 'off', 'six', 'seven', 
                'bed', 'yes', 'stop', 'no', 'eight', 'house', 'one', 'three']
    with open(new_csv_file, mode ="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['AudioPath', 'Label', 'Class'])
        with open(old_csv_file, mode="r") as old_file:
            reader = csv.reader(old_file)
            next(reader)
            for row in reader:
                writer.writerow([row[0], row[1], 1])
                for fol in folders:
                    if fol != row[1]:
                        writer.writerow([row[0], fol, 0])

def sampling_train_data():
    original_csv_path = "/home/karrolla.1/KWD1/KWD/data/csv_files/st_validation_data.csv"
    df_original = pd.read_csv(original_csv_path)

    # Assuming 'Class' is the column indicating the class (0 or 1)
    class_0_entries = df_original[df_original['Class'] == 0]
    class_1_entries = df_original[df_original['Class'] == 1]

    # Specify the number of entries you want to select randomly for each class
    num_entries_to_select = 4991 #min(len(class_0_entries), len(class_1_entries))

    # Randomly sample an equal number of entries from each class
    df_subset = pd.concat([
        class_0_entries.sample(n=num_entries_to_select, random_state=42),
        class_1_entries.sample(n=num_entries_to_select, random_state=42)
    ])

    # Shuffle the rows in the DataFrame
    df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Write the subset to a new CSV file
    subset_csv_path = "/home/karrolla.1/KWD1/KWD/data/csv_files/st_validation_data_resampled.csv"
    df_subset.to_csv(subset_csv_path, index=False)

def extract_one_second(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each WAV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the audio file using pydub
            audio = AudioSegment.from_file(input_path, format="wav")

            # Extract the first second
            one_second_audio = audio[:1000]  # 1000 milliseconds = 1 second

            # Save the one-second audio to the output folder
            one_second_audio.export(output_path, format="wav")

def create_speech_text_csv_swbd():
    #dataset_root = "/homes/2/karrolla.1/KWD/data/speech_commands"
    new_csv_file = "/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_nxt.csv"
    old_csv_file = "/home/karrolla.1/KWD1/KWD/data/csv_files/train_nxt.csv"

    '''folders = ['right', 'up', 'go', 'bird', 'five', 'wow', 'forward', 'follow', 
                'visual', 'down', 'marvin', '_background_noise_', 'left', 
                'sheila', 'learn', 'backward', 'dog', 'four', 'zero', 'tree', 
                'nine', 'happy', 'cat', 'two', 'on', 'off', 'six', 'seven', 
                'bed', 'yes', 'stop', 'no', 'eight', 'house', 'one', 'three']'''
    with open(new_csv_file, mode ="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['AudioPath', 'Label', 'Class'])
        with open(old_csv_file, mode="r") as old_file:
            reader = csv.reader(old_file)
            next(reader)
            for row in reader:
                row[3] = row[3].replace('/data/corpora/swb/swb1/data/', '/research/nfs_fosler_1/vishal/audio/swbd/')
                row[3] = row[3].replace(row[0]+'.sph', row[0]+'_'+row[1]+'_'+row[2]+'.wav')
                writer.writerow([row[3], row[4], 1])
                '''for fol in folders:
                    if fol != row[1]:
                        writer.writerow([row[0], fol, 0])'''

def create_negative_csv_swbd():
    csv_path = "/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data_filtered.csv"
    df = pd.read_csv(csv_path)

    # Assuming 'image1' and 'image2' are the columns representing positive pairs
    positive_pairs = list(zip(df['AudioPath'], df['Label']))

    # Generate negative pairs by randomly selecting two images that are not in the positive pairs
    negative_pairs = []

    # Ensure a balance between negative and positive pairs
    num_negative_pairs = len(positive_pairs)

    while len(negative_pairs) < num_negative_pairs:
        pair = (
            random.choice(df['AudioPath'].tolist()),
            random.choice(df['Label'].tolist())
        )
        if pair not in positive_pairs and pair not in negative_pairs:
            negative_pairs.append(pair)
            print(len(negative_pairs))

    # Create a new DataFrame for negative pairs
    negative_df = pd.DataFrame(negative_pairs, columns=['AudioPath', 'Label'])
    negative_df['Class'] = 0  # Assign class 0 label to negative pairs

    # Concatenate the positive and negative DataFrames
    result_df = pd.concat([df, negative_df], ignore_index=True)

    # Save the result DataFrame to a new CSV file
    result_csv_path = "/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data.csv"
    result_df.to_csv(result_csv_path, index=False)

def filter_valid_paths(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    # Filter rows with valid file paths
    valid_rows = []
    for index, row in df.iterrows():
        file_path = row['AudioPath']  # Replace 'file_path' with your actual column name
        if os.path.exists(file_path):
            valid_rows.append(row)
    valid_df = pd.DataFrame(valid_rows)
    # Save the new DataFrame to a new CSV file
    new_csv_path = '/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data_filtered.csv'
    valid_df.to_csv(new_csv_path, index=False)
    print(f"Filtered {len(df) - len(valid_df)} invalid entries. New CSV saved at {new_csv_path}")


def selected_swbd_train_data():
    df = pd.read_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data.csv')
    valid_rows = []
    for index, row in df.iterrows():
        print(index)
        input_path = os.path.join(row['AudioPath'])
        audio = AudioSegment.from_file(input_path, format="wav")
        if len(audio) < 5000 and len(row['Label']) < 40:
            valid_rows.append(row)
        '''if len(row['Label']) < 40:
            valid_rows.append(row)'''
    valid_df = pd.DataFrame(valid_rows)
    valid_df.to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data_length.csv', index=False)
    #df.to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data_length.csv', index=False)


def path_verification(csv_path):
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        file_path = row['AudioPath']
        #curr = self.data.iloc[index]
        #import ipdb;ipdb.set_trace()
        try:
            wavform, sr = torchaudio.load(file_path)
        except:
            print(file_path)
            print("some error in loading audio")
            return None
        if not os.path.exists(file_path):
            print(f"Invalid file path: {file_path}")

#path_verification('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data.csv')
# Replace 'path/to/your/old_file.csv' with the actual path to your CSV file
#filter_valid_paths('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_nxt.csv')
#extract_one_second("/home/karrolla.1/KWD1/KWD/data/speech_commands/_background_noise_", "/home/karrolla.1/KWD1/KWD/data/speech_commands/_")
#make_happy_1_class(sys.argv[1])
#create_speech_text_csv()
#sampling_train_data()
#make_background_as_invalid()
#create_speech_text_csv_swbd()
#create_negative_csv_swbd()
selected_swbd_train_data()