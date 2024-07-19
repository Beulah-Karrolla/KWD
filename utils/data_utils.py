import csv
import sys
import os
import random
import pydub
from pydub import AudioSegment
import torchaudio
import torch
import numpy as np
from collections import Counter
#from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
#import nltk
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
sys.path.append('/homes/2/karrolla.1/')
#import whisperx
#from VAD.whisperX import whisperx

DEV = torch.device('cuda:{:d}'.format(2))
#nltk.download('punkt')
dataset_root = "/home/karrolla.1/KWD/data/speech_commands"
#torch.cuda.set_device(2)
#TOK = BertTokenizer.from_pretrained('bert-base-uncased')
#whisperx_trans = whisperx.load_model("large-v2",'cuda', device_index=0, compute_type="float16")
#whisperx_align, metadata = whisperx.load_align_model(language_code="en", device=DEV)

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
    original_csv_path = "/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data_small.csv"
    df_original = pd.read_csv(original_csv_path)

    # Assuming 'Class' is the column indicating the class (0 or 1)
    class_0_entries = df_original[df_original['Class'] == 0]
    class_1_entries = df_original[df_original['Class'] == 1]

    # Specify the number of entries you want to select randomly for each class
    num_entries_to_select = 77 #min(len(class_0_entries), len(class_1_entries))

    # Randomly sample an equal number of entries from each class
    df_subset = pd.concat([
        class_0_entries.sample(n=num_entries_to_select, random_state=42),
        class_1_entries.sample(n=num_entries_to_select, random_state=42)
    ])

    # Shuffle the rows in the DataFrame
    df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Write the subset to a new CSV file
    subset_csv_path = "/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data_smallest.csv"
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

def shuffle_csv(csv_path):
    
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Separate the header from the data
    header = data[0]
    data_without_header = data[1:]

    
    # Shuffle the data randomly
    random.shuffle(data_without_header, random=None)

    # Combine the header and shuffled data
    shuffled_data = [header] + data_without_header

    # Write the shuffled data back to the original CSV file
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(shuffled_data)


def shuffle_csv_effective(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Shuffle the DataFrame randomly
    shuffled_df = df.sample(frac=1, random_state=43).reset_index(drop=True)

    # Write the shuffled DataFrame back to the original CSV file
    shuffled_df.to_csv(csv_path, index=False)
    
    

def make_small_csv(csv_path, new_csv_path):
    df = pd.read_csv(csv_path)
    class_0 = df[df['Class'] == 0]
    class_1 = df[df['Class'] == 1]
    
    # Determine the minimum number of samples from each class
    min_samples = 5000
    
    # Sample an equal number of samples from each class
    sampled_class_0 = class_0.sample(n=min_samples, random_state=43)
    sampled_class_1 = class_1.sample(n=min_samples, random_state=43)
    sampled_df = pd.concat([sampled_class_0, sampled_class_1])
    
    # Shuffle the dataframe
    sampled_df = sampled_df.sample(frac=1, random_state=42)
    
    # Save the sampled dataframe to a new CSV file
    sampled_df.to_csv(new_csv_path, index=False)


def get_word_counts(csv_path):
    word_counts = Counter()
    total = 0
    chunk_size = 5000  # Adjust the chunk size based on your memory constraints
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        #import ipdb;ipdb.set_trace()
        # Concatenate all text entries in the current chunk into a single string
        all_text = ' '.join(chunk['Label'].astype(str))
        all_text = all_text.lower()
        tokens = word_tokenize(all_text)
        #tokens = TOK.tokenize(all_text)
        total += len(tokens)
        word_counts.update(tokens)
    sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    #print(word_counts)
    print(f"Total words: {total}")
    print(f"Unique words: {len(word_counts)}")
    with open('/home/karrolla.1/KWD1/KWD/data/csv_files/counts_nltk.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Word', 'Count'])
        for word, count in sorted_counts:
            #word_counts.items():
            writer.writerow([word, count])

def word_counts(csv_path):
    word_counts = Counter()
    df = pd.read_csv(csv_path)
    count = Counter(df['Label'])
    print(count)

def loss_plot(txt_path, save_path):
    log_file_path = txt_path
    epochs = []
    loss_train_values = []
    loss_val_values = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'epoch' in line and 'loss_train' in line:
                # Extract epoch and loss_vad values
                epoch = int(line.split('|')[1].split('=')[1].strip())
                loss_train = float(line.split('|')[2].split('=')[1].strip())
                loss_val = float(line.split('|')[4].split('=')[1].strip())
                
                # Append values to the lists
                epochs.append(epoch)
                loss_train_values.append(loss_train)
                loss_val_values.append(loss_val)

    # Plot the training loss curve
    plt.plot(epochs, loss_train_values, marker='.', linestyle='-', color='b', label='Train Loss')
    plt.plot(epochs, loss_val_values, marker='.', linestyle='-', color='r', label='Validation Loss')
    
    plt.title('Training Loss Vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (KWD)')
    #plt.ylim(0, np.mean(loss_values) + 0.1)
    #plt.ylim(-1, 0 + 0.1)
    plt.yticks([i * 0.05 / (0 + 1) for i in range(10)])
    plt.grid(True)
    plt.legend()
    #plt.show()
    plt.savefig(save_path)

def get_word_level_ground_truth(csv_path, device):
    df = pd.read_csv(csv_path)
    selected_words = get_selected_list('/home/karrolla.1/KWD1/KWD/data/csv_files/selected_bert.csv')
    word_level_ground_truth = {}
    valid_rows = []
    for index, row in df.iterrows():
        print(index)
        audio_path = row['AudioPath']
        label = row['Label']
        try:
            audio = whisperx.load_audio(audio_path)
            transcripts = whisperx_trans.transcribe(audio, 16000)
            result = whisperx.align(transcripts["segments"], whisperx_align, metadata, audio, device, return_char_alignments=False)
            for word in result["word_segments"]:
                new_row = {}
                if word['word'] in selected_words:
                    new_row['Label'] = word['word']
                    new_row['AudioPath'] = audio_path
                    new_row['Class'] = 1
                    new_row['Start'] = word['start']
                    new_row['End'] = word['end']
                    valid_rows.append(new_row)
        except:
            print(f"Error processing {audio_path}")
            continue
    import ipdb;ipdb.set_trace()
    valid_df = pd.DataFrame(valid_rows)
    valid_df.to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/word_level_ground_truth.csv', index=False)   
    #word_level_ground_truth[audio_path] = result["segments"]

def remove_hash_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    valid_rows = []
    for index, row in df.iterrows():
        if '##' not in row['Word']:
            valid_rows.append(row)
        #row['AudioPath'] = row['Word'].replace('#', '')
    valid_df = pd.DataFrame(valid_rows)
    valid_df.to_csv(csv_path, index=False)

def get_selected_list(csv_path):
    df = pd.read_csv(csv_path)
    selected_words = []
    for index, row in df.iterrows():
        selected_words.append(row['Word'])
    #valid_df = pd.DataFrame(valid_rows)
    return selected_words

def add_column_to_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['DatasetCountd_sample'] = 0
    df.to_csv(csv_path, index=False)


def update_count_column(csv_path):
    df1 = pd.read_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_word_level_whisperx_500_limit.csv')
    df2 = pd.read_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/selected_bert.csv')
    count = Counter(df1['Label'])
    valid_rows = []
    for index, row in df2.iterrows():
        #import ipdb;ipdb.set_trace()
        row['DatasetCountd_sample'] = count[row['Word']]
        valid_rows.append(row)
    valid_df = pd.DataFrame(valid_rows)
    valid_df.to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/selected_bert.csv', index=False) 

def create_500_samples_for_bert_selected():
    df = pd.read_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/selected_bert.csv')
    df1 = pd.read_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_word_level_whisperx.csv')
    empty_df = pd.DataFrame(columns=['Label', 'AudioPath','Class','Start','End'])
    for index, row in df.iterrows():
        label = row['Word']
        if row['DatasetCount'] > 500:
            sampled_df = df1[df1['Label'] == label].sample(n=500, random_state=42)
        else:
            sampled_df = df1[df1['Label'] == label]
        empty_df = pd.concat([empty_df, sampled_df])
    empty_df.to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_word_level_whisperx_500_limit.csv', index=False)
    #valid_df = pd.DataFrame(valid_rows)
    #valid_df.to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/selected_bert_500.csv', index=False)

def create_neg_csv_swbd():
    df = pd.read_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_word_level_whisperx_500_limit.csv')
    neg_rows = []
    positive_pairs = list(zip(df['AudioPath'], df['Label']))
    for index, row in df.iterrows():
        new_row = row
        pair = (row['AudioPath'], random.choice(df['Label'].tolist()))
        if (pair not in positive_pairs):
            new_row['Class'] = 0
            new_row['AudioPath'] = row['AudioPath']
            new_row['Label'] = pair[1]
            new_row['Start'] = row['Start']
            new_row['End'] = row['End'] 
            neg_rows.append(new_row)
    neg_df = pd.DataFrame(neg_rows, columns=['Label', 'AudioPath','Class','Start','End'])
    result_df = pd.concat([df, neg_df])
    result_df.to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_word_level_data.csv', index=False)

def swbd_train_valid_test_split(csv_path=None):
    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_word_level_data.csv')

    # Step 2: Split the DataFrame into two based on the class
    class_0 = df[df['Class'] == 0]
    class_1 = df[df['Class'] == 1]

    # Step 3: Shuffle each class-specific subset
    class_0 = class_0.sample(frac=1, random_state=42).reset_index(drop=True)
    class_1 = class_1.sample(frac=1, random_state=42).reset_index(drop=True)

    # Step 4: Split each class-specific subset into train, validation, and test sets
    train_0, temp_0 = train_test_split(class_0, test_size=0.3, random_state=42)
    valid_0, test_0 = train_test_split(temp_0, test_size=0.5, random_state=42)

    train_1, temp_1 = train_test_split(class_1, test_size=0.3, random_state=42)
    valid_1, test_1 = train_test_split(temp_1, test_size=0.5, random_state=42)

    # Step 5: Concatenate the class-specific sets to obtain the final train, validation, and test sets
    train_set = pd.concat([train_0, train_1], ignore_index=True).sample(frac=1, random_state=42)
    valid_set = pd.concat([valid_0, valid_1], ignore_index=True).sample(frac=1, random_state=42)
    test_set = pd.concat([test_0, test_1], ignore_index=True).sample(frac=1, random_state=42)

    train_set.to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/word_level_train_set.csv', index=False)
    valid_set.to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/word_level_valid_set.csv', index=False)
    test_set.to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/word_level_test_set.csv', index=False)
    # Now, train_set, valid_set, and test_set have balanced class distribution

def data_prep():
    df = pd.read_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/word_level_train_set.csv')
    for index, row in df.iterrows():
        import ipdb;ipdb.set_trace()
        audio_path = row['AudioPath']
        #audio = whisperx.load_audio(audio_path)
        #transcripts = whisper


torch.cuda.set_device(2)
device = torch.device('cuda:{:d}'.format(2))
#make_small_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data_length.csv', '/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data_small.csv')
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
#create_neg_csv_swbd()
#selected_swbd_train_data()
#shuffle_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_word_level_data.csv')
#shuffle_csv_effective('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_word_level_data.csv')
#path_verification('/home/karrolla.1/KWD1/KWD/data/csv_files/st_train_data_resampled.csv')
#sampling_train_data()
#get_word_counts('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data_filtered.csv')
#loss_plot('/home/karrolla.1/KWD1/KWD/saved/beulah/swbd_length_train_512_bert1_st00_2fc_conf_5_sig_bce_rop.txt', '/home/karrolla.1/KWD1/KWD/saved/conformer/loss_plots/swbd_length_train_512_bert1_st00_2fc_conf_5_sig_bce_rop_train.png')
loss_plot('/home/karrolla.1/KWD1/KWD/saved/beulah/plot_data.txt', '/home/karrolla.1/KWD1/KWD/saved/conformer/loss_plots/train_vs_val.png')
#get_word_level_ground_truth('/home/karrolla.1/KWD1/KWD/data/csv_files/swbd_train_data_filtered.csv', device)
#add_column_to_csv('/home/karrolla.1/KWD1/KWD/data/csv_files/selected_bert.csv')
#update_count_column('/home/karrolla.1/KWD1/KWD/data/csv_files/selected_bert.csv')
#create_500_samples_for_bert_selected()
#swbd_train_valid_test_split()
#data_prep()