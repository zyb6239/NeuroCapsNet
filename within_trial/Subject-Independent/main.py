import os
import statistics
import mne
import numpy as np
from keras.callbacks import EarlyStopping
from eutils.kflod import five_fold
from preproc.util import list_data_split, normalize_data

from models.neurocap import create_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# path settings
model_name = 'neurocap' 
data_folder_name = '/disk2/ybzhang/dataset/asa_data' # it should be changed as your asa_data path
result_folder_name = './my_results' # it is recommanded to define a new result folder

# preprocessing settings
label_type = 'all_direction'
time_len = 1
sampling_rate = 128
sample_len, channels_num = int(sampling_rate * time_len), 64
overlap_rate = 0.5
window_sliding = int(sample_len * time_len * overlap_rate)
freq_bands = [[1, 50]]

# hyperparameters
lr = 5e-4
epochs = 100
batch_size = 64
num_class = 10


def preprocess_data(sub_id, freq_bands=[[1, 50]], data_folder="DATA", fs_new=128, label_type='all_direction'):
    """
    Preprocesses EEG data for a given subject, filtering and resampling the data according to specified frequency bands.

    Parameters:
    - sub_id: Identifier for the subject, used in file naming.
    - freq_bands: List of tuples, each specifying a frequency band to be filtered.
    - data_folder: Path to the folder where the raw EEG data is stored.
    - fs_new: New sampling rate for the resampled data.
    - label_type: Type of labels to be used, determining which trials to include.

    Returns:
    - data_list: List of preprocessed EEG data segments.
    - label_list: Corresponding labels for each data segment.
    """

    # Define labels based on direction
    labels = [0,9,9,0,1,8,1,8,2,7,7,2,3,6,3,6,4,5,5,4]
    # Select trials based on the label type
    if label_type == 'all_direction':
        selected_trails = range(20)
    elif label_type == '90':
        selected_trails = [0,1,2,3]
    elif label_type == '60':
        selected_trails = [4,5,6,7]
    elif label_type == '45':
        selected_trails = [8,9,10,11]
    elif label_type == '30':
        selected_trails = [12,13,14,15]
    elif label_type == '5':
        selected_trails = [16,17,18,19]

    # Print the subject ID being processed
    print(f"Processing Subject S{sub_id}")
    data_list = []
    label_list = []

    # Construct the file name based on the subject ID
    if len(sub_id) == 1:
        tmp_name = '00' + sub_id
    else:
        tmp_name = '0' + sub_id

    # Loop over the selected trials to read and preprocess the data
    for j in selected_trails:
        fifname = 'S' + tmp_name + '_E1_Trial' + str(j + 1) + '_raw.fif'
        path = data_folder + '/S' + tmp_name + '/E1/'
        # Read the raw EEG data file
        raw = mne.io.read_raw_fif(path + fifname, preload=True)

        # Perform average rereferencing on the EEG data
        raw = raw.set_eeg_reference(ref_channels='average', verbose=False)

        combined_data = []
        # Loop over each frequency band to filter and resample the data
        for freq_band in freq_bands:
            l_freq, h_freq = freq_band
            
            # Apply a low-pass filter if the lower frequency is below 1Hz
            if l_freq <= 1:
                raw.filter(None, h_freq, fir_design='firwin', verbose=False)
            else:
                # Otherwise, apply a band-pass filter
                raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)

            # Resample the data to the new sampling rate
            raw = raw.resample(fs_new, npad="auto", verbose=False)

            # Extract the data and transpose it for consistency
            data = raw.get_data()
            data = data.transpose()

            # Append the filtered and resampled data to the combined list
            combined_data.append(data)
        
        # Concatenate the data along the channel dimension
        combined_data = np.concatenate(combined_data, axis=-1)

        # Assign the label based on the trial index
        label = labels[j]
        
        # Append the combined data and label to the respective lists
        data_list.append(combined_data)
        label_list.append(label)

    # Return the preprocessed data and labels
    return data_list, label_list



def main():
    
    # Initialize dictionaries to store data, labels, and fold indices for all subjects
    all_subjects_data = {}
    all_subjects_label = {}
    all_subjects_fold_index = {}

    # Iterate over each subject to preprocess and partition their data
    for subject_id in range(20):
        sub_id = f'{subject_id + 1}'

        # Load and preprocess data for the subject
        # Data is expected to be in the form of trail * time * channels, with a sampling frequency of 128Hz
        data, label = preprocess_data(freq_bands=freq_bands, data_folder=data_folder_name, fs_new=sampling_rate, sub_id=sub_id, label_type=label_type)

        # Partition the data into training and validation sets
        data, label, split_index = list_data_split(
            data, None, label, time_len, window_sliding, 
            sampling_rate=sampling_rate, num_classes=num_class
        )

        # Normalize the data to have zero mean and unit variance
        data = [normalize_data(d) for d in data]

        # Split the data into 5 folds, ensuring the trials are not shuffled
        train_fold, test_fold = five_fold(
            data, label, split_index, shuffle=False, keep_trial=True
        )

        # Concatenate data and labels for the subject and store in the respective dictionaries
        all_subjects_data[sub_id] = np.concatenate(data, axis=0)
        all_subjects_label[sub_id] = np.concatenate(label, axis=0)

        # Store the fold indices for the subject
        all_subjects_fold_index[sub_id] = [train_fold, test_fold]

    # List to store the results of all folds
    all_folds_results = []

    # Perform 5-fold cross-validation
    for i_flod in range(5):
        x_train = []
        y_train = []

        # Prepare training data for the current fold
        for sub_id, [train_fold, test_fold] in all_subjects_fold_index.items():
            train_index = np.concatenate(train_fold[i_flod], axis=0)
            test_index = np.concatenate(test_fold[i_flod], axis=0)
            data = all_subjects_data[sub_id]
            label = all_subjects_label[sub_id]

            # Append data and labels for the current subject to the training set
            x_train.append(data[train_index])
            y_train.append(label[train_index])

        # Concatenate all training data from different subjects
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # Create and train the model with the training data
        model = create_model(sample_len=sample_len, channels_num=channels_num, lr=lr)
        early_stopping = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[early_stopping])

        # Dictionary to store results for each subject in the current fold
        sub_results = {}
        fold_accuracies = []  # To collect accuracies for calculating the mean accuracy

        # Evaluate the model on the test data for the current fold
        for sub_id, [train_fold, test_fold] in all_subjects_fold_index.items():
            trial_results = []
            data = all_subjects_data[sub_id]
            label = all_subjects_label[sub_id]
            for trial_index, test_index in enumerate(test_fold[i_flod]):
                x_test = data[test_index]
                y_test = label[test_index]
                loss, acc = model.evaluate(x_test, y_test, batch_size=8)
                trial_results.append(acc)

            # Store the accuracies for the current subject
            sub_results[sub_id] = trial_results
            fold_accuracies.extend(trial_results)  # Add to the list for mean calculation

        # Store the results of the current fold
        all_folds_results.append(sub_results)

        # Calculate and print the mean accuracy for the current fold
        mean_accuracy_fold = np.mean(fold_accuracies)
        print(f"Mean accuracy for fold {i_flod + 1}: {mean_accuracy_fold:.4f}")

    # Return the results of all folds
    return all_folds_results


def calculate_mean_and_std_accuracy(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Store all accuracies and each subject's average accuracy and standard deviation
    subject_means = []
    subject_results = {}
    
    for line in lines:
        parts = line.strip().split(':')
        subject_id = parts[0].strip()
        accuracies = [float(x) for x in parts[1].split(',')]
        mean_accuracy = sum(accuracies) / len(accuracies)
        subject_means.append(mean_accuracy)
        std_deviation = statistics.stdev(accuracies)
        subject_results[subject_id] = (mean_accuracy, std_deviation)
    
    return subject_results, subject_means


if __name__ == '__main__':
    """
    The main entry point of the script that processes the results from the main function,
    calculates the accuracy & standard deviation, and saves the results & average results.
    """

    # Call the main function to get the results of all folds
    all_folds_results = main()

    # Dictionary to store the mean accuracy scores for each subject, per fold
    subject_fold_scores = {}

    # Initialize the dictionary to ensure each subject has an entry for each fold
    for fold_index, fold_results in enumerate(all_folds_results):
        for sub_id in fold_results.keys():
            # Create a list of zeros for each subject's fold, assuming equal number of folds
            if sub_id not in subject_fold_scores:
                subject_fold_scores[sub_id] = [0] * len(all_folds_results)

    # Calculate the mean accuracy for each fold and store it in the dictionary
    for fold_index, fold_results in enumerate(all_folds_results):
        for sub_id, sub_results in fold_results.items():
            # Sum the accuracies of all trials in the current fold for the current subject
            fold_accuracy_sum = sum(sub_results)
            # Calculate the mean accuracy for the current fold
            mean_fold_accuracy = fold_accuracy_sum / len(sub_results)
            # Store the mean accuracy in the subject's fold scores
            subject_fold_scores[sub_id][fold_index] = mean_fold_accuracy

    # Construct the file name for the results based on the test version and sampling rate
    result_file_name = f"results_{model_name}_SR{sampling_rate}_OR{int(overlap_rate*100)}.txt"
    # Define the full path for the result file
    result_file_path = os.path.join(result_folder_name, result_file_name)

    # Check if the result folder exists, and create it if it does not
    if not os.path.exists(result_folder_name):
        os.makedirs(result_folder_name)

    # Open a file at the specified path and write the results
    with open(result_file_path, 'w') as file:
        for sub_id, scores in subject_fold_scores.items():
            # Write the subject's ID and their mean accuracy scores for each fold to the file
            file.write(f"Subject {sub_id}: {', '.join([f'{score:.4f}' for score in scores])}\n")

    # Print the location where the results have been saved
    print(f"Results saved to {result_file_path}")

    # Calculate each subject's mean accuracy and standard deviation, and collect all accuracies
    subject_results, subject_means = calculate_mean_and_std_accuracy(result_file_path)
    total_mean_accuracy = sum(result[0] for result in subject_results.values()) / len(subject_results)

    # Calculate the variance of the mean accuracies
    variance = sum((x - total_mean_accuracy) ** 2 for x in subject_means) / len(subject_means)

    # Calculate the standard deviation of the mean accuracies
    std_deviation_across_subjects = variance ** 0.5

    # Prepare data to be saved to file
    lines_to_save = [
        f"{subject_id} average accuracy: {result[0]:.4f}  standard deviation across 5 folds: {result[1]:.4f}"
        for subject_id, result in subject_results.items()
    ]
    lines_to_save.append(f"Overall average accuracy across all subjects: {total_mean_accuracy:.4f}")
    lines_to_save.append(f"Standard deviation across all subjects: {std_deviation_across_subjects:.4f}")

    # Construct the output file path
    output_dir = os.path.dirname(result_file_path)
    output_file_path = os.path.join(output_dir, 'averages_' + os.path.basename(result_file_path)[8:])

    # Save the results to a file
    with open(output_file_path, 'w') as output_file:
        for line in lines_to_save:
            output_file.write(line + '\n')

    print(f"Results saved to {output_file_path}")