import mne
import numpy as np
import pickle
import os

data_folder_name = '/disk2/ybzhang/dataset/asa_data' 

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

def normalize_data(data):
    """
    Normalize the data to have a mean of 0 and standard deviation of 1.
    Args:
        data: The data to be normalized.

    Returns:
        The normalized data.
    """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def main():

    # Iterate over each subject to preprocess and partition their data
    for subject_id in range(20):
        sub_id = f'{subject_id + 1}'

        # Load and preprocess data for the subject
        # Data is expected to be in the form of trail * time * channels, with a sampling frequency of 128Hz
        data, label = preprocess_data(freq_bands=freq_bands, data_folder=data_folder_name, fs_new=sampling_rate, sub_id=sub_id, label_type=label_type)

        # Normalize the data to have zero mean and unit variance
        data = [normalize_data(d) for d in data]

        for tr in range(20):
            labels = np.full((data[tr].shape[0],),label[tr])
            labels_onehot = np.eye(num_class)[labels]
            label[tr] = labels_onehot

        save_dir = '/disk2/ybzhang/dataset/asa_processed/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + 'Sub' + sub_id + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({'data': data, 'label': label}, f)

    return None




if __name__ == '__main__':
    main()
