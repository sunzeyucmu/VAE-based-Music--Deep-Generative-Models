import os
import sys
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# __all__ = ['hello_world']


def load_audio(file, sr=22050, offset=0.0, duration=None, mono=False):
    # Librosa loads more filetypes than soundfile
    x, _ = librosa.load(file, sr=sr, mono=mono, offset=offset, duration=duration)
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    return x    

def hello_world():
	print("Hello world")

def hello_world2():
	print("Hello world")


"""
@description: Method to split a song into multiple songs using overlapping windows
- window==1.0, overlap==0.0 -> no split (original full sample length)
"""
def splitsongs(X, y, window = 0.05, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # pdb.set_trace()

    # Get the input song array size
    xshape = X.shape[-1] # signal length
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    if len(X.shape) == 1:
    	spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    else:
      spsong = [X[:, i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        # ignore the boundaries
        if s.shape[-1] != chunk:
            continue

        temp_X.append(s)
        temp_y.append(y)

    # (#samples, 1, sample_chunk_size)
    return np.array(temp_X), np.array(temp_y)


''' 
  1. Split each single 30s song sample into smaller segment
  2. convert each segment to melspectrogram
'''
def split_convert(X, y, sample_rate=3000, duration=30, max_signal_len=660000, split_window=1.0, split_overlap=0.0):
    arr_waves, arr_genres = [], []
    
    # Convert to spectrograms and split into small windows
    for fn, genre in zip(X, y):
        # for debug
        print(f"Loading audio file: {fn}...")

        # (1, sample_len)
        signal = load_audio(fn, sr=sample_rate, duration=duration)

        # to avoid inconsistent image/spectragram size
        signal = signal[:, :max_signal_len]

        # Convert to dataset of spectograms/melspectograms
        signals, y = splitsongs(signal, genre, window=split_window, overlap=split_overlap)

        # Convert to "spec" representation
        # specs = to_melspectrogram(signals, hop_length=512, n_fft=2048)

        # Save files
        print(signals.shape)
        ## attach the forked list of sample segments
        arr_waves.extend(signals)
        arr_genres.extend(y)

    # pdb.set_trace()

    # print(len(arr_waves), arr_waves[0].shape)
    # print(arr_genres.shape)

    return np.array(arr_waves), np.array(arr_genres)
    # return arr_waves, arr_genres

'''
 Generate data-set based on file_path for each audio sample
- then preprocess and convert each sample accordingly
'''
def read_data(src_dir, genres, test_data_percentage=0.1, sample_rate=22050, duration=30, split_window=1.0, split_overlap=0.0, max_signal_len=660000):    
    # Empty array of dicts with the processed features from all files
    arr_fn = []
    arr_genres = []

    # Get file list from the folders
    for x,_ in genres.items():
        folder = src_dir + x
        print(f"Loading Audio files under Genere: {x}")
        for root, subdirs, files in os.walk(folder):
            for file in files:
                file_name = folder + "/" + file

                ### TODO: some samples just don't load...
                if file_name.find('jazz/jazz.00054.wav') != -1:
                  print(f"Skipping file: {file_name}...")
                  continue

                # Save the file name and the genre
                arr_fn.append(file_name)
                # attach the label
                arr_genres.append(genres[x])
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        arr_fn, arr_genres, test_size=test_data_percentage, random_state=42, stratify=arr_genres #data is split in a stratified fashion under genre label
    )
    
    # Split into small segments and convert to spectrogram
    ## TODO: map this to TF DataSet process
    
    print("Loading and Preprocessing Testing data......")
    X_test, y_test = split_convert(X_test, y_test, sample_rate=sample_rate, duration=duration, max_signal_len=max_signal_len, split_window=split_window, split_overlap=split_overlap)

    print("Loading and Preprocessing Training data......")
    X_train, y_train = split_convert(X_train, y_train, sample_rate=sample_rate, duration=duration, max_signal_len=max_signal_len, split_window=split_window, split_overlap=split_overlap)

    # return X_test, y_test
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
	splitsongs(np.array([1,2,3,4]), np.array([0,1,0,1]), window=1.0, overlap=0.0)