import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# __all__ = ['hello_world']

SAMPLE_RATE = 3000

GENRES = {'metal': 0, 'disco': 1, 'classical': 2, 'rock': 3, 'jazz': 4,
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'hiphop': 9}

IDX_TO_GENRES = {v: k for (k, v) in GENRES.items()}

# TODO: hyper-parm this...
STFT_ARGS = [(2048, 1024, 512),  # n_fft
             (240, 120, 50),  # hop_length
             (1200, 600, 240)  # window_size
             ]


def spectral(x, n_fft, hop_length, window_length):
    # (N, F, T)
    # print("[DEBUG] X shape: {}".format(tf.shape(x)))
    raw_stft = tf.signal.stft(x, frame_length=window_length, frame_step=hop_length, fft_length=n_fft)
    # complex to float
    return tf.math.abs(raw_stft)


def norm(x):
    # [N, F, f] -> [N, Ff] -> [N]
    # n = tf.sqrt(tf.reduce_sum(tf.reshape(x, [tf.shape(x)[0], -1]) ** 2, axis=-1))
    # print("[DEBUG] STFT: \n", x)
    # TODO: 1. abs? 2: scale down the value (by bandwidth?)
    n = tf.norm(x, ord='fro', axis=[-2, -1])
    # return tf.cast(n, dtype=tf.float32)
    return n


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


def splitsongs(X, y, window=0.05, overlap=0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # pdb.set_trace()

    # Get the input song array size
    xshape = X.shape[-1]  # signal length
    chunk = int(xshape * window)
    offset = int(chunk * (1. - overlap))

    # Split the song and create new ones on windows
    if len(X.shape) == 1:
        spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
    else:
        spsong = [X[:, i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
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
    # book-keep the audio file name
    arr_file_label = []

    # Convert to spectrograms and split into small windows
    for fn, genre in zip(X, y):
        # for debug
        print(f"Loading audio file: {fn}...")
        file_label = fn.split('/')[-1]

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
        # under same file, clone the file_label for each sample
        arr_file_label.extend(np.repeat(file_label, len(y)))

    # pdb.set_trace()

    # print(len(arr_waves), arr_waves[0].shape)
    # print(arr_genres.shape)

    return np.array(arr_waves), np.array(arr_genres), np.array(arr_file_label)
    # return arr_waves, arr_genres


'''
 Generate data-set based on file_path for each audio sample
- then preprocess and convert each sample accordingly
'''


def read_data(src_dir, genres, test_data_percentage=0.1, sample_rate=22050, duration=30, split_window=1.0,
              split_overlap=0.0, max_signal_len=660000, shuffle_after_split=False, max_files_per_genre=1000):
    # Empty array of dicts with the processed features from all files
    arr_fn = []
    arr_genres = []

    # Get file list from the folders
    for x, _ in genres.items():
        folder = src_dir + x
        print(f"Loading Audio files under Genere: {x}")
        for root, subdirs, files in os.walk(folder):
            for file in files[:max_files_per_genre]:
                file_name = folder + "/" + file

                ### TODO: some samples just don't load...
                if file_name.find('jazz/jazz.00054.wav') != -1:
                    print(f"Skipping file: {file_name}...")
                    continue

                # Save the file name and the genre
                arr_fn.append(file_name)
                # attach the label
                arr_genres.append(genres[x])

    ## TODO: shuffle after songs split!!! Make sure 'stratify' at samples/songs level!
    if shuffle_after_split:
        print(f">>>>>>>>>>>>>>>>>> Splitting Songs First... >>>>>>>>>>>>>>>>>>>>>>>>")
        X, y, y_file = split_convert(arr_fn, arr_genres, sample_rate=sample_rate, duration=duration,
                      max_signal_len=max_signal_len, split_window=split_window,
                      split_overlap=split_overlap)

        print(f'>>>>>>>>>>>>>>>>>>Shuffling {len(X)} Instances with Test/Val Data percentage: {test_data_percentage}...>>>>>>>>>>>>>>>>')
        X_train, X_test, y_train, y_test, y_file_train, y_file_test = train_test_split(
            X, y, y_file,  test_size=test_data_percentage, random_state=42,
            # 1. data is split in a stratified fashion under genre label
            # 1. data is split in a stratified fashion under file label
            stratify=y_file
        )

    else:
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            arr_fn, arr_genres, test_size=test_data_percentage, random_state=42, stratify=arr_genres
            # data is split in a stratified fashion under genre label
        )

        # Split into small segments and convert to spectrogram
        ## TODO: map this to TF DataSet process

        print("Loading and Preprocessing Testing data......")
        X_test, y_test, y_file_test = split_convert(X_test, y_test, sample_rate=sample_rate, duration=duration,
                                       max_signal_len=max_signal_len, split_window=split_window,
                                       split_overlap=split_overlap)

        print("Loading and Preprocessing Training data......")
        X_train, y_train, y_file_train = split_convert(X_train, y_train, sample_rate=sample_rate, duration=duration,
                                         max_signal_len=max_signal_len, split_window=split_window,
                                         split_overlap=split_overlap)

    # return X_test, y_test
    return X_train, y_train, y_file_train, X_test, y_test, y_file_test


def generate_genre_samples(X, y, return_genre=False):
    """

    :param y:
    :return:
    """
    ## Multiple Genres Case
    idx_to_genres = {v: k for (k, v) in GENRES.items()}
    # create a dict of each unique entry and the associated indices
    generes_train_idx = {v: np.where(y == v)[0].tolist()[:6] for v in np.unique(y)}

    print(generes_train_idx)

    train_samples = []
    train_labels = []

    for i in range(len(np.unique(y))):
        train_samples.append(X[int(generes_train_idx[i][0])])
        # Quantization Version
        # train_samples.append(X_train_Q[int(generes_train_idx[i][0])])
        train_labels.append(i)

    # Numpy Array
    train_samples = np.stack(train_samples, axis=0)
    print(train_samples.shape, train_samples[0])

    if return_genre:
        return (train_samples, np.array(train_labels))
    else:
        return train_samples


if __name__ == '__main__':
    splitsongs(np.array([1, 2, 3, 4]), np.array([0, 1, 0, 1]), window=1.0, overlap=0.0)

    # plt.plot(np.arange(10))
