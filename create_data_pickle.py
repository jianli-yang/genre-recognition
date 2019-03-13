from common import load_track, GENRES
import sys
import numpy as np
from math import pi
from pickle import dump
import os
from optparse import OptionParser

TRACK_COUNT = 8000

def get_default_shape(dataset_path):
    # tmp_features, _ = load_track(os.path.join(dataset_path,
    #     'blues/blues.0000'), is_mel=True)
    # return tmp_features.shape
    return (934, 128)

def collect_data(dataset_path):
    '''
    Collects data from the GTZAN dataset into a pickle. Computes a Mel-scaled
    power spectrogram for each track.

    :param dataset_path: path to the GTZAN dataset directory
    :returns: triple (x, y, track_paths) where x is a matrix containing
        extracted features, y is a one-hot matrix of genre labels and
        track_paths is a dict of absolute track paths indexed by row indices in
        the x and y matrices
    '''
    default_shape = get_default_shape(dataset_path)
    print(default_shape)
    #x = np.zeros((TRACK_COUNT,) + default_shape, dtype=np.float32)
    x = []
    y = np.zeros((TRACK_COUNT, len(GENRES)), dtype=np.float32)
    track_paths = {}

    for (genre_index, genre_name) in enumerate(GENRES):
        for i in range(TRACK_COUNT // len(GENRES)):
            file_name = '{}/{}.{}'.format(genre_name,
                    genre_name, str(i).zfill(4))
            print('Processing', file_name)
            path = os.path.join(dataset_path, file_name)
            track_index = genre_index * (TRACK_COUNT // len(GENRES)) + i
            # x[track_index], _ = load_track(path, default_shape, is_mel=True)
            # x[track_index] = 0
            y[track_index, genre_index] = 1
            track_paths[track_index] = os.path.abspath(path)
            #track_paths[track_index] = file_name

    return x, y, track_paths

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--dataset_path', dest='dataset_path',
            default=os.path.join(os.path.dirname(__file__), 'data/genres'),
            help='path to the GTZAN dataset directory', metavar='DATASET_PATH')
    parser.add_option('-o', '--output_pkl_path', dest='output_pkl_path',
            default=os.path.join(os.path.dirname(__file__), 'data/data.pkl'),
            help='path to the output pickle', metavar='OUTPUT_PKL_PATH')
    options, args = parser.parse_args()

    (x, y, track_paths) = collect_data(options.dataset_path)

    data = {'x': x, 'y': y, 'track_paths': track_paths}
    # with open(options.output_pkl_path, 'wb') as f:
    #     dump(data, f)
    np.save(options.output_pkl_path, np.array(data))
