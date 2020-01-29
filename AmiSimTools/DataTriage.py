
"""
This file contains classes used to load and handle the data in order for it to be read from file and into the AMI
"""

__author__ = 'Calum Hand'
__version__ = '1.0.0'

import warnings

import numpy as np
from scipy.io import loadmat


class DataTriage(object):
    """
    This Parent takes a dataset which is to be assessed by the AMI and then performs the necessary pre-processing
    steps on the data set including: loading the data into numpy arrays, formatting the target values into the correct
    representation and other values.

    Children of this class allow for multiple types of files to be loaded and used depending on user requirements
    """

    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.n = len(self.y)
        self.y_true, self.y_experimental = self.format_target_values(self.y, self.n)
        self.status = np.zeros((self.n, 1))
        self.top_100 = np.argsort(self.y)[-100:]


    @staticmethod
    def _load_dataset_from_path(data_path):
        """
        Method to be overloaded in further subclasses, loads data for the object

        :param data_path: str, path to data
        :return: x, y: (np.ndarray, np.ndarray)
        """
        return NotImplemented


    @classmethod
    def load_from_path(cls, data_path):
        """
        load data from the overloaded class method based on the location and then instantiate the object

        :param data_path: str, path to data
        :return: object, instantiated DataTriage object
        """
        X, y = cls._load_dataset_from_path(data_path)
        return cls(X, y)


    @staticmethod
    def format_target_values(y, n):
        """
        For simulated screenings, AMI requires an experimental column of results it has determined itself and the
        "True" target values which it uses to evaluate chosen materials against. These must be in the correct
        matrix / vector shape.

        :param y: np.array(), size `n` array containing the loaded target values
        :param n: int, the number of entries in the passed array `y`
        :return: (y_true, y_experimental), column vectors, [0] with all target values, [1] for determined values
        """
        y_true = y.reshape(-1, 1)  # column vector
        y_experimental = np.full((n, 1), np.nan)  # nan as values not yet determined on initialisation
        return y_true, y_experimental


class DataTriageCSV(DataTriage):
    """
    Child class which allows for the loading of data from a csv file
    """
    @staticmethod
    def _load_dataset_from_path(path: str) -> (np.ndarray, np.ndarray):
        """
        Loads the features and target variables for the AME to assess from a delimited file, assumed csv.
        The default loading removes the first row to allow headed files to be read and so should be specified if not.
        The delimited file is assumed to be structured with target values as the final right hand column.

        :param path: str, location of the csv data file to be read
        :return: features: np.array(), `m` by 'n' array which is the feature matrix of the data being modelled
        :return: targets: np.array(), `m` sized array containing the target values for the passed features
        """
        data_set = np.loadtxt(path, delimiter=",", skiprows=1, dtype='float')
        if data_set.size <= 0:
            warnings.warn('Loaded data set was empty')
        features, targets = data_set[:, :-1], data_set[:, -1]
        return features, targets


class DataTriageMatlab(DataTriage):
    """
    Child class which allows for the loading of data from a matlab file
    """
    @staticmethod
    def _load_dataset_from_path(path):
        """
        Loads the feature and target variables from a matlab file where the feature matrix and target array are in
        separate keys. The size of both the feature matrix and arrays are assessed and a warning is issued
        if the arrays are empty.

        :param path: str, location of the matlab file to be read
        :return: features: np.array(), `m` by 'n' array which is the feature matrix of the data being modelled
        :return: targets: np.array(), `m` sized array containing the target values for the passed features
        """
        data_set = loadmat(path, appendmat=False)
        feature_key, target_key = 'X', 'y'
        features, targets = data_set[feature_key], data_set[target_key]
        if features.size <= 0 or targets.size <= 0:
            warnings.warn('Loaded feature matrix or target array was empty')
        if features.shape[0] != targets.shape[0]:
            warnings.warn('The number of entries in the feature matrix and target array do not match')
        return features, targets

########################################################################################################################
