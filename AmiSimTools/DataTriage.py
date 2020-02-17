
"""
This file contains classes used to load and handle the data in order for it to be read from file and into the AMI
"""

__version__ = '1.0.0'

import warnings

import numpy as np


class DataTriageCSV(object):
    """
    This Parent takes a dataset which is to be assessed by the AMI and then performs the necessary pre-processing
    steps on the data set including: loading the data into numpy arrays, formatting the target values into the correct
    representation and other values.

    Children of this class allow for multiple types of files to be loaded and used depending on user requirements
    """

    def __init__(self, data_path, n_tests=1):
        self.data_path = data_path
        self.n_tests = n_tests
        self.X, self.y = self._load_dataset_from_path()
        self.n = len(self.y)
        self.y_true, self.y_experimental = self._format_target_values(self.y, self.n)
        self.status = np.zeros((self.n, 1))
        self.top_100 = np.argsort(self.y[-1])[-100:]


    def _format_target_values(self, y, n):
        """
        For simulated screenings, AMI requires an experimental column of results it has determined itself and the
        "True" target values which it uses to evaluate chosen materials against. These must be in the correct
        matrix / vector shape.

        :param y: np.array(), size `n` array containing the loaded target values
        :param n: int, the number of entries in the passed array `y`
        :return: (y_true, y_experimental), column vectors, [0] with all target values, [1] for determined values
        """
        y_true = y
        y_experimental = np.full((n, self.n_tests), np.nan)  # nan as values not yet determined on initialisation
        return y_true, y_experimental


    def _load_dataset_from_path(self):
        """
        Loads the features and target variables for the AME to assess from a delimited file, assumed csv.
        The default loading removes the first row to allow headed files to be read and so should be specified if not.
        The delimited file is assumed to be structured with target values as the right hand columns.

        Since multiple tests can be run, the number of "results" columns are used for parsing the results from features.
        It is assumed that the file is structured with:
            1) All results as far right hand columns
            2) The tests proceed from cheapest to most expensive - going left to right along the columns

        :return: features: np.array(), `m` by 'n' array which is the feature matrix of the data being modelled
        :return: targets: np.array(), `m` sized array containing the target values for the passed features
        """
        data_set = np.loadtxt(self.data_path, delimiter=",", skiprows=1, dtype='float')
        if data_set.size <= 0:
            warnings.warn('Loaded data set was empty')
        features, targets = data_set[:, :-self.n_tests], data_set[:, -self.n_tests:]
        return features, targets


########################################################################################################################
