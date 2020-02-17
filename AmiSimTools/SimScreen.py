"""
This file contains classes used to run the simulated screenings of the AMI either in series or parallel
"""

__version__ = '1.0.0'

import numpy as np


class SimulatedScreenerParallel(object):
    """
    Class which uses an AMI model to perform a parallel simulated screening of materials from a dataset
    containing all features and target values for the entries.

    Additionally, this code performs for a multi step testing scenario where cheap and expensive tests can be conducted.
    """

    def __init__(self, data_params, test_costs, sim_budget, nthreads, num_init, min_samples):
        """
        :param data_params: DataTriage obj, contains triaged data including `y_true`, `y_experimental` and `status`
        :param test_costs: dict, {experiment_label:experiment_cost} used for updating simulation budget
        :param sim_budget: float, total amount of resources which can be used for the screening
        :param nthreads: int, number of threads to work on
        :param num_init: int, number of initial random samples to take
        :param min_samples: int, number of materials to be assessed before AMI model can fit
        """
        self.data_params = data_params
        self.test_costs = test_costs
        self.sim_budget = sim_budget
        self.nthreads = nthreads
        self.num_init = num_init
        self.min_samples = min_samples

        self.untested = 0
        self.max_cost = max(self.test_costs.values())
        self.lowest_test, self.highest_test = min(self.test_costs.keys()), max(self.test_costs.keys())
        self.workers = [(None, None)] * self.nthreads  # [(candidate_being_tested, test_id), ...]
        self.finish_time = np.zeros(self.nthreads)
        self.init_samples, self.queued = self._determine_queue()
        self.count_init_done = 0
        self.init_complete = False
        self.history = []
        self.model_fitted = False

    def _determine_queue(self):
        """
        Draws n samples from the pool of candidates i.e. [100, 4, 7892, ..., 0] and queues for screening

        :return: list[[material, test], ...]
        """
        queue = []

        init_samples = np.random.choice(self.data_params.n, self.num_init, replace=False)  # dont draw candidate twice

        for mat in init_samples:
            queue.append([mat, self.lowest_test])

        return init_samples, queue

    @staticmethod
    def determine_material_value(material, true_results, test=0):
        """
        Performs pseudo experiment for the AMI where the performance value of the AMI selected material is looked up in
        the loaded data array

        :param material: int, index of the material chosen in the target values
        :param true_results: np.array(), `m` sized array containing the target values for the passed features
        :param test: int, id of the test being run which will affect the indexing of the array slicing
        :return: determined_value: float, the target value for the passed material index
        """
        determined_value = true_results[material, test]
        return determined_value

    def _log_history(self, **kwargs):
        """
        Logs the history of the parallel screening.
        Continually updates by appending a dictionary to the history attribute which is eventually returned as a list.
        The list of dictionaries can then be readily converted into a pandas DataFrame after screening is complete.
        The start and end of experiments require different details saved, pandas allows for multiple different keywords
        allowing different keys to be passed depending on time of experiment.

        :param kwargs: dict, contains useful information about the simulated screening
        """
        self.history.append(kwargs)

    def _run_experiment(self, i, ipick, exp, exp_note):
        """
        Passed model selects a material to sample.
        If the material has not been tested before then a cheap test is run, otherwise run expensive.
        After each test, the budget is updated (contained within the model ?) and the worker finish time updated
        :param i: int, index of the worker to perform the task
        :param ipick: int, index of the material to be assessed
        :param exp: int, label of the experiment to be run, accesses cost records to price experiment
        :param exp_note: str, a comment to be saved to the logger
        """
        self.workers[i] = (ipick, exp)
        experiment_cost = np.random.uniform(self.test_costs[exp], self.test_costs[exp] * 2)
        self.sim_budget -= experiment_cost

        start = self.finish_time[i]
        self.data_params.status[ipick] += 1  # update status
        self.finish_time[i] += experiment_cost

        if self.init_complete == False:
            if ipick in self.init_samples:
                if exp == 1:
                    self.count_init_done += 1
                    if self.count_init_done == self.num_init:
                        self.init_complete = True

        self._log_history(note=exp_note, worker=i, candidate=ipick, time=start, exp_len=experiment_cost, test_id=exp)

    def _record_experiment(self, final):
        """
        After each experiment has been run, need to figure out the worker that will finish next.
        After each experiment, the model has to update its internal records of what has been tested and how.
        It then will update the history of the screening.
        Finally the index of the worker which has now finished is returned so that more work can be assigned.
        If the final parameter is `True` then there is no need to assign further work and so jobs are killed
        :param final: Boolean, indicates if on the final loop and should return anything or not
        :return: i: int, the index of the worker which is going to finish first
        """
        i = np.argmin(self.finish_time)  # get the worker which is closest to finishing
        idone, exp = self.workers[i]

        experimental_value = self.determine_material_value(idone, self.data_params.y_true, test=exp)
        self.data_params.y_experimental[idone][exp] = experimental_value
        self.data_params.status[idone] += 1  # update status
        end = self.finish_time[i]

        self._log_history(note='end', worker=i, candidate=idone, time=end, exp_value=experimental_value, test_id=exp)

        if self.init_complete == False:
            if idone in self.init_samples:
                if exp == 0:
                    self.queued.append([idone, 1])

        if final:
            self.workers[i] = (None, None)
            self.finish_time[i] = np.inf
        else:
            return i

    def _fit_if_safe(self, model):
        """
        If enough materials have been assessed as per user threshold, then allow the model to fit on the data
        obtained. If not enough then could potentially cause the model to crash on fitting which ruins the experiment.
        :param model: AMI model
        """
        complete_experiment = 4
        if sum(self.data_params.status == complete_experiment) >= self.min_samples:
            model.fit(self.data_params.y_experimental, self.data_params.status)
            self.model_fitted = True

    def _initial_materials(self):
        """
        Assigns each of the available workers/threads a random material from the initial queue.
        The worker then runs the material, and the queue is updated accordingly to remove the material.
        The queue is updated to ensure the same material isn't allocated multiple times.

        """
        for i in range(self.nthreads):
            ipick, exp = self.queued[0]
            del self.queued[0]
            self._run_experiment(i, ipick, exp, exp_note='start initial sample')

    def perform_screening(self, model):
        """
        Performs the full automated screening with multiple workers.
        First each worker (determined by the number of threads) is assigned a material to investigate.
        After this initialisation, the screener alternates selecting and recording experiments.
        This proceeds until the budget is spent (all the while recording the history of the work).
        After the budget is spent s.t. no expensive tests can be run, the remaining jobs finish.
        :param model: The AMI object performing the screening of the materials being investigated
        :return: self.history: list, full accounting of what materials were sampled when and where
        """
        self._initial_materials()

        while self.sim_budget >= self.max_cost:  # spend budget till cant afford any more expensive tests

            i = self._record_experiment(final=False)

            if self.init_complete == False:

                if len(self.queued) > 0:  # if queued materials then sample, else let model sample
                    ipick, exp = self.queued[0]
                    note = 'start initial sample'
                    del self.queued[0]
                else:
                    available_cheap = np.where(self.data_params.status == self.untested)[0]
                    ipick, exp = np.random.choice(available_cheap), self.lowest_test
                    note = 'start cheap filler'
            else:

                self._fit_if_safe(model)
                if self.model_fitted:
                    ipick, exp = model.pick_next(self.data_params.status)  # fit model and then allow to pick
                    note = 'start ami sample'
                else:
                    available_cheap = np.where(self.data_params.status == self.untested)[0]
                    available_expensive = np.where(self.data_params.status == 2)[0]
                    if np.random.rand() < 0.1 and len(available_expensive) > 0:  # 50:50 to pick cheap or expensive
                        ipick, exp = np.random.choice(available_expensive), 1
                    else:
                        ipick, exp = np.random.choice(available_cheap), self.lowest_test
                    note = 'start ami rand sample'

            self._run_experiment(i, ipick, exp=exp, exp_note=note)

        for i in range(self.nthreads):  # finish up any remaining jobs and record their results
            self._record_experiment(final=True)

        return self.history
