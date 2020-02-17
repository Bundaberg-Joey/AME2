#!/usr/bin/env python3

import argparse
from uuid import uuid4
import json

import pandas as pd
from Prospector_2 import prospector2

from AmiSimTools.DataTriage import DataTriageCSV
from AmiSimTools.SimScreen import SimulatedScreenerParallel


def load_costs(json_path):
    """
    For two stage experiments, experiments may have different costs associated with them which need to load into screen.
    These are just stored in an external `.json` file, the location of which is passed as a flag to this script.

    :param json_path: str, path to json file containing costs
    :return: costs, dict{int:float}, key is type integer for index slicing and value is float for costing
    """
    with open(json_path) as f:
        data = json.load(f)
    costs = {int(t): float(v) for t, v in zip(data['test_ids'], data['test_costs'])}
    return costs


def save_data(df, meta):
    """
    Want to save the output dataframe and the massociated meta data of the experiment to an output file.
    Can't do this as one file in pandas and hdf5 / .mat get really weird so will just use different files.
    file_id is generated by uuid4() meaning we won't need to worry about duplicate names till heat death of universe.

    :param df: DataFrame, history of the simulation we can be converted to and stored with pandas
    :param meta: dict, all the command line arguments used to set up the simulation

    """
    file_id = str(uuid4())
    meta['file_id'] = file_id

    df.to_csv(F'{file_id}.csv', index=False)
    with open(F'{file_id}.json', 'w') as f:
        f.write(json.dumps(meta, indent=4))


if __name__ == '__main__':

    data_location = r'Scaled_HCOF_F2.csv'
    cost_location = 'costs_HCOF.json'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', action='store', default=data_location, help='path to data file')
    parser.add_argument('-c', '--cost', action='store', type=str, default=cost_location, help='json file with costs')
    parser.add_argument('-b', '--budget', action='store', type=float, default=1000.0, help='simulation budget')
    parser.add_argument('-n', '--nthreads', action='store', type=int, default=200, help='# of parallel threads to run')
    parser.add_argument('-i', '--initial_samples', action='store', type=int, default=200, help='# of init rand samples')
    parser.add_argument('-m', '--min_samples', action='store', type=int, default=20, help='# of exp before fit')
    parser.add_argument('-a', '--acquisition', action='store', type=str, default='Thompson', help='Acquisition func')
    args = parser.parse_args()

    test_costs = load_costs(args.cost)
    sim_data = DataTriageCSV(args.data_file, n_tests=len(test_costs))

    sim_screen = SimulatedScreenerParallel(data_params=sim_data,
                                           test_costs=test_costs,
                                           sim_budget=args.budget,
                                           nthreads=args.nthreads,
                                           num_init=args.initial_samples,
                                           min_samples=args.min_samples)

    ami = prospector2(X=sim_data.X, costs=test_costs, acquisition_function=args.acquisition)

    sim_history = sim_screen.perform_screening(model=ami)

    df_sim = pd.DataFrame(sim_history)
    sim_meta = vars(args)
    save_data(df_sim, sim_meta)  # data is saved here rather than by `sim_screen` to preserve passed flags / meta data
