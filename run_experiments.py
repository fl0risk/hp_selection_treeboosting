import argparse
import os

# import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization
from SMAC_method import ParameterOptimizationSMAC


def main(args):
    '''This function is used to run the experiments on the Euler Cluster. We changed the obj based on our goal, i.e. change to ParameterOptimizationSMAC or 
    change the boolean variable of try_num_leaves, try_max_depth etc..'''
    os.makedirs(args.result_folder, exist_ok=True)
    seed_folder = f"seed_{args.seed}"
    os.makedirs(os.path.join(args.result_folder, seed_folder), exist_ok=True)
    file_path = f"seed_{args.seed}/{args.suite_id}_{args.task_id}.csv"
    path = f"data/{args.suite_id}_{args.task_id}"

    # Read the data from local files
    X = pd.read_csv(os.path.join(path, f"{args.suite_id}_{args.task_id}_X.csv"))
    y = pd.read_csv(os.path.join(path, f"{args.suite_id}_{args.task_id}_y.csv"))
    categorical_indicator = np.load(os.path.join(path, f"{args.suite_id}_{args.task_id}_categorical_indicator.npy"))

    # benchmark_suite = openml.study.get_suite(args.suite_id)  # obtain the benchmark suite
    # task = openml.tasks.get_task(args.task_id)  # download the OpenML task
    # dataset = task.get_dataset()
    # X, y, categorical_indicator, attribute_names = dataset.get_data(
    #     dataset_format="dataframe", target=dataset.default_target_attribute
    # )

    suite_id = int(args.suite_id)
    task_id = int(args.task_id)
    seed = int(args.seed)

    # Run the experiment for the current task
    obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=False, seed=seed)
    final_results = obj.run_methods()


    # Format the DataFrame
    final_results["task_id"] = task_id
    final_results["classification"] = 1 if suite_id in [334, 337] else 0

    # Save the results
    final_results.to_csv(os.path.join(args.result_folder, file_path), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite_id')
    parser.add_argument('--task_id')
    parser.add_argument('--seed')
    parser.add_argument('--result_folder')
    args = parser.parse_args()
    main(args)
