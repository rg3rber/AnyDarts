import os
import os.path as osp
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--all-models', action='store_true') # run all models
    parser.add_argument('-p', '--project', default='run2') # training config
    parser.add_argument('-e', '--experiment', default='train2') # experiment name

    args = parser.parse_args()
    
    list_of_models = [(args.project, args.experiment)]

    if args.all_models == True:
        list_of_models = [('first_run', 'train'), ('run2', 'train'), ('run2', 'train2'), ('run3', 'train'), ('run3', 'train-640'), ('run4', 'train'), ('run_gray', 'train'), ('run_v3', 'train')]

    for project, experiment in list_of_models:
        project_name = project
        experiment_name = experiment

        dir_to_check = osp.join(project_name, experiment_name)
        test_sets = []
        fps = []
        pcs = []
        mase = []
        for file in os.listdir(dir_to_check):
            if file.startswith('test_') and file.endswith('.csv'):
                test_sets.append(osp.splitext(file)[0])
                with open(osp.join(dir_to_check, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('# FPS:'):
                            number = float(line.split(':')[-1].strip())
                            fps.append(number)
                        if line.startswith('# PCS:'):
                            number_str = line.split(':')[-1].strip().replace('%', '')
                            number = float(number_str)/100
                            pcs.append(number)
                        elif line.startswith('# MASE:'):
                            number = float(line.split(':')[-1].strip())
                            mase.append(number)
                        else:
                            continue
            else:
                continue

        stats = pd.DataFrame()
        stats['Test Set Name'] = test_sets
        stats['FPS'] = fps
        stats['PCS'] = pcs
        stats['MASE'] = mase

        with open(osp.join(dir_to_check, 'combined_test_results.csv'), 'w') as f:
            stats.to_csv(f, index=False)