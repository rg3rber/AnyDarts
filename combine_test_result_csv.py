import os
import os.path as osp
import argparse
import pandas as pd
from yacs.config import CfgNode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='holo_v2') # run all models
    parser.add_argument('-a', '--all-models', action='store_true') # run all models
    parser.add_argument('-p', '--project', default='run2') # training config
    parser.add_argument('-e', '--experiment', default='train2') # experiment name

    args = parser.parse_args()

    cfg_path = osp.join('configs', args.cfg + '.yaml')
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    
    list_of_models = [(args.project, args.experiment)]

    if args.all_models == True:
        list_of_models = cfg.model.all_models
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
        
        # add total average
        test_sets.append('Total')
        fps.append(sum(fps)/len(fps))
        pcs.append(sum(pcs)/len(pcs))
        mase.append(sum(mase)/len(mase))
        
        stats = pd.DataFrame()
        stats['Test Set Name'] = test_sets
        stats['FPS'] = fps
        stats['PCS'] = pcs
        stats['MASE'] = mase

        with open(osp.join(dir_to_check, 'combined_test_results.csv'), 'w') as f:
            stats.to_csv(f, index=False)