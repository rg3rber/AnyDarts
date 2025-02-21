import pandas as pd
import os

def is_pkl_file(file):
    return file.endswith('.pkl')

path = 'dataset/annotations/'
# Get list of files in the directory
label_files = [f for f in os.listdir(path) if is_pkl_file(os.path.join(path, f))]

labels = []
for label_file in label_files:
    df = pd.read_pickle(os.path.join(path, label_file))
    df["img_folder"] = label_file.split(".")[0]
    labels.append(df)

labels = pd.concat(labels)
pd.to_pickle(obj=labels, filepath_or_buffer="dataset/labels.pkl")