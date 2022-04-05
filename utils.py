import os
import yaml

def get_history_logs_for_dataset(dataset):
    directory = os.fsencode("history/" + dataset)
    offline_logs = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            log = {"offline_policy": filename[:-(len('_dd-mm-yyyy_HH:MM:SS.csv'))], 
                   "filename": os.path.join(directory.decode('utf-8'), filename)}
            offline_logs.append(log)
        else:
            continue
    return offline_logs

def get_config_for_dataset(policy, dataset):
    if policy == 'random':
        return {}

    with open('configs/' + dataset +'/' + policy + '.yaml', "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config
