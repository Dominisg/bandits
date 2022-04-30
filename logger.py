from cProfile import run
import wandb
import neptune.new as neptune
import os
from datetime import datetime

class Logger:
    def log_config():
        raise NotImplementedError
    
    def log():
        raise NotImplementedError
    
    def finish():
        pass

def get_logger(name, run_name, project_name = 'bandits'):
    if name == 'wandb':
        return WandbLogger(run_name, project_name)
    elif name == 'neptune':
        return NeptuneLogger(run_name)
    else:
        return DummyLogger(run_name)

class WandbLogger(Logger):
    def __init__(self, run_name, project_name):
        wandb.init(project=project_name, reinit=True)
        wandb.run.name = run_name + ' (' + wandb.run.name + ')'
    
    def log_config(self, config):
        wandb.config.update(config)

    def log(self, dict):
        wandb.log(dict)

    def finish(self):
        wandb.finish()

class NeptuneLogger(Logger):
    def __init__(self, project_name):
        self.run = neptune.init(
            project=os.environ.get("NEPTUNE_PROJECT"),
            api_token=os.environ.get("NEPTUNE_API"),
        )

    def log_config(self, config):
        self.run["parameters"] = config

    def log(self, dict):
        for key, val in dict:
            self.run[key] = val
        
    def __del__(self):
        self.run.stop()

class DummyLogger(Logger):
    def __init__(self, project_name):
        pass

    def log_config(self, config):
        print(config)

    def log(self, dict):
        print(dict)
 
class HistoryLogger:
    def __init__(self, run_name, dataset):
        now = datetime.now() # current date and time
        self.filename = 'history/' + dataset + '/' + run_name + "_" + now.strftime("%m-%d-%Y_%H:%M:%S.csv")
    
    def log(self, context, action, reward):
        mode = 'a' if os.path.exists(self.filename) else 'w'
        with open(self.filename, mode) as f:
            if mode == 'w':
                for i in range(len(context[0])):
                    f.write(f"ctx{i},")
                for i in range(len(action[0])):
                    f.write(f"action{i},")
                f.write(f"reward\n")

            for i in range(len(context[0])):
                f.write(f"{context[0][i]},")
            for i in range(len(action[0])):
                f.write(f"{action[0][i]},")
            f.write(f"{reward[0]}\n")

        
        

                


    

