#!/usr/bin/env python3

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import ray.cloudpickle as pickle

from training import SMALabellerApp

import os

home_dir = os.path.expanduser('~')

def ray_tune_do2_training(config):
    # === Step 1: Build CLI-style args ===
    # These were passed in from the Ray Tune config as a dict
    app_args = config["app_args"]  # should be a dict of CLI args

    # === Step 2: Instantiate your application with args ===
    app = SMALabellerApp(args=app_args)  # your class must now accept `args=dict`

    # === Step 3: Inject Ray-tuned hyperparameters ===
    # Ensure these are converted to native Python types

    # === Step 4: Run training ===
    app.do2dTraining(config,ray_tune=True)

"""
debug_config = config = {
                "lr": 1e-3,
                "max_lr": 1e-2,
                "loss_function": 'combination',
                "augmentations": True,
                "dropout_rate": 0,
                "app_args": {
                            "unet_dimensions": "2D",
                            "num_workers": 0,
                            "batch_size": 5,
                            "epochs": 10,
                            "training_images": "/panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/pytorch_model/Dataset503_MICCAI2012_and_316_CBIT_SMA_GM_ribbon/imagesTr_resampled",
                            "training_labels": "/panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/pytorch_model/Dataset503_MICCAI2012_and_316_CBIT_SMA_GM_ribbon/labelsTr_resampled",
                            "tb_prefix": "raytune",
                            "val_set_size": 0.2,
                }
            }
"""

config = {
                "lr": tune.loguniform(1e-5, 1e-2),
                "max_lr": tune.loguniform(1e-3, 1e-1),
                "loss_function": tune.choice(['dice', 'cross_entropy','combination']),
                "augmentations": tune.choice([True, False]),
                "dropout_rate": tune.uniform(0.0, 0.5),
                "app_args": {
                            "unet_dimensions": "2D",
                            "num_workers": 0,
                            "batch_size": 5,
                            "epochs": 10,
                            "training_images": "/panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/pytorch_model/Dataset503_MICCAI2012_and_316_CBIT_SMA_GM_ribbon/imagesTr_resampled",
                            "training_labels": "/panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/pytorch_model/Dataset503_MICCAI2012_and_316_CBIT_SMA_GM_ribbon/labelsTr_resampled",
                            "tb_prefix": "raytune",
                            "val_set_size": 0.2
                }
            }
algo=OptunaSearch()
tuner = tune.Tuner(ray_tune_do2_training,
    param_space=config,
    tune_config=tune.TuneConfig(
        metric="validation_loss",
        mode="min",
        num_samples=20,
        search_alg=algo,
        max_failures=0),
    run_config=tune.RunConfig(
        name="SMALabellerApp-2D",
        storage_path=os.path.join(home_dir, 'ray_results'),
        failure_config=FailureConfig(max_failures=0)
        checkpoint_config=tune.CheckpointConfig(
            num_to_keep=5,
            checkpoint_score_attribute="validation_loss",
            checkpoint_score_order="min"),
    verbose=1,
    progress_reporter=tune.JupyterNotebookReporter(overwrite=True, max_report_frequency=10)),
)
results = tuner.fit()
print("Best config: ", results.get_best_result().config)



