"""Utility funcitons for running fitting experiment."""
import re
import os

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube import Experiment

from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from .modules import DeepMultiFusionMIL, DeepGenoFusionMIL, DeepGeno, FullFusionMIL

def run_fitting_experiment(Model_class, hparams, **kwargs):
    exp = Experiment(
        name=hparams.exp_name,
        debug=hparams.debug,
        save_dir=hparams.log_dir,
        version=hparams.exp_version,
        autosave=True
    )

    # set the hparams for the experiment
    exp.argparse(hparams)

    try:
        import git
        # Get current git commit has and store in experiment
        repo = git.Repo(search_parent_directories=True)
        exp.tag({
            'git-hash': repo.head.object.hexsha,
            'git-repo-state': 'dirty' if repo.is_dirty() else 'clean',
            # 'git-untracked-files': ','.join(repo.untracked_files)
        })
    except ImportError:
        # In case git python is not installed
        pass
    except ValueError:
        # Sometimes we have problems with gitpython
        pass

    exp.save()

    # Useful variables
    experiment_folder = exp.get_data_path(exp.name, exp.version)
    gpus = hparams.gpus if hparams.gpus != '' else None
    backend = 'dp' if hparams.gpus != '' else None

    # build model
    model = Model_class(hparams)

    # Callbacks
    checkpoint_callback = None
    if not hparams.debug:
        checkpoint_callback = ModelCheckpoint(
            filepath=experiment_folder,
            save_best_only=True,
            prefix='model',
            verbose=True,
            monitor='val_loss',
            mode='min'
        )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=False,
        mode='min'
    )

    trainer = pl.Trainer(
        experiment=exp,
        max_nb_epochs=hparams.max_epochs,
        gpus=gpus,
        log_gpu_memory=False,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping_callback,
        distributed_backend=backend,
        **kwargs
    )
    #amp_level='O2', use_amp=True
        # overfit_pct = 0.01,

    # train model
    trainer.fit(model)
    exp.save()
    if isinstance(model.model, DeepMultiFusionMIL):
        test_metrics = test_thermal_model(experiment_folder, model, gpu=gpus)
    elif isinstance(model.model, DeepGeno) or callable(getattr(model.model, "fwd_genotype", None)):
        test_metrics = test_genotype_model(experiment_folder, model, gpu=gpus, baseline=True)
    elif isinstance(model.model, DeepGenoFusionMIL):
        test_metrics = test_genotype_model(experiment_folder, model, gpu=gpus)
    elif isinstance(model.model, FullFusionMIL):
        test_metrics = test_genotype_model(experiment_folder, model, gpu=gpus, dem=True)
    else:
        test_metrics = test_model(experiment_folder, model, gpu=gpus)
    print('Model evaluation results:', test_metrics)
    exp.tag(test_metrics)
    exp.tag({'best_val_loss': early_stopping_callback.best})
    exp.save()


def test_model(experiment_folder, model, gpu):
    """Test model using either recovered weights from cehckpoint or last state.
    Args:
        experiment_folder: Folder where to look for checkpoint and
            hyperparameter settings.
        model: Model to use if we cannot recover from checkpoint
        gpu: Gpu on which we should load the model
    Returns:
        dictionary with metrics
    """
    if gpu is not None:
        # gpu is a string, thus this should also work with a singe gpu
        if len(gpu) > 1:
            gpu = gpu.split(',')[0]

    try:
        # Load model from checkpoint
        last_checkpoint = find_last_checkpoint(experiment_folder)
        if last_checkpoint is not None:
            # Load model from checkpoint
            model = model.__class__.load_from_metrics(
                weights_path=last_checkpoint,
                tags_csv=os.path.join(experiment_folder, 'meta_tags.csv'),
                on_gpu=gpu
            )
    except FileNotFoundError:
        print('Unable to load checkpoint, running test on current model state.')

    model_device = next(model.parameters()).device

    model.eval()
    model.freeze()

    val_dataloader = model.val_dataloader
    test_dataloader = model.test_dataloader

    metrics = {}
    for dataset_name, dataloader in zip(
        ['val', 'test'], [val_dataloader, test_dataloader]):
        y_preds = []
        all_labels = []
        for batch in dataloader:
            if len(batch) == 3:
                data, labels, length = batch
                dates = None
            elif len(batch) == 4:
                # This is for the temporal vectors
                data, labels, length, dates = batch 
                dates = dates.to(model_device)

            data = data.to(model_device)
            length = length.to(model_device)
            y_pred = model(data, length, dates).cpu().numpy()
            # batch_size = y_pred.size()[0]
            # y_pred = y_pred.squeeze()
            y_preds.append(y_pred)
            all_labels.append(labels)
        y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
        all_labels = np.concatenate(all_labels, axis=0).reshape(-1)

        loss = model.loss(
            torch.tensor(all_labels), torch.tensor(y_preds)).item()
        metrics[dataset_name + '_loss'] = loss
        # Compute other metrics MSE, r2, pearson
        mae = torch.mean(torch.abs(torch.tensor(all_labels)-torch.tensor(y_preds))).numpy()
        r2 = r2_score(all_labels, y_preds)
        pearson_score = pearsonr(all_labels, y_preds)[0]

        
        metrics[dataset_name + '_mae'] = mae
        metrics[dataset_name + '_r2'] = r2
        metrics[dataset_name + '_pearson'] = pearson_score

    return metrics

def test_thermal_model(experiment_folder, model, gpu):
    """Test model using either recovered weights from cehckpoint or last state, 
        adapted to the more complex model for data fusion.
    Args:
        experiment_folder: Folder where to look for checkpoint and
            hyperparameter settings.
        model: Model to use if we cannot recover from checkpoint
        gpu: Gpu on which we should load the model
    Returns:
        dictionary with metrics
    """
    if gpu is not None:
        # gpu is a string, thus this should also work with a singe gpu
        if len(gpu) > 1:
            gpu = gpu.split(',')[0]

    try:
        # Load model from checkpoint
        last_checkpoint = find_last_checkpoint(experiment_folder)
        if last_checkpoint is not None:
            # Load model from checkpoint
            model = model.__class__.load_from_metrics(
                weights_path=last_checkpoint,
                tags_csv=os.path.join(experiment_folder, 'meta_tags.csv'),
                on_gpu=gpu
            )
    except FileNotFoundError:
        print('Unable to load checkpoint, running test on current model state.')

    model_device = next(model.parameters()).device

    model.eval()
    model.freeze()

    val_dataloader = model.val_dataloader
    test_dataloader = model.test_dataloader

    metrics = {}
    for dataset_name, dataloader in zip(
        ['val', 'test'], [val_dataloader, test_dataloader]):

        y_preds = []
        all_labels = []
        for data_batch in dataloader:
            if model.model.temporal_encoding:
                # Also receives dates
                data_multispectral, data_thermal, data_genotype, labels, lengths_multispectral, \
                    lengths_thermal, dates_multispectral, dates_thermal = data_batch
                dates_multispectral = dates_multispectral.to(model_device)
                dates_thermal = dates_thermal.to(model_device)
            else:
                data_multispectral, data_thermal, data_genotype, labels, lengths_multispectral,\
                     lengths_thermal = data_batch
            data_multispectral = data_multispectral.to(model_device)
            data_thermal = data_thermal.to(model_device)
            data_genotype = data_genotype.to(model_device)
            lengths_multispectral = lengths_multispectral.to(model_device)
            lengths_thermal = lengths_thermal.to(model_device)
            if model.model.temporal_encoding:
                y_pred = model((data_multispectral, lengths_multispectral, data_thermal, data_genotype, \
                                        lengths_thermal,dates_multispectral, dates_thermal)).cpu().numpy()
            else:
                y_pred = model((data_multispectral, lengths_multispectral, data_thermal, data_genotype, lengths_thermal)).cpu().numpy()
            # batch_size = y_pred.size()[0]
            # y_pred = y_pred.squeeze()
            y_preds.append(y_pred)
            all_labels.append(labels)
        y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
        all_labels = np.concatenate(all_labels, axis=0).reshape(-1)

        loss = model.loss(
            torch.tensor(all_labels), torch.tensor(y_preds)).item()
        metrics[dataset_name + '_loss'] = loss
        # Compute other metrics MSE, r2, pearson
        mae = torch.mean(torch.abs(torch.tensor(all_labels)-torch.tensor(y_preds)))
        r2 = r2_score(all_labels, y_preds)
        pearson_score = pearsonr(all_labels, y_preds)[0]

        
        metrics[dataset_name + '_mae'] = mae
        metrics[dataset_name + '_r2'] = r2
        metrics[dataset_name + '_pearson'] = pearson_score

    return metrics

def test_genotype_model(experiment_folder, model, gpu, dem=False, baseline=False):
    """Test model using either recovered weights from cehckpoint or last state, 
        adapted to the more complex model for data fusion.
    Args:
        experiment_folder: Folder where to look for checkpoint and
            hyperparameter settings.
        model: Model to use if we cannot recover from checkpoint
        gpu: Gpu on which we should load the model
    Returns:
        dictionary with metrics
    """
    if gpu is not None:
        # gpu is a string, thus this should also work with a singe gpu
        if len(gpu) > 1:
            gpu = gpu.split(',')[0]

    try:
        # Load model from checkpoint
        last_checkpoint = find_last_checkpoint(experiment_folder)
        if last_checkpoint is not None:
            # Load model from checkpoint
            model = model.__class__.load_from_metrics(
                weights_path=last_checkpoint,
                tags_csv=os.path.join(experiment_folder, 'meta_tags.csv'),
                on_gpu=gpu
            )
    except FileNotFoundError:
        print('Unable to load checkpoint, running test on current model state.')

    model_device = next(model.parameters()).device

    model.eval()
    model.freeze()

    val_dataloader = model.val_dataloader
    test_dataloader = model.test_dataloader

    metrics = {}
    for dataset_name, dataloader in zip(
        ['val', 'test'], [val_dataloader, test_dataloader]):

        y_preds = []
        all_labels = []
        for data_batch in dataloader:
            x_forward, labels = get_x_forward(model, data_batch, model_device, 
                return_dem=dem, baseline=baseline)
            y_pred = model(x_forward).cpu().numpy()
            # batch_size = y_pred.size()[0]
            # y_pred = y_pred.squeeze()
            y_preds.append(y_pred)
            all_labels.append(labels)
        y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
        all_labels = np.concatenate(all_labels, axis=0).reshape(-1)

        loss = model.loss(
            torch.tensor(all_labels), torch.tensor(y_preds)).item()
        metrics[dataset_name + '_loss'] = loss
        # Compute other metrics MSE, r2, pearson
        mae = torch.mean(torch.abs(torch.tensor(all_labels)-torch.tensor(y_preds)))
        r2 = r2_score(all_labels, y_preds)
        pearson_score = pearsonr(all_labels, y_preds)[0]

        
        metrics[dataset_name + '_mae'] = mae
        metrics[dataset_name + '_r2'] = r2
        metrics[dataset_name + '_pearson'] = pearson_score

    return metrics

def get_x_forward(model, data_batch, model_device, return_dem=False, baseline=False):
    if baseline:
        data_genotype, labels = data_batch
        data_genotype = data_genotype.to(model_device)
        x_forward = data_genotype
    else: 
        if return_dem:
            if model.model.temporal_encoding:
                # Also receives dates
                data_multispectral, data_thermal, data_dem, data_genotype, labels, lengths_multispectral, \
                    lengths_thermal, lengths_dem, dates_multispectral, dates_thermal, dates_dem = data_batch
                dates_multispectral = dates_multispectral.to(model_device)
                dates_thermal = dates_thermal.to(model_device)
                dates_dem = dates_thermal.to(model_device)
            else:
                data_multispectral, data_thermal, data_dem, data_genotype, labels, lengths_multispectral, \
                    lengths_thermal, lengths_dem = data_batch
            data_dem = data_dem.to(model_device)
            lengths_dem = lengths_dem.to(model_device)
        else:
            if model.model.temporal_encoding:
                # Also receives dates
                data_multispectral, data_thermal, data_genotype, labels, lengths_multispectral, \
                    lengths_thermal, dates_multispectral, dates_thermal = data_batch
                dates_multispectral = dates_multispectral.to(model_device)
                dates_thermal = dates_thermal.to(model_device)
            else:
                data_multispectral, data_thermal, data_genotype, labels, \
                    lengths_multispectral, lengths_thermal = data_batch
        data_multispectral = data_multispectral.to(model_device)
        data_thermal = data_thermal.to(model_device)
        data_genotype = data_genotype.to(model_device)
        lengths_multispectral = lengths_multispectral.to(model_device)
        lengths_thermal = lengths_thermal.to(model_device)
        if return_dem:
            x_forward = (data_multispectral, lengths_multispectral, data_thermal, \
                            lengths_thermal, data_dem, lengths_dem, data_genotype, \
                            dates_multispectral, dates_thermal, dates_dem) if model.model.temporal_encoding \
                            else (data_multispectral, lengths_multispectral, data_thermal, \
                            lengths_thermal, data_dem, lengths_dem, data_genotype)
        else:    
            x_forward = (data_multispectral, lengths_multispectral, data_thermal, \
                            lengths_thermal, data_genotype, dates_multispectral, \
                            dates_thermal) if model.model.temporal_encoding else \
                            (data_multispectral, lengths_multispectral, data_thermal, \
                            lengths_thermal, data_genotype)
    return x_forward, labels

def find_last_checkpoint(filepath):
    """Find the last valid checkpoint inside the log dir.
    Return None if it cannot be found.
    """
    last_epoch = -1
    last_ckpt_name = None
    checkpoints = os.listdir(filepath)
    for name in checkpoints:
        # ignore hpc ckpts
        if 'hpc_' in name:
            continue

        if '.ckpt' in name:
            epoch = name.split('epoch_')[1]
            epoch = int(re.sub('[^0-9]', '', epoch))

            if epoch > last_epoch:
                last_epoch = epoch
                last_ckpt_name = name

    return os.path.join(filepath, last_ckpt_name) if last_ckpt_name else None
