import os
import copy
import pytorch_lightning as pl
import os
os.environ["NCCL_DEBUG"] = "INFO"

from meter.config import ex
from meter.modules import METERTransformerSS
from meter.datamodules.multitask_datamodule import MTDataModule

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

@ex.automain
def main(_config):



    # _config["num_gpus"] = 8

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    
    print(f'exp_name: {_config["exp_name"]}!')
    print(f'learning rate: {_config["learning_rate"]}!')
    print(f'optim_type: {_config["optim_type"]}!')
    print(f'batch_size: {_config["batch_size"]}!')
    print(f'max_epoch: {_config["max_epoch"]}!')
    print(f'max_steps: {_config["max_steps"]}!')
    print(f'resume_from: {_config["resume_from"]}!')
    print(f'masking_strategy: {_config["masking_strategy"]}!')
    print(f'mlm_prob: {_config["mlm_prob"]}!')
    print(f'loss_names: {_config["loss_names"]}!')
    print(f'max_text_len rate: {_config["max_text_len"]}!')
    

    dm = MTDataModule(_config, dist=True)
    model = METERTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )


    from_path = f'{_config["load_path"].split("/")[-4]}+{_config["load_path"].split("/")[-3]}' if len(_config["load_path"].split("/")) >1 else ""


    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from#{from_path}',
        # name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",#"ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)

