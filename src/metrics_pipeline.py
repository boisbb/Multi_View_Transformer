import os
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
import torch
from sklearn.metrics import precision_recall_fscore_support

from src import utils

log = utils.get_logger(__name__)


def get_metrics(config: DictConfig) -> None:
    """Contains minimal example of the testing pipeline. Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Loading checkpoint from path <{config.ckpt_path}>")
    checkpoint = torch.load(config.ckpt_path, map_location=torch.device(config.device))
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    datamodule.setup()
    tst_loader = datamodule.test_dataloader()

    log.info(f"Predicting ..")
    for i, batch in enumerate(tst_loader):
        if i == 0:
            inputs = batch["view_patches"].to(device)
            targets = batch["labels"]
            preds = model.predict(inputs).detach().cpu()
        else:
            inputs = batch["view_patches"].to(device)
            outputs = model.predict(inputs).detach().cpu()
            targets = torch.cat((targets, batch["labels"]))
            preds = torch.cat((preds, outputs))

    targets = targets.numpy()
    preds = preds.detach().cpu().numpy()
    log.info(f"calculating metrics ..")
    precision, recall, fscore, _ = precision_recall_fscore_support(targets, preds, average="macro")
    log.info(f"precision: "+ str(precision))
    log.info(f"recall: " + str(recall))
    log.info(f"fscore: " + str(fscore))

    # Init lightning trainer
    log.info(f"Finished")

