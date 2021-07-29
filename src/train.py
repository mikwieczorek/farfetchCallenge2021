# Train

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from checkpointer_callback import ModelCheckpointPeriodic
from dataset import FarfetchBaselineDataset
from model import Model
from trainer import FarfetchTrainer

N_SKETCHES = 4  # 4 sketches: clicked products in previous sessions, clicked in this session, viewed in this session, viewed in this query

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where files will be saved",
    )

    parser.add_argument(
        "--train-datapoints",
        type=str,
        default="train_datapoints_sequential_reproducing",
        help="Training datapoints",
    )
    parser.add_argument(
        "--val-datapoints",
        type=str,
        default="validation_datapoints_sequential_reproducing",
        help="Validation datapoints",
    )
    parser.add_argument(
        "--test-datapoints",
        type=str,
        default="test_datapoints_sequential_reproducing",
        help="Test datapoints",
    )
    parser.add_argument(
        "--codes-filename_clicked",
        type=str,
        default="codes_clicked",
        help="Filename of final sketches for clicked products",
    )
    parser.add_argument(
        "--codes-filename_viewed",
        type=str,
        default="codes_viewed",
        help="Filename of final sketches for viewed products",
    )
    parser.add_argument("--sketch-dim", type=int, default=128, help="Sketch width")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=2900, help="Hidden size")
    return parser


def get_nunique_values(value, train, valid):
    return max(set([i[value] for i in train] + [i[value] for i in valid])) + 1


def sparse2dense(codes, input_dim):
    res = np.zeros(input_dim)
    res[codes] += 1
    return res


def main(params: dict):
    data_dir = Path(params.data_dir)

    product2codes = np.load(
        data_dir / params.codes_filename_clicked, allow_pickle=True, mmap_mode=True
    )

    product2codes_viewed = np.load(
        data_dir / params.codes_filename_viewed, allow_pickle=True, mmap_mode=True
    )

    products_dict = np.load(
        data_dir / "products_dict", allow_pickle=True, mmap_mode=True
    )

    n_sketches = len(product2codes[list(product2codes.keys())[0]])
    log.info(f"Sketch depth: {n_sketches}")

    input_dim = params.sketch_dim * n_sketches

    train_datapoints = np.load(
        data_dir / params.train_datapoints,
        allow_pickle=True,
        mmap_mode=True,
    )

    train_dataset = FarfetchBaselineDataset(
        train_datapoints,
        product2codes,
        products_dict,
        input_dim,
        params.sketch_dim,
        product2codes_viewed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        num_workers=10,
        shuffle=True,
        drop_last=False,
    )

    validation_datapoints = np.load(
        data_dir / params.val_datapoints, allow_pickle=True, mmap_mode=True
    )
    validation_dataset = FarfetchBaselineDataset(
        validation_datapoints,
        product2codes,
        products_dict,
        input_dim,
        params.sketch_dim,
        product2codes_viewed,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=params.batch_size,
        num_workers=10,
        shuffle=False,
        drop_last=False,
    )
    test_original_datapoints = np.load(
        data_dir / params.test_datapoints, allow_pickle=True, mmap_mode=True
    )
    test_original_dataset = FarfetchBaselineDataset(
        test_original_datapoints,
        product2codes,
        products_dict,
        input_dim,
        params.sketch_dim,
        product2codes_viewed,
    )
    test_original_loader = DataLoader(
        test_original_dataset,
        batch_size=params.batch_size,
        num_workers=10,
        shuffle=False,
        drop_last=False,
    )

    all_products_viewed = list(product2codes_viewed.keys())
    abs_codes_th_viewed = [product2codes_viewed[pid] for pid in all_products_viewed]
    abs_codes_th_viewed = np.array(abs_codes_th_viewed)
    abs_codes_th_viewed = torch.from_numpy(abs_codes_th_viewed)

    n_page_type = get_nunique_values(
        "page_type",
        train_datapoints,
        validation_datapoints + test_original_datapoints,
    )
    n_previous_page_type = get_nunique_values(
        "previous_page_type2id",
        train_datapoints,
        validation_datapoints + test_original_datapoints,
    )
    n_device_category = get_nunique_values(
        "device_category2id",
        train_datapoints,
        validation_datapoints + test_original_datapoints,
    )
    n_device_platform = get_nunique_values(
        "device_platform2id",
        train_datapoints,
        validation_datapoints + test_original_datapoints,
    )
    n_user_tier = get_nunique_values(
        "user_tier2id",
        train_datapoints,
        validation_datapoints + test_original_datapoints,
    )
    n_user_country = get_nunique_values(
        "user_country2id",
        train_datapoints,
        validation_datapoints + test_original_datapoints,
    )
    n_context_type = get_nunique_values(
        "context_type2id",
        train_datapoints,
        validation_datapoints + test_original_datapoints,
    )
    n_context_value = get_nunique_values(
        "context_value2id",
        train_datapoints,
        validation_datapoints + test_original_datapoints,
    )

    logger = TensorBoardLogger(
        "logs",
        name="runs",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="{epoch}",
        monitor="mrr",
        mode="max",
        verbose=True,
    )

    periodic_checkpointer = ModelCheckpointPeriodic(
        dirname=os.path.join(logger.log_dir, "auto_checkpoints"),
        filename_prefix="checkpoint",
        n_saved=1,
        save_interval=1,
    )

    net = Model(
        n_sketches,
        params.sketch_dim,
        N_SKETCHES,
        params.hidden_size,
        n_page_type,
        n_previous_page_type,
        n_device_category,
        n_device_platform,
        n_user_tier,
        n_user_country,
        n_context_type,
        n_context_value,
        n_sketches_output=1,
    )

    model = FarfetchTrainer(
        net,
        params.lr,
        params.sketch_dim,
        all_products_viewed,
        abs_codes_th_viewed,
    )
    trainer = pl.Trainer(
        gpus=1,
        auto_select_gpus=True,
        max_epochs=3,
        logger=logger,
        callbacks=[periodic_checkpointer],
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    main(params)
