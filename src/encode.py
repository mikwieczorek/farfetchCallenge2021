import argparse
import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from cleora import run_cleora_directed
from coder import DLSH

log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where files will be saved",
    )
    parser.add_argument(
        "--clicked",
        type=str,
        default="cleoraInput_sessionIdGrouped_onlyClicked",
        help="Filename of cliced products dataset",
    )
    parser.add_argument(
        "--viewed",
        type=str,
        default="cleoraInput_sessionIdGrouped_viewed",
        help="Filename of cliced products dataset",
    )
    parser.add_argument("--sketch-dim", type=int, default=128, help="Sketch width")
    parser.add_argument(
        "--n-sketches", type=int, default=40, help="Cleora sketch depth"
    )
    parser.add_argument(
        "--n-sketches-random", type=int, default=40, help="Random sketch depth"
    )
    parser.add_argument(
        "--cleora-dim",
        type=int,
        default=1024,
        help="Emedding length of Cleora embedding",
    )
    parser.add_argument(
        "--cleora-iterations_clicked",
        nargs="+",
        default=[1, 3],
        help="Iteration numbers of cleora for clicked products",
    )
    parser.add_argument(
        "--cleora-iterations_viewed",
        nargs="+",
        default=[12, 15],
        help="Iteration numbers of cleora for viewed products",
    )
    parser.add_argument(
        "--cleora-columns",
        type=str,
        default='"complex::reflexive::product_id"',
        help="Modifiers and names of columns for Cleora",
    )
    return parser


def compute_codes(embeddings: np.ndarray, n_sketches: int, sketch_dim: int):
    """
    Compute LSH codes
    """
    vcoder = DLSH(n_sketches, sketch_dim)
    vcoder.fit(embeddings)
    codes = vcoder.transform_to_absolute_codes(embeddings)
    return codes


def merge_modalities(
    all_products: List[str], modalities: List[dict], offsets: List[int]
):
    product2codes = {}
    for product in all_products:
        codes = []
        for i, modality in enumerate(modalities):
            codes.append(modality[product] + offsets[i])
        product2codes[str(product)] = list(np.concatenate(codes))
    return product2codes


def save_pickle(dir_path, fname, data):
    with open(dir_path / fname, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_and_save_codes(
    params, cleora_iterations, cleora_input_filename, emb_name, save_codes_fname
):

    modalities = []
    for iter_ in cleora_iterations:
        log.info(f"Run Cleora on {cleora_input_filename} with iteration {iter_}")
        emb_name = emb_name + f"_iter_{iter_}"
        ids, embeddings = run_cleora_directed(
            params.data_dir,
            cleora_input_filename,
            params.cleora_dim,
            iter_,
            columns=params.cleora_columns,
            emb_name=emb_name,
        )
        log.info("Computing LSH codes")
        codes = compute_codes(embeddings, params.n_sketches, params.sketch_dim)
        modalities.append(dict(zip(ids, codes)))

    log.info("Generate random sketch codes")
    random_embeddings = np.random.normal(0, 0.1, size=[len(ids), params.cleora_dim])
    random_codes = compute_codes(
        random_embeddings,
        n_sketches=params.n_sketches_random,
        sketch_dim=params.sketch_dim,
    )
    modalities.append(dict(zip(ids, random_codes)))
    product2codes = merge_modalities(
        ids,
        modalities,
        offsets=[
            i * params.n_sketches * params.sketch_dim for i in range(len(modalities))
        ],
    )

    save_pickle(Path(params.data_dir), save_codes_fname, product2codes)


def main(params):
    params.data_dir = Path(params.data_dir)
    log.info("Running for clicked products...")
    get_and_save_codes(
        params,
        cleora_iterations=params.cleora_iterations_clicked,
        cleora_input_filename=params.data_dir / params.clicked,
        emb_name="emb_clicked",
        save_codes_fname="codes_clicked",
    )
    log.info("Running for viewed products...")
    get_and_save_codes(
        params,
        cleora_iterations=params.cleora_iterations_viewed,
        cleora_input_filename=params.data_dir / params.viewed,
        emb_name="emb_viewed",
        save_codes_fname="codes_viewed",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    main(params)
