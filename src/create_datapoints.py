import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where files will be saved",
    )
    return parser


def save_pickle(dir_path, fname, data):
    with open(dir_path / fname, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_csv(fname, load_data_dir, is_train=False):
    columns2assert = [
        "query_clicked_product_ids",
        "product_ids_viewed_in_session_upto_now",
        "product_ids_clicked_in_session_upto_now",
    ]
    if not is_train:
        columns2assert.append("query_clicked_product_ids")

    if is_train:
        dataframe = pd.read_csv(
            load_data_dir / fname,
            converters={
                "product_ids_viewed_in_previous_sessions": eval,
                "product_ids_clicked_in_previous_sessions": eval,
                "product_ids_viewed_in_session_upto_now": eval,
                "product_ids_clicked_in_session_upto_now": eval,
                "query_viewed_product_ids": eval,
                "query_clicked_product_ids": eval,
            },
        )
    else:
        dataframe = pd.read_csv(
            load_data_dir / fname,
            converters={
                "product_ids_viewed_in_previous_sessions": eval,
                "product_ids_clicked_in_previous_sessions": eval,
                "product_ids_viewed_in_session_upto_now": eval,
                "product_ids_clicked_in_session_upto_now": eval,
                "query_viewed_product_ids": eval,
            },
        )

    dataframe["query_viewed_product_ids"] = dataframe["query_viewed_product_ids"].apply(
        set
    )
    for col in columns2assert:
        dataframe[col] = dataframe[col].astype(object)
        for row in dataframe.loc[dataframe[col].isnull(), col].index:
            dataframe.at[row, col] = set([])
        dataframe[col] = dataframe[col].apply(list)

    if is_train:
        dataframe = dataframe.loc[dataframe["is_click"] == 1]

    return dataframe


def preapre_datapoints(dataframe, save_dir_path):
    COLUMNS_TO_MAP = dataframe.columns.to_list()[4:12]

    mapping_dicts = {}
    for name, value in dataframe[COLUMNS_TO_MAP].iteritems():
        value = set(value)
        mapping_dicts[name + "2id"] = {k: idx for idx, k in enumerate(value)}

    training_datapoints = []
    validation_datapoints = []
    test_datapoints = []

    log.info("Start preparing datapoints...")
    for i, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):

        datapoint = {
            "query_id": row["query_id"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "product_id": row["product_id"],
            "page_type": mapping_dicts["page_type2id"][row["page_type"]],
            "previous_page_type2id": mapping_dicts["previous_page_type2id"][
                row["previous_page_type"]
            ],
            "device_category2id": mapping_dicts["device_category2id"][
                row["device_category"]
            ],
            "device_platform2id": mapping_dicts["device_platform2id"][
                row["device_platform"]
            ],
            "user_tier2id": mapping_dicts["user_tier2id"][row["user_tier"]],
            "user_country2id": mapping_dicts["user_country2id"][row["user_country"]],
            "context_type2id": mapping_dicts["context_type2id"][row["context_type"]],
            "context_value2id": mapping_dicts["context_value2id"][row["context_value"]],
            "product_price": row[
                "product_price"
            ],  # Maybe it should be normalized to [0,1]?
            "week": row["week"] - 1,  # So it starts from 0
            "week_day": row["week_day"],
            "is_click": row["is_click"],
            "product_ids_viewed_in_previous_sessions": row[
                "product_ids_viewed_in_previous_sessions"
            ],
            "product_ids_clicked_in_previous_sessions": row[
                "product_ids_clicked_in_previous_sessions"
            ],
            "product_ids_viewed_in_session_upto_now": row[
                "product_ids_viewed_in_session_upto_now"
            ],
            "product_ids_clicked_in_session_upto_now": row[
                "product_ids_clicked_in_session_upto_now"
            ],
            "absolute_day": row["abs_day"],
            "query_clicked_product_ids": row["query_clicked_product_ids"],
            "query_viewed_product_ids": row["query_viewed_product_ids"],
        }

        if row["set"] == "train":
            training_datapoints.append(datapoint)
        elif row["set"] == "val":
            validation_datapoints.append(datapoint)
        elif row["set"] == "test":
            test_datapoints.append(datapoint)
        else:
            raise ValueError
    log.info("Finished preprocessing datapoints...")

    log.info("Saving datapoints...")
    save_pickle(
        save_dir_path, "train_datapoints_sequential_reproducing", training_datapoints
    )
    save_pickle(
        save_dir_path,
        "validation_datapoints_sequential_reproducing",
        validation_datapoints,
    )
    save_pickle(
        save_dir_path, "test_datapoints_sequential_reproducing", test_datapoints
    )


def main(params):
    ### Params
    data_dir = Path(params.data_dir)

    ### DATA
    log.info("Loading data...")
    train_df = load_csv(
        "train_original_processed_reproducing.csv", data_dir, is_train=True
    )
    val_df = load_csv("val_original_processed_reproducing.csv", data_dir)
    test_df = load_csv("test_original_processed_reproducing.csv", data_dir)
    data = pd.concat([train_df, val_df, test_df])
    preapre_datapoints(data, data_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    main(params)
