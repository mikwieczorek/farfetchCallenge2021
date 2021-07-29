import argparse
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

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


def add_abs_day(dataframe):
    dataframe["abs_day"] = ((dataframe["week"] - 1) * 7) + dataframe["week_day"]
    return dataframe


def write_cleora_input(fpath, data_list):
    with open(fpath, "w") as f:
        for _list in data_list:
            f.write(f"{' '.join(_list)}\n")


def transform_dataframe(
    dataframe,
    query2viewd_clicked,
    session2queryIdsDict=None,
    session2all_queries_cum=None,
    is_train=False,
):
    dataframe = dataframe.sort_values(by=["user_id", "abs_day", "set"]).reset_index(
        drop=True
    )
    if is_train:
        data_qid_in_ses_id = (
            dataframe[["session_id", "query_id"]]
            .groupby("session_id")["query_id"]
            .apply(list)
            .reset_index()
            .rename(columns={"query_id": "query_ids_in_session"})
        )
        session2queryIdsDict = dict(
            zip(data_qid_in_ses_id.session_id, data_qid_in_ses_id.query_ids_in_session)
        )  ## Return if train

        dataframe = dataframe.merge(data_qid_in_ses_id, on="session_id", how="left")
        # Add column query_ids_in_session
        out = []
        for row in dataframe.itertuples():
            out.append(set(row.query_ids_in_session))
        dataframe["query_ids_in_session"] = out

        # Add column non_current_query_ids_in_session
        out = []
        for row in dataframe.itertuples():
            item = set(row.query_ids_in_session)
            item.remove(row.query_id)
            out.append(list(item))
        dataframe["non_current_query_ids_in_session"] = out

        # Add next two columns product_ids_viewed_in_session_upto_now, product_ids_clicked_in_session_upto_now
        clicked_out = []
        viewed_out = []
        for row in dataframe.itertuples():
            item = row.non_current_query_ids_in_session
            clicked = set(
                [
                    x
                    for query_id in item
                    for x in query2viewd_clicked[query_id]["query_clicked_product_ids"]
                ]
            )
            viewed = set(
                [
                    x
                    for query_id in item
                    for x in query2viewd_clicked[query_id]["query_viewed_product_ids"]
                ]
            )
            clicked_out.append(clicked)
            viewed_out.append(viewed)
        dataframe["product_ids_viewed_in_session_upto_now"] = viewed_out
        dataframe["product_ids_clicked_in_session_upto_now"] = clicked_out

        # Sort
        dataframe = dataframe.sort_values(
            by=["user_id", "abs_day", "query_id"]
        ).reset_index(drop=True)
        # Group data and built quasi-sequential data
        data_grouped = (
            dataframe.groupby(by=["user_id", "abs_day", "session_id"])["query_id"]
            .agg(lambda x: np.unique(x).tolist())
            .reset_index()
        )

        ### Count number of sessions per user
        user_counter = Counter(data_grouped["user_id"])
        data_grouped["total_rows"] = data_grouped.apply(
            lambda row: user_counter[row["user_id"]], axis=1
        )

        row_num = []
        counter = 1
        for row in data_grouped.itertuples():
            row_num.append(counter)
            counter += 1
            if counter > row.total_rows:
                counter = 1
        data_grouped["row_num"] = row_num

        data_grouped_previous = (
            data_grouped.groupby(by=["user_id"], as_index=False)["query_id"]
            .shift(+1)
            .rename(columns={"query_id": "all_query_ids_from_previous_sessions"})
        )
        data_grouped = data_grouped.merge(
            data_grouped_previous["all_query_ids_from_previous_sessions"],
            left_index=True,
            right_index=True,
        )
        # Fill NaNs with empty list
        for row in data_grouped.loc[
            data_grouped.all_query_ids_from_previous_sessions.isnull(),
            "all_query_ids_from_previous_sessions",
        ].index:
            data_grouped.at[row, "all_query_ids_from_previous_sessions"] = []

        # Add column all_query_ids_from_previous_sessions_cumulative, so we now sequential history of queries in sessions
        out = []
        for idx, row in enumerate(data_grouped.itertuples()):
            if row.row_num == 1:
                out.append([])
            else:
                item = out[-1] + row.all_query_ids_from_previous_sessions
                out.append(item)
        data_grouped["all_query_ids_from_previous_sessions_cumulative"] = out
        session2all_queries_cum = dict(
            zip(
                data_grouped["session_id"],
                data_grouped["all_query_ids_from_previous_sessions_cumulative"],
            )
        )  ## Return if train

        clicked_out = []
        viewed_out = []
        for row in dataframe.itertuples():
            sess = row.session_id
            all_queries = session2all_queries_cum[sess]
            clicked = set(
                [
                    x
                    for query_id in all_queries
                    for x in query2viewd_clicked[query_id]["query_clicked_product_ids"]
                ]
            )
            viewed = set(
                [
                    x
                    for query_id in all_queries
                    for x in query2viewd_clicked[query_id]["query_viewed_product_ids"]
                ]
            )
            clicked_out.append(clicked)
            viewed_out.append(viewed)
        dataframe["product_ids_viewed_in_previous_sessions"] = viewed_out
        dataframe["product_ids_clicked_in_previous_sessions"] = clicked_out

    else:
        out = []
        for row in dataframe.itertuples():
            item = session2queryIdsDict.get(row.session_id, [])
            item = set(item)
            out.append(item)
        dataframe["query_ids_in_session"] = out

        clicked_out = []
        viewed_out = []
        clicked_in_session = []
        viewed_in_session = []
        for row in dataframe.itertuples():
            sess = row.session_id
            # Whole history until the current session_id
            all_queries = session2all_queries_cum.get(sess, [])

            clicked = set(
                [
                    x
                    for query_id in all_queries
                    for x in query2viewd_clicked.get(
                        query_id, {"query_clicked_product_ids": []}
                    )["query_clicked_product_ids"]
                ]
            )
            viewed = set(
                [
                    x
                    for query_id in all_queries
                    for x in query2viewd_clicked.get(
                        query_id, {"query_viewed_product_ids": []}
                    )["query_viewed_product_ids"]
                ]
            )
            clicked_out.append(clicked)
            viewed_out.append(viewed)

            # Current session other query_ids
            current_sess_queries = session2queryIdsDict.get(sess, [])
            clicked_current = set(
                [
                    x
                    for query_id in current_sess_queries
                    for x in query2viewd_clicked.get(
                        query_id, {"query_clicked_product_ids": []}
                    )["query_clicked_product_ids"]
                ]
            )
            viewed_current = set(
                [
                    x
                    for query_id in current_sess_queries
                    for x in query2viewd_clicked.get(
                        query_id, {"query_viewed_product_ids": []}
                    )["query_viewed_product_ids"]
                ]
            )
            clicked_in_session.append(clicked_current)
            viewed_in_session.append(viewed_current)

        dataframe["product_ids_viewed_in_previous_sessions"] = viewed_out
        dataframe["product_ids_clicked_in_previous_sessions"] = clicked_out
        dataframe["product_ids_viewed_in_session_upto_now"] = viewed_in_session
        dataframe["product_ids_clicked_in_session_upto_now"] = clicked_in_session

    if is_train:
        return dataframe, session2queryIdsDict, session2all_queries_cum
    else:
        return dataframe


def main(params):
    ### Params
    data_dir = Path(params.data_dir)
    filename_viewed = "cleoraInput_sessionIdGrouped_viewed"  ### For all viewed products in all sessions
    cleora_input_filename_viewed = data_dir / filename_viewed
    filename_clicked = "cleoraInput_sessionIdGrouped_onlyClicked"  ### For all clicked products in all sessions
    cleora_input_filename_clicked = data_dir / filename_clicked

    ### DATA
    log.info("Loading data...")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    log.info("Start preparing data...")

    ### Calucalte absolute days
    train_df = add_abs_day(train_df)
    val_df = add_abs_day(val_df)
    test_df = add_abs_day(test_df)

    ### Add set set indicator column
    train_df["set"] = "train"
    val_df["set"] = "val"
    test_df["set"] = "test"

    ### Concat
    data = pd.concat([train_df, val_df, test_df])
    data = data.sort_values(by=["user_id", "abs_day", "set"]).reset_index(drop=True)

    ### We will use both clicked and viewed data
    data_clicked = data.loc[
        (data["is_click"] == 1) & (data["set"] == "train")
    ].reset_index(drop=True)

    ### Get list of all products clicked in the query
    query_product_list_df = (
        data_clicked.groupby(["query_id"])["product_id"]
        .apply(list)
        .reset_index()
        .rename(columns={"product_id": "query_clicked_product_ids"})
    )
    ### Get list of all products clicked/viewed in the given session
    # CLICKED
    session_product_list_df = (
        data_clicked.groupby(["session_id"])["product_id"]
        .apply(list)
        .reset_index()
        .rename(columns={"product_id": "session_clicked_product_ids"})
    )
    # VIEWED
    session_product_list_df_viewed = (
        data.groupby(["session_id"])["product_id"]
        .apply(list)
        .reset_index()
        .rename(columns={"product_id": "session_viewed_product_ids"})
    )

    ### Prepare input file to cleora
    # CLICKED products in sessions
    log.info("Prepare and write data for Cleora...")
    session_pids_list = session_product_list_df["session_clicked_product_ids"].to_list()
    write_cleora_input(cleora_input_filename_clicked, session_pids_list)
    # VIEWED products in sessions
    session_pids_list_viewed = session_product_list_df_viewed[
        "session_viewed_product_ids"
    ].to_list()
    write_cleora_input(cleora_input_filename_viewed, session_pids_list_viewed)

    ### Merge to data DataFrame
    # Merge clicked products in query
    data = data.merge(query_product_list_df, on="query_id", how="left")
    # Viewed in query
    data = data.merge(
        data.groupby("query_id")["product_id"]
        .apply(list)
        .reset_index()
        .rename(columns={"product_id": "query_viewed_product_ids"}),
        on="query_id",
        how="left",
    )
    ### Create dict 'query_id': {'query_clicked':[], 'query_viewed':[]}
    query2viewd_clicked = (
        data[["query_id", "query_viewed_product_ids", "query_clicked_product_ids"]]
        .drop_duplicates(subset="query_id")
        .set_index("query_id", drop=True)
        .to_dict(orient="index")
    )

    ### Transform
    log.info("Start transforming train dataset...")
    data_train, session2queryIdsDict, session2all_queries_cum = transform_dataframe(
        data.loc[data["set"] == "train"], query2viewd_clicked, is_train=True
    )
    log.info("Start transforming val dataset...")
    data_val = transform_dataframe(
        data.loc[data["set"] == "val"],
        query2viewd_clicked,
        session2queryIdsDict=session2queryIdsDict,
        session2all_queries_cum=session2all_queries_cum,
    )
    log.info("Start transforming test dataset...")
    data_test = transform_dataframe(
        data.loc[data["set"] == "test"],
        query2viewd_clicked,
        session2queryIdsDict=session2queryIdsDict,
        session2all_queries_cum=session2all_queries_cum,
    )

    data_train.to_csv(data_dir / "train_original_processed_reproducing.csv", index=None)
    data_val.to_csv(data_dir / "val_original_processed_reproducing.csv", index=None)
    data_test.to_csv(data_dir / "test_original_processed_reproducing.csv", index=None)

    ### ATTRIBUTES
    log.info("Transform attributes to dict...")
    attr_df = pd.read_parquet(data_dir / "attributes.parquet")
    for name, value in attr_df[
        attr_df.columns.difference(["product_id", "start_online_date"])
    ].iteritems():
        value = set(value)
        mapping = {k: idx for idx, k in enumerate(value)}
        attr_df[name] = attr_df[name].apply(lambda x: mapping[x])
    products_dict = attr_df.set_index("product_id").to_dict("index")
    with open(data_dir / "products_dict_reproducing", "wb") as f:
        pickle.dump(products_dict, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    main(params)
