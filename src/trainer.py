import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def flatten(t):
    return [item for sublist in t for item in sublist]


def decode(sketch, product_decoder_codes):
    return torch.exp(torch.mm(torch.log(1e-9 + sketch), product_decoder_codes))


def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()


class FarfetchTrainer(pl.LightningModule):
    def __init__(
        self, model, learning_rate, sketch_dim, all_products_viewed, abs_codes_th_viewed
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.sketch_dim = sketch_dim

        self.register_buffer("abs_codes_th_viewed", abs_codes_th_viewed)
        self.all_products_viewed = all_products_viewed
        self.product2id_viewed = {
            pid: idx for idx, pid in enumerate(all_products_viewed)
        }

    def forward(self, batch):
        return self.model(
            batch["history_sketches"],
            batch["session_sketches"],
            batch["page_type"],
            batch["previous_page_type"],
            batch["device_category"],
            batch["device_platform"],
            batch["user_tier"],
            batch["user_country"],
            batch["context_type"],
            batch["context_value"],
            batch["history_sketches_viewed"],
            batch["query_viewed_sketches"],
        )

    def training_step(self, batch, batch_idx):
        output = self(batch)

        loss = categorical_cross_entropy(
            output.view(-1, self.sketch_dim).float(),
            batch["target_sketch_viewed"].float().view(-1, self.sketch_dim).float(),
        )

        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = categorical_cross_entropy(
            output.view(-1, self.sketch_dim),
            batch["target_sketch_viewed"].view(-1, self.sketch_dim),
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)

        bs = output.size(0)
        target_clicked_products = batch["product_id"]

        ### VIEWED
        items_n = self.abs_codes_th_viewed.size(0)
        sketch_depth = self.abs_codes_th_viewed.size(1)
        scores = torch.zeros([bs, items_n], device=output.device)
        sketch_log = output.log()
        for i in range(sketch_depth):
            scores += sketch_log[:, self.abs_codes_th_viewed[:, i]]
        scores /= sketch_depth
        scores = scores.exp()
        op_geom = scores.cpu().numpy()

        target_viewed_products_inds = [
            self.product2id_viewed.get(pid, -1) for pid in target_clicked_products
        ]
        top_product_preds_viewed = op_geom[
            range(op_geom.shape[0]), target_viewed_products_inds
        ]

        out_dict = {
            "pred": top_product_preds_viewed,
            "target_clicked_products": target_clicked_products,
            "query_id": batch["query_id"],
            "is_click": batch["is_click"].cpu().numpy(),
        }
        return {"loss": loss, "out_dict": out_dict}

    def validation_epoch_end(self, out):
        preds = np.concatenate([i["out_dict"]["pred"] for i in out])
        target_clicked_products = np.concatenate(
            [i["out_dict"]["target_clicked_products"] for i in out]
        )
        query_ids = np.concatenate([i["out_dict"]["query_id"] for i in out])
        is_clicks = np.concatenate([i["out_dict"]["is_click"] for i in out])

        from collections import defaultdict

        session_datapoints = defaultdict(list)
        for idx, (query_id, is_click) in enumerate(zip(query_ids, is_clicks)):
            if query_id not in session_datapoints:
                session_datapoints[query_id] = defaultdict(list)
                session_datapoints[query_id]["sum_click"] = 0

            session_datapoints[query_id]["ids"].append(idx)
            session_datapoints[query_id]["is_click"].append(is_click)
            if is_click:
                session_datapoints[query_id]["sum_click"] += 1

        ranks = np.zeros(len(session_datapoints))
        for ind, (query_id, sdata) in enumerate(tqdm(session_datapoints.items())):
            pred = preds[sdata["ids"]].argsort()[::-1]
            num_take = sdata["sum_click"]
            hits_true = np.asarray(sdata["is_click"]).argsort()[::-1][:num_take]
            rank = np.in1d(pred, hits_true).argmax()

            ranks[ind] = rank

        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        mrr = np.mean(np.reciprocal(ranks + 1))

        log_dict = {
            "r1": r1,
            "medr": medr,
            "meanr": meanr,
            "mrr": mrr,
        }
        log.info(log_dict)
        self.trainer.logger_connector.callback_metrics.update(log_dict)
        self.trainer.logger.log_metrics(log_dict, step=self.trainer.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def test_step(self, batch, batch_idx):
        output = self(batch)

        bs = output.size(0)
        target_clicked_products = batch["product_id"]

        ### VIEWED
        items_n = self.abs_codes_th_viewed.size(0)
        sketch_depth = self.abs_codes_th_viewed.size(1)
        scores = torch.zeros([bs, items_n], device=output.device)
        sketch_log = output.log()
        for i in range(sketch_depth):
            scores += sketch_log[:, self.abs_codes_th_viewed[:, i]]
        scores /= sketch_depth
        scores = scores.exp()
        op_geom = scores.cpu().numpy()

        target_viewed_products_inds = [
            self.product2id_viewed.get(pid, -1) for pid in target_clicked_products
        ]
        top_product_preds_viewed = op_geom[
            range(op_geom.shape[0]), target_viewed_products_inds
        ]

        out_dict = {
            "pred": top_product_preds_viewed,
            "target_clicked_products": target_clicked_products,
            "query_id": batch["query_id"],
            "is_click": batch["is_click"].cpu().numpy(),
        }
        return out_dict

    def test_epoch_end(self, out) -> None:
        preds = np.concatenate([i["pred"] for i in out])
        target_clicked_products = np.concatenate(
            [i["target_clicked_products"] for i in out]
        )
        query_ids = np.concatenate([i["query_id"] for i in out])

        is_clicks = np.concatenate([i["is_click"] for i in out])

        #### Test part
        query_out = []
        pids_out = []
        ranks_out = []
        scores_out = []
        click_out = []

        session_datapoints = defaultdict(list)
        for idx, (query_id, is_click) in enumerate(zip(query_ids, is_clicks)):
            if query_id not in session_datapoints:
                session_datapoints[query_id] = defaultdict(list)
                session_datapoints[query_id]["sum_click"] = 0
            session_datapoints[query_id]["ids"].append(idx)
            session_datapoints[query_id]["is_click"].append(is_click)
            if is_click:
                session_datapoints[query_id]["sum_click"] += 1

        ranks_score = np.zeros(len(session_datapoints.keys()))
        for ind, (query_id, sdata) in enumerate(tqdm(session_datapoints.items())):
            scores = preds[sdata["ids"]]
            ranks = scores.argsort()[::-1]
            product_ids = target_clicked_products[sdata["ids"]][ranks]
            query = query_ids[sdata["ids"]][ranks]
            ranking = list(range(1, len(query) + 1))

            num_take = sdata["sum_click"]
            hits_true = np.asarray(sdata["is_click"]).argsort()[::-1][:num_take]
            rank = np.in1d(ranks, hits_true).argmax()

            ranks_score[ind] = rank + 1

            assert (
                len(query)
                == len(product_ids)
                == len(ranking)
                == len(scores[ranks])
                == len(np.asarray(sdata["is_click"])[ranks])
            )

            query_out.extend(query)
            pids_out.extend(product_ids)
            ranks_out.extend(ranking)
            scores_out.extend(scores[ranks])
            click_out.extend(np.asarray(sdata["is_click"])[ranks])

        test_results = pd.DataFrame(
            {
                "query_id": query_out,
                "product_id": pids_out,
                "rank": ranks_out,
                "scores": scores_out,
                "is_click": click_out,
            }
        )

        test_results.to_csv("rawDataFrame_output.csv", index=None)
        test_results_save = test_results.drop(columns=["scores", "is_click"])
        test_results_save.to_csv("output.csv", index=None)
