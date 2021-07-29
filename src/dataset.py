import numpy as np
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset


def multiscale(x, scales):
    return np.hstack([x.reshape(-1, 1) / pow(2.0, i) for i in scales])


def encode_scalar_column(x, scales=[-1, 0, 1, 2, 3, 4, 5, 6]):
    return np.hstack([np.sin(multiscale(x, scales)), np.cos(multiscale(x, scales))])


class FarfetchBaselineDataset(Dataset):
    def __init__(
        self,
        data,
        product2codes,
        products_dict,
        input_dim,
        sketch_dim,
        product2codes_viewed,
        decay_value=1.0,
    ):
        self.data = data
        self.product2codes = product2codes
        self.products_dict = products_dict
        self.decay_value = decay_value
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        self.product2codes_viewed = product2codes_viewed

    def __len__(self):
        return len(self.data)

    def codes_to_sketch(self, codes, size):
        """
        Convert abosulte codes into sketch sparse vector
        """
        x = np.zeros(size)
        for ind in codes:
            x[ind] += 1
        return x

    def __getitem__(self, idx):
        example = self.data[idx]
        product_ids_clicked_in_previous_sessions = example[
            "product_ids_clicked_in_previous_sessions"
        ]
        product_ids_clicked_in_session_upto_now = example[
            "product_ids_clicked_in_session_upto_now"
        ]

        product_ids_viewed_in_session_upto_now = example[
            "product_ids_viewed_in_session_upto_now"
        ]

        query_viewed_product_ids = example["query_viewed_product_ids"]

        product_code = self.product2codes.get(example["product_id"], None)
        if product_code is None:
            product_sketch = np.zeros(self.input_dim)
        else:
            product_sketch = self.codes_to_sketch(product_code, self.input_dim)

        # Build sketch based on history sessions
        history_sketches = np.zeros(self.input_dim)
        for pid in product_ids_clicked_in_previous_sessions:
            pid_codes = self.product2codes.get(pid, None)
            if pid_codes is None:
                continue
            history_sketches *= self.decay_value
            history_sketches += self.codes_to_sketch(pid_codes, self.input_dim)
        history_sketches = normalize(
            history_sketches.reshape(-1, self.sketch_dim), "l2"
        ).reshape((self.input_dim,))

        session_sketches = np.zeros(self.input_dim)
        for pid in product_ids_clicked_in_session_upto_now:
            pid_codes = self.product2codes.get(pid, None)
            if pid_codes is None:
                continue
            session_sketches *= self.decay_value
            session_sketches += self.codes_to_sketch(pid_codes, self.input_dim)
        session_sketches = normalize(
            session_sketches.reshape(-1, self.sketch_dim), "l2"
        ).reshape((self.input_dim,))

        history_sketches_viewed = np.zeros(self.input_dim)
        for pid in product_ids_viewed_in_session_upto_now:
            pid_codes = self.product2codes_viewed.get(pid, None)
            if pid_codes is None:
                continue
            history_sketches_viewed *= self.decay_value
            history_sketches_viewed += self.codes_to_sketch(pid_codes, self.input_dim)
        history_sketches_viewed = normalize(
            history_sketches_viewed.reshape(-1, self.sketch_dim), "l2"
        ).reshape((self.input_dim,))

        query_viewed_sketches = np.zeros(self.input_dim)
        for pid in query_viewed_product_ids:
            pid_codes = self.product2codes_viewed.get(pid, None)
            if pid_codes is None:
                continue
            query_viewed_sketches *= self.decay_value
            query_viewed_sketches += self.codes_to_sketch(pid_codes, self.input_dim)
        query_viewed_sketches = normalize(
            query_viewed_sketches.reshape(-1, self.sketch_dim), "l2"
        ).reshape((self.input_dim,))

        product_code_viewed = self.product2codes_viewed.get(example["product_id"], None)
        if product_code_viewed is None:
            product_sketch_viewed = np.zeros(self.input_dim)
        else:
            product_sketch_viewed = self.codes_to_sketch(
                product_code_viewed, self.input_dim
            )

        result = {
            "history_sketches": history_sketches,
            "session_sketches": session_sketches,
            "target_sketch": product_sketch,
            "page_type": example["page_type"],
            "previous_page_type": example["previous_page_type2id"],
            "device_category": example["device_category2id"],
            "device_platform": example["device_platform2id"],
            "user_tier": example["user_tier2id"],
            "user_country": example["user_country2id"],
            "context_type": example["context_type2id"],
            "context_value": example["context_value2id"],
            "is_click": example["is_click"],
            "query_id": example["query_id"],
            "product_id": example["product_id"],
            "history_sketches_viewed": history_sketches_viewed,
            "query_viewed_sketches": query_viewed_sketches,
            "target_sketch_viewed": product_sketch_viewed,
        }

        return result
