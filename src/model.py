import torch
import torch.nn as nn
import torch.nn.functional as F

SMALL_EMBEDDING_SIZE = 20
BIG_EMBEDDING_SIZE = 120


def trunc_normal_(x, mean=0.0, std=1.0):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"

    def __init__(self, ni, nf, std=0.01):
        super().__init__(ni, nf)
        trunc_normal_(self.weight.data, std=std)


class Model(nn.Module):
    def __init__(
        self,
        n_sketches_all,
        sketch_dim,
        num_count_sketches_input,
        hidden_size,
        n_page_type,
        n_previous_page_type,
        n_device_category,
        n_device_platform,
        n_user_tier,
        n_user_country,
        n_context_type,
        n_context_value,
        n_sketches_output,
    ):
        super().__init__()

        input_dim = (
            n_sketches_all * sketch_dim * num_count_sketches_input
            + SMALL_EMBEDDING_SIZE * 7
            + BIG_EMBEDDING_SIZE * 1
        )

        ### Event related ###
        self.page_type = Embedding(n_page_type, SMALL_EMBEDDING_SIZE)
        self.previous_page_type = Embedding(n_previous_page_type, SMALL_EMBEDDING_SIZE)
        self.device_category = Embedding(n_device_category, SMALL_EMBEDDING_SIZE)
        self.device_platform = Embedding(n_device_platform, SMALL_EMBEDDING_SIZE)
        self.user_tier = Embedding(n_user_tier, SMALL_EMBEDDING_SIZE)
        self.user_country = Embedding(n_user_country, SMALL_EMBEDDING_SIZE)
        self.contex_type = Embedding(n_context_type, SMALL_EMBEDDING_SIZE)
        self.context_value = Embedding(n_context_value, BIG_EMBEDDING_SIZE)

        self.n_sketches_all = n_sketches_all
        self.output_dim = n_sketches_all * sketch_dim * n_sketches_output
        self.sketch_dim = sketch_dim
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l_output = nn.Linear(hidden_size, self.output_dim)
        self.projection = nn.Linear(input_dim, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        history_sketches,
        session_sketches,
        page_type,
        previous_page_type,
        device_category,
        device_platform,
        user_tier,
        user_country,
        contex_type,
        context_value,
        history_sketches_viewed,
        query_viewed_sketches,
    ):
        """
        Feed forward network with residual connections.
        """
        x_input = torch.cat(
            (
                history_sketches.float(),
                session_sketches.float(),
                self.page_type(page_type),
                self.previous_page_type(previous_page_type),
                self.device_category(device_category),
                self.device_platform(device_platform),
                self.user_tier(user_tier),
                self.user_country(user_country),
                self.contex_type(contex_type),
                self.context_value(context_value),
                history_sketches_viewed.float(),
                query_viewed_sketches.float(),
            ),
            axis=-1,
        )
        x_proj = self.projection(x_input)
        x_ = self.bn1(F.leaky_relu(self.l1(x_input)))
        x = self.bn2(F.leaky_relu(self.l2(x_) + x_proj))
        x = self.bn3(F.leaky_relu(self.l3(x) + x_proj))
        x = self.l_output(self.bn4(F.leaky_relu(self.l4(x) + x_)))
        x = F.softmax(x.view(-1, self.n_sketches_all, self.sketch_dim), dim=2).view(
            -1, self.output_dim
        )
        return x
