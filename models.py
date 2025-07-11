
import torch.nn as nn


class DNNFlex(nn.Module):
    def __init__(self, input_dim, nodes_per_layer, dropout_rate=0.0, tag=""):
        super().__init__()

        self.input_dim = input_dim
        self.nodes_per_layer = nodes_per_layer
        self.dropout_rate = dropout_rate
        self.tag = tag

        hidden_part  = "-".join(str(n) for n in self.nodes_per_layer)
        dropout_part = f"dr{self.dropout_rate:g}"
        str_tag = f"{self.tag}_" if self.tag else ""
        self.model_string = f"{str_tag}in{self.input_dim}__{hidden_part}__{dropout_part}"

        layers = []
        prev_dim = input_dim

        for hidden_dim in nodes_per_layer:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def get_model_string(self):
        return self.model_string

    def override_model_string(self, new_model_string):
        self.model_string = new_model_string



### Some hard-coded models

# # --- Define Model ---
# class RegressionDNN(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         return self.model(x)

# # --- Define Model ---
# class RegressionDNN4L(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(512, 64),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         return self.model(x)



class RegressionLinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x)