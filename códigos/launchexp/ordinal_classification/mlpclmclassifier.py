import numpy as np
from dlordinal.output_layers import CLM
from torch import nn

from ..classification.mlpclassifier import MLPClassifier


class MLPCLMModel(nn.Module):
    def __init__(
        self, input_shape, num_classes, link_function="logit", hidden_units=[5]
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = []
        for i, hidden_unit in enumerate(hidden_units):
            self.hidden.append(
                nn.Sequential(
                    nn.Linear(
                        (
                            np.prod(input_shape).astype(int)
                            if i == 0
                            else hidden_units[i - 1]
                        ),
                        hidden_unit,
                    ),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            )
        self.projection = nn.Linear(hidden_units[-1], 1)
        self.clm = CLM(num_classes, link_function)

    def forward(self, x):
        x = self.flatten(x)
        hidden = x
        for layer in self.hidden:
            hidden = layer(hidden)
        projection = self.projection(hidden)
        probas = self.clm(projection)
        return probas


class MLPCLMClassifier(MLPClassifier):
    def __init__(
        self,
        *,
        hidden_units=[5],
        device="cpu",
        max_iter=1000,
        class_weight=None,
        learning_rate=1e-3,
        verbose=0,
        n_jobs=1,
        batch_size=-1,
        link_function="logit",
        weight_decay=0.0
    ):
        super().__init__(
            hidden_units=hidden_units,
            device=device,
            max_iter=max_iter,
            class_weight=class_weight,
            learning_rate=learning_rate,
            verbose=verbose,
            n_jobs=n_jobs,
            batch_size=batch_size,
            weight_decay=weight_decay
        )

        self.link_function = link_function

    def get_model(self, input_shape, num_classes, hidden_units):
        return MLPCLMModel(input_shape, num_classes, self.link_function, hidden_units)
