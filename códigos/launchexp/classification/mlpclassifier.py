import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class MLPModel(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_units=[5]):
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
            
        self.classification = nn.Linear(hidden_units[-1], num_classes)
        # self.output = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.flatten(x)
        hidden = x
        for layer in self.hidden:
            hidden = layer(hidden)
        logits = self.classification(hidden)
        # logits = self.output(classification)
        return logits


class MLPClassifier(BaseEstimator, ClassifierMixin):
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
        weight_decay=0.0
    ):

        if not isinstance(hidden_units, (list, tuple)):
            raise ValueError(
                "hidden_units must be a list that contain the number of units"
                " in each hidden layer"
            )

        self.hidden_units = hidden_units
        self.device = device
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.initialized = False
        self.weight_decay = weight_decay

    # Añade esto dentro de la clase MLPClassifier en mlpclassifier.py

    def set_params(self, **params):
        # Si el parámetro empieza por optimizer__, se lo guardamos para la red
        for key, value in params.items():
            setattr(self, key, value)
        return super().set_params(**params)

    def _initialize(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.input_shape = X.shape[1:]

        model = self.get_model(self.input_shape, self.num_classes, self.hidden_units)

        self.computed_weights_ = torch.tensor(
            compute_class_weight(
                self.class_weight,
                classes=np.arange(self.num_classes),
                y=y,
            ),
            dtype=torch.float32,
        )

        self.net_ = NeuralNetClassifier(
            module=model,
            criterion=self.get_loss(),  # type: ignore
            optimizer=Adam,
            lr=self.learning_rate,
            max_epochs=self.max_iter,
            optimizer__weight_decay=self.weight_decay,
            callbacks=[EarlyStopping(
                monitor = 'valid_loss',
                patience=10,
                lower_is_better=True
            )],
            device=self.device,
            verbose=self.verbose,
            batch_size=self.batch_size,
        )

        self.initialized = True

    def get_loss(self):
        return CrossEntropyLoss(weight=self.computed_weights_)

    def get_model(self, input_shape, num_classes, hidden_units):
        return MLPModel(input_shape, num_classes, hidden_units).to(self.device)

    def fit(self, X, y, **fit_params):
        if not self.initialized:
            self._initialize(X, y)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return self.net_.fit(X, y, **fit_params)

        epochs_run=len(self.net_.history)
        valid_losses=self.net_.history[:, 'valid_loss']
        best_epoch=np.argmin(valid_losses)+1
        if self.verbose >0:
            print(f"Entrenamiento finalizado en la iteración: {epochs_run}")
            print(f"El modelo dejó de aprender en la iteración: {best_epoch} (Pérdida: {valid_losses[best_epoch-1]:.4f})")
            return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.net_.predict(X)

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.net_.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return self.net_.score(X, y, sample_weight)
