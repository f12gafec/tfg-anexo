from ..classification.mlpclassifier import MLPClassifier


class MLPTriangularClassifier(MLPClassifier):
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
        loss_alpha2=0.05,
        loss_eta=1.0,
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
        )

        self.loss_alpha2 = loss_alpha2
        self.loss_eta = loss_eta

    def get_loss(self):
        from dlordinal.losses import TriangularCrossEntropyLoss

        return TriangularCrossEntropyLoss(
            num_classes=self.num_classes,
            alpha2=self.loss_alpha2,
            eta=self.loss_eta,
            weight=self.computed_weights_,
        )
