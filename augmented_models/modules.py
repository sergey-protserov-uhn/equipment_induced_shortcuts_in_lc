import torch as t
from torch import nn
import lightning as L


class SemanticSegmentationModule(L.LightningModule):
    def __init__(
        self,
        model_class,
        model_args,
        model_kwargs,
        initial_lr,
        max_lr,
        min_lr,
        weight_decay,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.model = model_class(*model_args, **model_kwargs)
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay

    def setup(self, stage):
        if stage == "fit":
            dm = self.trainer.datamodule
            train_class_freqs = dm.train_dataset.class_frequencies
            if not t.all(train_class_freqs):
                raise RuntimeError(f"{train_class_freqs=} has 0-valued elements")
            class_weights = 1 / train_class_freqs
            class_weights = class_weights / class_weights.sum()
            self.register_buffer("class_weights", class_weights)
            print(f"{self.class_weights=}")
            self.loss_fun = t.nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise NotImplementedError(f"Code for {stage=} is not implemented")

    def training_step(self, batch, batch_idx):
        _, images, masks = batch
        outputs = self.model(images)
        if (not isinstance(outputs, t.Tensor)) and hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        loss = self.loss_fun(logits, masks)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        sanity_check = self.trainer.state.stage == "sanity_check"
        _, images, masks = batch
        outputs = self.model(images)
        if (not isinstance(outputs, t.Tensor)) and hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        loss = self.loss_fun(logits, masks)
        if not sanity_check:
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        # NOTE: Apparently, no parameters of batch or layer normalization layers
        # should be weight-decayed, and bias terms in linear or convolutional
        # layers shouldn't either.
        wd_params = []
        no_wd_params = []
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                    no_wd_params.append(param)
                elif isinstance(module, (nn.Linear, nn.Conv2d)):
                    if param_name == "weight":
                        wd_params.append(param)
                    elif param_name == "bias":
                        no_wd_params.append(param)
                    else:
                        raise RuntimeError(
                            f"Don't know if weight decay should be enabled for {param_name=} of {module_name=}"
                        )
                else:
                    raise RuntimeError(
                        f"Don't know if weight decay should be enabled for {param_name=} of {module_name=}"
                    )
        # TODO: Check out the official model architecture implementation for
        # any other tricks, such as, e.g., different LR values for some parameter
        # groups...?
        optimizer = t.optim.AdamW(
            [{"params": wd_params}, {"params": no_wd_params, "weight_decay": 0.0}],
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = t.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.max_lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            div_factor=self.max_lr / self.initial_lr,
            final_div_factor=self.initial_lr / self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
