import lightning as L
import torch as t
from torch import nn

from .utils import should_have_progress_bar


class SemanticSegmentationModule(L.LightningModule):
    def __init__(
        self,
        *,
        model_class,
        model_args,
        model_kwargs,
        initial_lr,
        max_lr,
        min_lr,
        weight_decay,
        label_smoothing_alpha,
        metric_class,
        metric_args,
        metric_kwargs,
        metric_name,
    ):
        super().__init__()

        self.save_hyperparameters(logger=True)

        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.label_smoothing_alpha = label_smoothing_alpha
        self.metric_class = metric_class
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs
        self.metric_name = metric_name

        self.model = model_class(*model_args, **model_kwargs)

    def setup(self, stage):
        if stage == "fit":
            dm = self.trainer.datamodule
            class_weights = 1 / dm.train_class_freqs
            class_weights /= class_weights.sum(dim=-1, keepdim=True)
            self.loss_module = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.label_smoothing_alpha,
            )
            self.train_metric_module = self.metric_class(
                *self.metric_args, **self.metric_kwargs
            )
            self.val_metric_module = self.metric_class(
                *self.metric_args, **self.metric_kwargs
            )
        elif stage == "predict":
            pass
        else:
            msg = f"{stage=}"
            raise NotImplementedError(msg)

    def training_step(self, batch, _):
        _, inputs, masks = batch
        logits = self.model(inputs)
        loss_val = self.loss_module(logits, masks)
        metric_module = self.train_metric_module
        metric_module.reset()
        metric_module.update(logits, masks)
        metric_val = metric_module.compute()
        self.log(
            f"train_{self.metric_name}",
            metric_val,
            on_step=False,
            on_epoch=True,
            prog_bar=should_have_progress_bar(),
            logger=True,
        )
        return loss_val

    def validation_step(self, batch, _):
        sanity_check = self.trainer.state.stage == "sanity_check"
        _, inputs, masks = batch
        logits = self.model(inputs)
        metric_module = self.val_metric_module
        metric_module.reset()
        metric_module.update(logits, masks)
        metric_val = metric_module.compute()
        if not sanity_check:
            self.log(
                f"val_{self.metric_name}",
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=should_have_progress_bar(),
                logger=True,
            )

    def configure_optimizers(self):
        wd_params = []
        no_wd_params = []
        for (
            module_name,
            module,
        ) in self.model.named_modules():
            for (
                param_name,
                param,
            ) in module.named_parameters(recurse=False):
                if isinstance(
                    module,
                    (
                        nn.LayerNorm,
                        nn.BatchNorm2d,
                    ),
                ):
                    no_wd_params.append(param)
                elif isinstance(
                    module,
                    (
                        nn.Linear,
                        nn.Conv2d,
                        nn.ConvTranspose2d,
                    ),
                ):
                    if param_name == "weight":
                        wd_params.append(param)
                    elif param_name == "bias":
                        no_wd_params.append(param)
                    else:
                        msg = f"Don't know whether weight decay should be enabled for {param_name=} of {module_name=} with {type(module)=}"
                        raise NotImplementedError(
                            msg,
                        )
                else:
                    msg = f"Don't know whether weight decay should be enabled for {param_name=} of {module_name=} with {type(module)=}"
                    raise NotImplementedError(
                        msg,
                    )
        optimizer = t.optim.AdamW(
            [
                {"params": wd_params},
                {
                    "params": no_wd_params,
                    "weight_decay": 0.0,
                },
            ],
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = t.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.max_lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(
                self.trainer.datamodule.train_dataloader()
            ),
            div_factor=self.max_lr / self.initial_lr,
            final_div_factor=self.initial_lr / self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
