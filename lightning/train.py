import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from dataset import MNISTDataModule
from model import NN
from callbacks import MyPrintingCallback, EarlyStopping
import config

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    logger = TensorBoardLogger(save_dir="tb_logs", name="mnist_model_v0")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )

    # Initialize Model
    model = NN(
        input_size=config.INPUT_SIZE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE,
    )

    # Data Module
    dm = MNISTDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # Train Network
    trainer = pl.Trainer(
        strategy="ddp",
        profiler=profiler,
        logger=logger,
        accelerator=config.ACCELERATOR,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
