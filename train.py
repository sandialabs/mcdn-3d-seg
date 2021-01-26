import fnmatch
import os

import keras.callbacks as callbacks
from keras.callbacks import Callback

from ctseg.config import ex
from ctseg.data_loader import DataLoader
from ctseg.model import get_model, path_to_model_file, compile_model
from ctseg.sampler import Sampler, RandomSampler
from ctseg.utils.model_checkpoint import ModelCheckpoint
from ctseg.utils.query_msg import query_msg


@ex.capture
def log_performance(_run, logs, train_config, normalization):
    models_dir = train_config["outputs"]["models_dir"]
    _run.add_artifact(path_to_model_file("weights.hdf5", models_dir, normalization))
    _run.log_scalar("loss", float(logs.get("loss")))
    _run.log_scalar("accuracy", float(logs.get("acc")))
    _run.log_scalar("val_loss", float(logs.get("val_loss")))
    _run.log_scalar("val_accuracy", float(logs.get("val_acc")))
    _run.result = float(logs.get("val_acc"))


class LogPerformance(Callback):
    def on_epoch_end(self, _, logs=None):
        logs = logs or {}
        log_performance(logs=logs)


@ex.automain
def main(
    model_config,
    data_config,
    train_config,
    test_config,
    num_classes,
    normalization,
    num_gpus,
):
    train_data_dir = train_config["inputs"]["data_dir"]
    train_targets_dir = train_config["inputs"]["targets_dir"]
    train_normalized_data_dir = train_config["outputs"]["normalized_data_dir"]
    csv_log_dir = train_config["outputs"]["csv_log_dir"]
    tensorboard_log_dir = train_config["outputs"]["tensorboard_log_dir"]
    models_dir = train_config["outputs"]["models_dir"]
    preprocessor_dir = train_config["outputs"]["preprocessor_dir"]
    test_data_dir = test_config["inputs"]["data_dir"]
    test_targets_dir = test_config["inputs"]["targets_dir"]
    test_normalized_data_dir = test_config["outputs"]["normalized_data_dir"]

    input_shape = model_config["architecture_config"]["input_shape"]
    resume_from = model_config["load_config"]["resume_from"]
    load_weights_from = model_config["load_config"]["load_weights_from"]

    sampler_config = data_config["sampler_config"]

    compilation_config = train_config["compilation"]
    initial_epoch = train_config["initial_epoch"]
    epochs = train_config["epochs"]
    batch_size = train_config["batch_size"]

    os.makedirs(train_normalized_data_dir, mode=0o775, exist_ok=True)
    os.makedirs(test_normalized_data_dir, mode=0o775, exist_ok=True)
    os.makedirs(csv_log_dir, mode=0o775, exist_ok=True)
    os.makedirs(tensorboard_log_dir, mode=0o775, exist_ok=True)
    os.makedirs(models_dir, mode=0o775, exist_ok=True)
    os.makedirs(preprocessor_dir, mode=0o775, exist_ok=True)

    print("Loading 'train' data...")
    train_dataset = DataLoader(
        data_dir=train_data_dir,
        data_config=data_config,
        num_classes=num_classes,
        sampler=Sampler.from_config(sampler_config, chunk_shape=input_shape),
        targets_dir=train_targets_dir,
        mode="train",
        normalization=normalization,
        normalized_image_dir=train_normalized_data_dir,
        normalizer_metadata_dir=preprocessor_dir,
    )
    print("Loading 'test' data...")
    test_dataset = DataLoader(
        data_dir=test_data_dir,
        data_config=data_config,
        num_classes=num_classes,
        sampler=RandomSampler.from_config(sampler_config, chunk_shape=input_shape),
        targets_dir=test_targets_dir,
        mode="test",
        normalization=normalization,
        normalized_image_dir=test_normalized_data_dir,
        normalizer_metadata_dir=preprocessor_dir,
    )
    print("Data loading complete")
    train_dataset.gen_chunk_list()
    test_dataset.gen_chunk_list()

    model = get_model(model_config, num_classes, train_config, normalization, num_gpus)
    compile_model(model, **compilation_config)

    latest_model_path = path_to_model_file("latest.hdf5", models_dir, normalization)
    best_model_path = path_to_model_file("best.hdf5", models_dir, normalization)
    prev_best_path = path_to_model_file("prev_best.npy", models_dir, normalization)
    if resume_from or load_weights_from:
        if os.path.isfile(best_model_path):
            # If resuming training, save the old best model so we don't overwrite it
            num_old_bests = len(fnmatch.filter(os.listdir(models_dir), "*old*"))
            old_best_model_path = path_to_model_file(
                f"old_best{num_old_bests}.hdf5", models_dir, normalization
            )
            os.rename(best_model_path, old_best_model_path)
    else:
        if os.path.isfile(best_model_path) or os.path.isfile(latest_model_path):
            choice = query_msg(
                "Previous model in output directory found yet not resuming."
                " Are you sure?",
                default="no",
            )
            if not choice:
                return
    # ModelFilename + '-{epoch}-{val_loss:.2f}.hdf5'
    csv_path = os.path.join(csv_log_dir, "training.log")

    best_checkpointer = ModelCheckpoint(
        filepath=best_model_path,
        verbose=1,
        save_best_only=True,
        previous_best=prev_best_path,
    )
    latest_checkpointer = callbacks.ModelCheckpoint(
        filepath=latest_model_path, verbose=1, save_best_only=False
    )
    csv_logger = callbacks.CSVLogger(csv_path)
    tensorboard = callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=False,
        write_images=True,
        embeddings_freq=0,
    )

    model.fit_generator(
        generator=train_dataset.create_generator(batch_size),
        steps_per_epoch=len(train_dataset) // batch_size,
        validation_data=test_dataset.create_generator(batch_size),
        validation_steps=len(test_dataset) // batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[
            best_checkpointer,
            latest_checkpointer,
            csv_logger,
            tensorboard,
            LogPerformance(),
        ],
    )
