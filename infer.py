import logging
import os
import shutil
import time

import numpy as np

from ctseg.config import ex
from ctseg.data_loader import DataLoader
from ctseg.model import load_model, predict_stochastic
from ctseg.sampler import OverlapSampler
import ctseg.utils.alignment_check as align


logger = logging.getLogger(__name__)


def save_output(
    loader, pred_raw, pred_std, pred_argmax, predictions_dir, check_alignment
):
    key = loader.get_key(0)

    output_filename = os.path.join(predictions_dir, "overlapped_pred_" + key + ".npy")
    logger.info(f"saving npy file to {output_filename}")
    np.save(output_filename, pred_argmax.astype(np.int8))

    output_filename = os.path.join(
        predictions_dir, "overlapped_pred_raw_" + key + ".npy"
    )
    logger.info(f"saving npy file to {output_filename}")
    np.save(output_filename, pred_raw)

    output_filename = os.path.join(
        predictions_dir, "overlapped_pred_std_" + key + ".npy"
    )
    logger.info(f"saving npy file to {output_filename}")
    np.save(output_filename, pred_std)

    if check_alignment >= 0:
        x_mean, y_mean, x_diff, y_diff = align.eval_align(pred_argmax, check_alignment)
        align_results = np.array([x_mean, y_mean, x_diff, y_diff])
        align_filename = os.path.join(predictions_dir, "align_check_" + key + ".npy")
        logger.info(f"saving npy file to {align_filename}")
        np.save(align_filename, align_results.astype(np.float16))


@ex.automain
def main(
    model_config,
    data_config,
    train_config,
    inference_config,
    num_classes,
    normalization,
    production_mode,
    check_alignment,
):
    unprocessed_queue_dir = inference_config["inputs"]["unprocessed_queue_dir"]
    normalized_data_dir = inference_config["outputs"]["normalized_data_dir"]
    processed_data_dir = inference_config["outputs"]["processed_data_dir"]
    predictions_dir = inference_config["outputs"]["predictions_dir"]

    preprocessor_dir = train_config["outputs"]["preprocessor_dir"]

    input_shape = model_config["architecture_config"]["input_shape"]
    sampler_config = data_config["sampler_config"]

    os.makedirs(normalized_data_dir, mode=0o775, exist_ok=True)
    os.makedirs(processed_data_dir, mode=0o775, exist_ok=True)
    os.makedirs(predictions_dir, mode=0o775, exist_ok=True)

    model = load_model(
        model_config, train_config, num_classes, normalization, compile=False
    )

    # TODO: avoid exception when no inference images left
    while 1:
        try:
            infer_dataset = DataLoader(
                data_dir=unprocessed_queue_dir,
                data_config=data_config,
                num_classes=num_classes,
                sampler=OverlapSampler.from_config(
                    sampler_config, chunk_shape=input_shape
                ),
                mode="infer",
                normalization=normalization,
                normalized_image_dir=normalized_data_dir,
                normalizer_metadata_dir=preprocessor_dir,
            )
            infer_dataset.gen_chunk_list()

            pred_raw, pred_std = predict_stochastic(
                loader=infer_dataset,
                model=model,
                iters=inference_config["inference_iters"],
            )

            logger.info("Finished predicting volume...")

            pred_argmax = np.argmax(pred_raw, axis=-1)

            u, c = np.unique(pred_argmax, return_counts=True)
            logger.info(f"  Unique: {u}")
            logger.info(f"  Counts: {c}")

            save_output(
                loader=infer_dataset,
                pred_raw=pred_raw,
                pred_std=pred_std,
                pred_argmax=pred_argmax,
                predictions_dir=predictions_dir,
                check_alignment=check_alignment,
            )

            key = infer_dataset.get_key(0)
            logger.info(f"Moving {key} to complete...")
            path = infer_dataset.get_raw_image_path(key, ".npy")
            shutil.move(path, processed_data_dir)
        except FileNotFoundError:
            if production_mode:
                logger.info("Waiting for input file")
                time.sleep(60)
            else:
                logger.info("No files found in infer directory")
                break
