import warnings

warnings.simplefilter("ignore", UserWarning)
import argparse
import os
import time
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub


import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import BERT_model.py as model
import BERT_preprocess.py as prep


def set_logger(log_path):
    """
    function; to set a logger to log model learning informations
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        default=None,
        type=str,
        required=True,
        help="load training parameters for BERT model",
    )

    params = Params(args.params)

    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    tf.random.set_seed(params.SEED)
    USE_GPU = True
    if USE_GPU:
        device = "/device:GPU:0"
    else:
        device = "/cpu:0"
    # Set Logger
    set_logger(os.path.join(".", params.LOG_DIR + params.NAME + ".log"))

    # Initialize session
    # tf.compat.v1.disable_v2_behavior()
    # tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()

    train_text, train_label, test_text, test_label = prep.read_data(
        params.DATA_PATI, params.TRAIN_RATIO, params.SEED
    )

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module()

    # Convert data to InputExample format
    train_examples = prep.convert_text_to_examples(train_text, train_label)
    test_examples = prep.convert_text_to_examples(test_text, test_label)

    # Convert to features
    (
        train_input_ids,
        train_input_masks,
        train_segment_ids,
        train_labels,
    ) = prep.convert_examples_to_features(
        tokenizer, train_examples, max_seq_length=params.MAX_SEQ_LENGTH
    )
    (
        test_input_ids,
        test_input_masks,
        test_segment_ids,
        test_labels,
    ) = prep.convert_examples_to_features(
        tokenizer, test_examples, max_seq_length=params.MAX_SEQ_LENGTH
    )

    # Instantiate model
    model = model.build_model(
        params.MAX_SEQ_LENGTH, params.NUM_CLASSES, params.N_FINETUNE_LAYERS
    )
    # Instantiate variables
    initialize_vars(sess)

    # Setting up call backs for tf.keras training iterator
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=params.PATIENCE, monitor="val_loss"),
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(
            log_dir=params.LOG_DIR + params.NAME, update_freq="batch"
        ),
        # Save Model Checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            params.SAVE_PATH + params.NAME + ".cpk",
            monitor="val_loss",
            save_freq=params.SAVE_FREQ,
        ),
    ]

    # Running tf.keras training iterator
    history = model.fit(
        [train_input_ids, train_input_masks, train_segment_ids],
        train_labels,
        validation_data=(
            [test_input_ids, test_input_masks, test_segment_ids],
            test_labels,
        ),
        epochs=params.EPOCHS,
        batch_size=params.batch_size,
        callbacks=callbacks,
    )

    logger.info(
        "-->Accuracy: {0:.4f}, AUC: {1:.4f} Precision: {2:.4f}, Recall: {3:.4f}, ".format(
            history.history.Accuracy,
            history.history.AUC,
            history.history.Precision,
            history.history.Recall,
        )
    )
