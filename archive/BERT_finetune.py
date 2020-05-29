import warnings

warnings.simplefilter("ignore", UserWarning)
import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import pdb
from utils.utils import Params

import tensorflow as tf
import tensorflow_hub as hub


import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import BERT_model
import BERT_preprocess as prep


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
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="load training parameters for BERT model",
    )
    args = parser.parse_args()

    params = Params(args.params)

    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("XLA_CPU"))
    )
    tf.random.set_random_seed(params.SEED)
    USE_GPU = True
    if USE_GPU:
        device = '/device:XLA_CPU:0'
    else:
        device = "/CPU:0"
    # Set Logger
    if not os.path.exists(params.LOG_DIR):
        os.makedirs(params.LOG_DIR)
    set_logger(os.path.join(".", params.LOG_DIR + params.NAME + ".log"))
    
    
    # Initialize session
    # tf.compat.v1.disable_v2_behavior()
    # tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()

    train_text, train_label, test_text, test_label = prep.read_dataset(
        params.DATA_PATH, params.TRAIN_RATIO, params.SEED, params.MAX_SEQ_LENGTH, params.N_SAMPLES, params.LABEL
    )

    # Instantiate tokenizer
    tokenizer = prep.create_tokenizer_from_hub_module(sess)
    
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
    
    # Convert Labels to OneHot Vectors
    if params.NUM_CLASSES>2:
        print("Applying One-Hot Encoding")
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=params.NUM_CLASSES)
        test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=params.NUM_CLASSES)

    # Instantiate model
    model = BERT_model.build_model(params.MAX_SEQ_LENGTH, params.NUM_CLASSES, params.N_FINETUNE_LAYERS)
    # Instantiate variables
    BERT_model.initialize_vars(sess)
    
    # Setting up call backs for tf.keras training iterator
    if not os.path.exists(params.SAVE_PATH):
        os.makedirs(params.SAVE_PATH)
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
        batch_size=params.BATCH_SIZE,
        callbacks=callbacks,
    )

    logging.info(
    "-->Accuracy: {0:.4f} //  AUC: {1:.4f} // Precision: {2:4f} // Recall: {3:.4f}".format(
        max(history.history["Accuracy"]),
        max(history.history["AUC"]),
        max(history.history["Precision"]),
        max(history.history["Recall"]),
        )
    )
    logging.info("-----> Elapsed time: {}".format(time.time() - start_time))


