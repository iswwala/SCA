FIXED_PATH = "data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
VARIABLE_PATH = "data/raw/ASCAD_variable/ascad-variable.h5"

TRAIN_SOURCE_NUM = 50000
TRAIN_TARGET_NUM = 50000
TEST_NUM = 5000

BATCH_SIZE = 128
EPOCHS = 10

NUM_CLASSES = 256
LAMBDA_GRL = 0.3
DOMAIN_LOSS_WEIGHT = 0.1

MODEL_SAVE_PATH = "outputs/checkpoints/cdan_fixed_to_variable.h5"
PRED_SAVE_PATH = "outputs/predictions/cdan_variable_predictions.npy"

MEAN_SAVE_PATH = "outputs/normalization/mean.npy"
STD_SAVE_PATH  = "outputs/normalization/std.npy"

HISTORY_SAVE_PATH = "outputs/logs/history.npy"