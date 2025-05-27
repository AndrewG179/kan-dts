# Model training hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.001

# Dataset configuration
TIMESERIES_LENGTH = 5000
INPUT_LEN = 100
PRED_LEN = 14
NOISE = True
SCALE = 0.3

# KAN model configuration
KAN_WIDTH = [INPUT_LEN, 5, 5, PRED_LEN]  # example: input_dim=114, output_dim=14
KAN_GRID = 5
KAN_OPTIMIZER = 'LBFGS'

# KAN model configuration
MLP_WIDTH = [INPUT_LEN, 128, 128, 128, PRED_LEN]