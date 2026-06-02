# Single source of truth for configs (paths, columns, model params)

# Folders
MODELS_FOLDER = "session_4/models"
DATASETS_FOLDER = "session_4/datasets"

# Dataset file name
DATA_FILE = "Churn_Modelling_train_test.csv"

# Target column (what we want to predict: 1 = customer left, 0 = stayed)
TARGET_COLUMN = "Exited"

# Base name for the saved model -> final file: class_model-Alejandro_Ospina-{timestamp}.joblib
MODEL_NAME = "class_model-Alejandro_Ospina"

# Identifier columns that carry no predictive value
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]

# Columns with two categories that we map to 0/1
BINARY_FEATURES = ["Gender"]

# Categorical columns to one-hot encode
ONE_HOT_ENCODE_COLUMNS = ["Geography"]

# Decision Tree parameters
MODEL_PARAMS = {
    "max_depth": 5,
    "random_state": 8888,
}
