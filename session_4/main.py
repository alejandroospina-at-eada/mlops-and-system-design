from src.source import load_data
from src.transform import Transformer
from src.train import train_model
from src.store import store_model
from metadata import DATA_FILE, MODEL_NAME, TARGET_COLUMN


def main():
    df = load_data(file_name=DATA_FILE)
    df = Transformer().transform(df)
    model = train_model(df=df, target_column=TARGET_COLUMN)
    store_model(model=model, model_name=MODEL_NAME)


# This allows running this code only when main.py is executed directly.
# It won't be executed when importing it.
if __name__ == "__main__":
    main()
