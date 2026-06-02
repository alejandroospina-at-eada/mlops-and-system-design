from src.transform import Transformer
import pandas as pd


def test_map_binary_column_to_int():
    transformer = Transformer()
    df = pd.DataFrame({"Gender": ["Female", "Male", "Male", "Female"]})

    expected_df = pd.DataFrame({"Gender": [0, 1, 1, 0]})

    transformed_df = transformer._map_binary_column_to_int(df)

    pd.testing.assert_frame_equal(transformed_df, expected_df)


def test_one_hot_encoding():
    transformer = Transformer()
    df = pd.DataFrame({"Geography": ["France", "Spain", "Germany", "France"]})

    transformed_df = transformer._one_hot_encoding(df)

    # The original Geography column should be gone
    assert "Geography" not in transformed_df.columns
    # With drop="first", 3 categories -> 2 encoded columns
    assert transformed_df.shape[1] == 2
    # No missing values after encoding
    assert transformed_df.isnull().sum().sum() == 0


def test_transform_drops_id_columns():
    transformer = Transformer()
    df = pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "CustomerId": [100, 200],
            "Surname": ["Smith", "Jones"],
            "Gender": ["Female", "Male"],
            "Geography": ["France", "Spain"],
            "Age": [30, 40],
        }
    )

    transformed_df = transformer.transform(df)

    for col in ["RowNumber", "CustomerId", "Surname", "Geography"]:
        assert col not in transformed_df.columns
    # Gender should now be numeric
    assert transformed_df["Gender"].tolist() == [0, 1]
