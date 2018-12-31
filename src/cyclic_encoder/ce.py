import pandas as pd
import numpy as np


test_df = pd.DataFrame({
    'dates': pd.date_range(
        start='1987-02-19',
        end='1987-02-21',
        freq='H')
})


def get_decomposition(df):
    assert isinstance(df, pd.DataFrame)
    df['hour'] = df['dates'].dt.hour
    return df


def get_cyclicl_encoding(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


if __name__ == "__main__":
    decomposed_df = get_decomposition(test_df)
    encoded_df = get_cyclicl_encoding(decomposed_df, 'hour', 23)
