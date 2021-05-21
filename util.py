from constants import *
import pandas as pd


def pd_read_csv(input_path, col_list = None):
    if col_list is not None:
        input_df = pd.read_csv(input_path, names=col_list)
    else:
        input_df = pd.read_csv(input_path)
    print_df(input_df)
    return input_df


def print_df(input_df, max_rows=DEFAULT_DF_PRINT_ROWS, name=""):
    if max_rows > 0:
        print(name)
        print(input_df.head(max_rows).to_string())
