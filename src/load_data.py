import argparse
import pandas as pd

## Load data from data directory and adds column names. Remove "col_names" code if data has column names

def load_data(path):
    col_names = ["id", "clump", "unif_size", "unif_shape", "adhesion",
                 "epi_size", "nuclei", "chromatin", "nucleoli",
                 "mitoses", "class"]
    dataset = pd.read_csv(path, names=col_names, sep=",")
    dataset.to_csv('../data/raw/breast_cancer_loaded.csv') #deleted return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load data")
    parser.add_argument("src_path", type=str, help="Path to data source") #need two arguments
    args = parser.parse_args()
    load_data(args.path)

