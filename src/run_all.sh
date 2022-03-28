# load data
python load_data.py "../data/raw/breast_cancer.txt" "../data/breast_cancer_loaded.csv"

# clean data
python clean_data.py "../data/breast_cancer_loaded.csv" "../data/train_df.csv"  "../data/test_df.csv"

# plot EDA
python EDA_plots.py "../data/processed/train_df.csv"

# build and test models
python build_test_model.py "../data/processed/train_df.csv" "../data/processed/test_df.csv"

# render report
jupyter-book build "../report/
