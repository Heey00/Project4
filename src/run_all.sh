# load data
python load_data.py "../data/raw/breast_cancer.txt" "../data/breast_cancer_loaded.csv"

# clean data
python clean_data.py "../data/breast_cancer_loaded.csv" "../data/train_df.csv"  "../data/test_df.csv"

# plot EDA
python EDA_plots.py "../data/train_df.csv" "../results/hist_output.png" "../results/boxplot_output.png"

# build and test models
python build_test_model.py "../data/train_df.csv" "../data/test_df.csv" "../results/"

# render report
jupyter-book build "../report/
