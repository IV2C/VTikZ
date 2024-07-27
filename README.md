# VarBench


## Dataset

The dataset is created from the scripts situated in varbench/dataset_workflow

- Each folder in the dataset it a split
- Each split contains a list of entries in the dataset in the form of a folder
- These folders contain a folder input and a folder reference (the folder input is a git repository so that it can be referenced in the dataset)
- Each entry in the dataset has a prompt and a patch(which is the difference between the reference and the input)