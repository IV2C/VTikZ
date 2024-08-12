# VarBench


## Dataset

The dataset is created from the scripts situated in varbench/dataset_workflow

- Each folder in the dataset it a split.
- Each split contains a list of entries in the dataset in the form of a folder.
- These folders contain a folder input and a folder reference (the folder input is a git repository so that it can be referenced in the dataset).
- Each entry in the dataset has a prompt and a patch as well(which is the difference between the reference and the input).

### Creating a new entry
First run the following command
```sh
python3 -m varbench.tools.create_entry <SPLIT_NAME>
```
It will create a new entry in the dataset situated at ./dataset  
You will then need to modify the content of the input and the reference, and put the associated prompt file named `prompt.txt`

### Removing an entry in the dataset
Simply run the following command with the name of the entry you want to delete
```sh
python3 -m varbench.tools.remove_entry <NAME>
```
Not that the name must be specified with `<SPLIT_NAME><ENTRY_NAME>`  
For example "tikz_entry5"

### Publishing the dataset
TODO

