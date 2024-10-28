# VarBench

## Evaluation

### Running the evaluation
The evaluation can be run for any subset using

```sh
python3 -m varbench.run_evaluation [-h] --subsets SUBSETS [SUBSETS ...] --model_type MODEL_TYPE --model MODEL [--gpu_number GPU_NUMBER] [--api_url API_URL] [--api_key API_KEY] [--temperature TEMPERATURE]
```

#### Arguments:
- `--subsets`, `-s`: Name of the subset(s) to evaluate the model on. Default is `["tikz", "svg"]`.
- `--model_type`, `-t`: Type of the model to evaluate. Required.
- `--model`, `-m`: Name of the model to evaluate. Required.
- `--gpu_number`: GPU number to use for evaluation. Default is `1`.
- `--api_url`: URL of the OpenAI completion compatible API. Default is `"https://api.openai.com/v1"`.
- `--api_key`: API key for authentication, defaults to the environment variable `OPENAI_API_KEY`.
- `--temperature`: Temperature setting for model sampling. Default is `0.7`.

This command will evaluate the specified model on the given subsets using the provided parameters.



## Dataset

The dataset is created from the scripts situated in varbench/dataset_workflow

- Each folder in the dataset is a subset.
- Each subset contains a list of entries in the dataset in the form of a folder.
- These folders contain an input and a folder solutions.
- Each entry in the dataset has a instruction as well.


### Publishing the dataset
You can run the following command:

```sh
python3 -m varbench.dataset_workflow.create_dataset [-h] --dataset DATASET
```

The script will first compute the patches for each entry, then will add, commit, and push the changes to get a commit id and create the dataset with the instruction, the repo, the id of the commit, and the patch.


