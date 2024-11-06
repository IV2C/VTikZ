<h1 align="center">
 VarBench
</h1>

<p align="center">  <a href="https://github.com/VarBench-SE/VarBench">🏠 Home Page</a> • <a href="https://huggingface.co/datasets/CharlyR/varbench">🤗 Dataset</a>   </p>

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
- `--passk`: number of gneerated responses, used to compute pass@k. default is 1
- `--no_batch`: whether or not to use batch requests, relevant when the api provider does not provide an equivalent. Default is False.
This command will evaluate the specified model on the given subsets using the provided parameters.

#### Examples:

- **Using the API model:**

  ###### With the openai api
  ```sh
  python3 -m varbench.run_evaluation --subsets tikz svg --model_type API --model gpt-3.5-turbo --api_key YOUR_API_KEY
  ```

  Replace `YOUR_API_KEY` with your actual API key.

  ###### With another openai compaptible api
  ```sh
  python3 -m varbench.run_evaluation --subsets tikz --model_type API --temperature 0.7 --passk 5 --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY -m llama-3.1-70b-versatile --no_batch
  ```


- **Using the VLLM model:**

  ```sh
  python3 -m varbench.run_evaluation --subsets tikz svg --model_type VLLM --model meta-llama/Llama-3.2-1B-Instruct --gpu_number 2
  ```

  This command uses the VLLM model named `llama-3.2` and specifies GPU number 2 for evaluation.

  tips: you can tweak the logging level of vllm via `export VLLM_LOGGING_LEVEL=DEBUG`


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


