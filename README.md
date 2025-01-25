<h1 align="center">
 VarBench
</h1>

<p align="center">  <a href="https://github.com/VarBench-SE/VarBench">🏠 Home Page</a> • <a href="https://huggingface.co/datasets/CharlyR/varbench">🤗 Dataset</a>   </p>

![](DOC/images/coverage.svg)


# Evaluation

## Installation

```sh
conda create -n "varbench" python=3.12.7
pip install -r requirements.txt
```

## Running the Evaluation


To execute the evaluation for any subset, use the following script:

```sh
python3 -m varbench.run_evaluation [-h] --subsets SUBSETS [SUBSETS ...] --metrics METRICS [METRICS ...] --agent AGENT --model MODEL [--vlm VLM] [--vlm_api_url VLM_API_URL] [--vlm_api_key VLM_API_KEY] [--vlm-temperature VLM_TEMPERATURE] [--interaction-amount INTERACTION_AMOUNT] [--run-model] [--api_url API_URL] [--api_key API_KEY] [--temperature TEMPERATURE] [--passk PASSK]
```


### Arguments

#### Core Arguments
- `--subsets`, `-s`: Subset(s) to evaluate the model on. Defaults to `["tikz"]`, futures subset could be "svg, p5js, pygame, etc".
- `--metrics`, `-me`: List of metrics for evaluation. Defaults to `["patch","line","clipImage","clipText","bleu","bleuPatch","crystalBleu","crystalBleuPatch","chrf","chrfPatch","TER","TERPatch","featureMatch","LPIPS","psnr","msssim","MSE",]`.
- `--agent`, `-a`: Name of the agent to use. **Required**. Choices: `["simpleLLM", "simpleLMM", "loopVLMLLM"]`.
- `--model`, `-m`: Name of the model to evaluate. **Required**.
- `--run-model`, `-r`: Launch the model locally for evaluation. If used with `--api_url`, `api_url` is ignored.

#### VLM Settings (for `loopVLMLLM` Agent)
- `--vlm`, `-v`: Name of the VLM to use.
- `--vlm_api_url`: URL of the OpenAI-compatible API for the VLM.
- `--vlm_api_key`: API key for VLM authentication. Defaults to the environment variable `OPENAI_API_KEY`.
- `--vlm-temperature`: Sampling temperature for the VLM. Defaults to `0`.
- `--interaction-amount`: Number of interactions between the LLM and VLM. Defaults to `2`.

#### API Settings
- `--api_url`: URL of the OpenAI-compatible API.
- `--api_key`: API key for authentication. Defaults to the environment variable `OPENAI_API_KEY`.
- `--temperature`: Sampling temperature for the model. Defaults to `0.7`.
- `--passk`: Number of responses per prompt for computing pass@k. Defaults to `1`.

### Outputs

- Results are saved in the `./results` directory under subfolders based on the model name. Each subset generates:
  - A `.json` file summarizing evaluation scores.
  - Evaluation datasets stored for further analysis or sharing.

All results are published to https://huggingface.co/datasets/CharlyR/varbench-evaluation for now, access is needed to publish new results.
The datasets are analysed using this [notebooks](/home/creux/Documents/AI/VariabilityBenchmark/notebooks/result_analysis.ipynb)

## Additional Configuration

Some additional configurations can be set up by modifying `config-varbench.cfg`
Each section is described below.
### \[VLLM\]
If you want to run the model locally, all the parameters can be set in this section, for example `trust-remote-code = True`.
See [this](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#cli-reference) documentation for reference.

### \[CLIP\]
The clip model for the clip metric is launched using the [open_clip](https://github.com/mlfoundations/open_clip) library.  
**model_name** (e.g., `ViT-bigG-14-quickgelu`),**pretrained_name**(e.g., `metaclip_fullcc`) and **force_cpu** parameters can be set.


### \[MAIN\]
General settings.
- **cache_enabled**: Enables caching (`True` or `False`).
- **cache_location**: Cache directory (e.g., `.cache`).

### \[API\]
API configuration.
- **seed**: Random seed for reproducibility (e.g., `456789`, will be used for the cache, and for the OpenAI-compaptible apis that support it).


### Work In progress
  #### CODE_CORRECT_AGENT
  Settings for the code correction agent.
  - **max_iteration**: Max correction attempts (e.g., `5`).

  #### RENDERER
  Renderer settings for visual output.
  - **p5js_browser_path**: Path to the browser for rendering the p5.js sketches.



## Examples

### Using the API Model

- **With the OpenAI API**:
  ```sh
  python3 -m varbench.run_evaluation --subsets tikz svg --model gpt-3.5-turbo --api_key YOUR_API_KEY --agent simpleLLM
  ```

- **With Another OpenAI-Compatible API**:
  ```sh
  python3 -m varbench.run_evaluation --subsets tikz --model llama-3.1-70b-versatile --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY --temperature 0.7 --passk 5 --agent simpleLLM
  ```

### Running Locally

- **Using a Locally Launched Model**:
  ```sh
  python3 -m varbench.run_evaluation --subsets tikz --model meta-llama/Llama-3.2-1B-Instruct --run-model --temperature 0.9 --passk 3 --agent simpleLLM
  ```

### Using the `loopVLMLLM` Agent

- **With a VLLM API at runpod**:
  ```sh
  python3 -m varbench.run_evaluation --subsets svg --model meta-llama/Llama-3.2-1B-Instruct --api_url https://api.runpod.ai/YOURAPI/openai/v1 --api_key $RUNPOD_API_KEY --vlm llava-hf/llava-1.5-7b-hf --vlm_api_url https://api.runpod.ai/YOURAPI/openai/v1 --vlm_api_key $VLM_RUNPOD_API_KEY --interaction-amount 2 --agent loopVLMLLM
  ```
  
- **With a Groq api**:
  ```sh
  python3 -m varbench.run_evaluation --subsets tikz --model llama-3.1-70b-versatile --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY --vlm llava-v1.5-7b-4096-preview --vlm_api_url https://api.groq.com/openai/v1 --vlm_api_key $GROQ_API_KEY --interaction-amount 1 --agent loopVLMLLM --passk 1
  ```

## Notes
1. Ensure required environment variables (e.g., `OPENAI_API_KEY` or `HF_TOKEN`) are set if not explicitly passed as arguments.
2. For local models, ensure compatibility with the `vllm` framework.
3. The script dynamically creates directories for saving results if they do not exist.

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

# Synthetic data generation
TODO: fix the script and Add documentation for the synthetic data generation

### Additional Notes
- **Environment Variables**: If `--api_key` is not provided, the script uses `OPENAI_API_KEY`.

