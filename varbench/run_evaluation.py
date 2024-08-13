from huggingface_hub import login
from datasets import load_dataset
import os


#login(token=os.environ.get("HF_TOKEN"))

dataset = load_dataset("CharlyR/varbench",split="train")



dataset.to_json("dataset.json")
df_varbench = dataset.to_pandas()

df_varbench

