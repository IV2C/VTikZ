from varbench.utils.parsing import get_config
import subprocess
from datetime import datetime
import time
from loguru import logger


def launch_model(model_name: str, **kwargs) -> str:
    llm_args = {**get_config("VLLM")}
    args = [
        f"--{arg} {value}" if not isinstance(value, bool) else f"--{arg}"
        for arg, value in llm_args.items()
        if value
    ]
    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    logfile = open(f".log/{model_name.replace("/","_")}_{dt_string}", "w+")
    logger.info(" ".join(["vllm", "serve"] + args))
    subprocess.Popen(
        ["vllm", "serve", model_name] + args, text=True, stdout=logfile, stderr=logfile
    )
    time.sleep(5000)
