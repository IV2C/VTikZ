from varbench.utils.parsing import get_config
import subprocess
from datetime import datetime
import time
from loguru import logger
import atexit

def _stop_process(process:subprocess.Popen):
    if process.poll() is None: 
        process.terminate()  
        try:
            process.wait(timeout=5)  
        except subprocess.TimeoutExpired:
            process.kill()
    logger.info("Vllm subprocess finished")
            
def launch_model(model_name: str, **kwargs) -> subprocess.Popen[str]:
    """Launches the provided model with the parameters in config-varbench

    Args:
        model_name (str): Name of the launched model

    Returns:
        subprocess.Popen[str]: The python vllm subprocess
    """
    llm_args = {**get_config("VLLM")}
    try :
        do_run = llm_args.pop("do-run")
    except KeyError:
        logger.info("parameter do-run not set in config file, not launching LLM via VLM")
        return
    if not do_run:
        logger.info("parameter do-run set to false in config file, not launching LLM via VLM")
        return
    args = [
        f"--{arg} {value}" if not isinstance(value, bool) else f"--{arg}"
        for arg, value in llm_args.items()
        if value
    ]
    args = [arg for cur_arg in args for arg in cur_arg.split(" ")]
    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    logfile = open(f".log/{model_name.replace("/","_")}_{dt_string}", "w+")
    full_args = ["vllm", "serve", model_name] + args
    logger.warning(full_args)
    process = subprocess.Popen(
        full_args, text=True, stdout=logfile, stderr=subprocess.PIPE
    )
    atexit.register(_stop_process,process)

    while True:
        line = process.stderr.readline()
        if len(line) > 0 and not line.strip() == "\n" and not line.strip() == "":
            logfile.write(line)
        if "Application startup complete" in line:
            break
    logger.info("Vllm server running")
    return process
