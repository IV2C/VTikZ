## GRID 5000 setup

> [!WARNING]  
> Not tested

1. Reserve a node
```sh
oarsub -I -i ~/.ssh/id_rsa -q production -p "gpu_mem>=24000" -l host=1,walltime=1 -t deploy 
```
2. Deploy the node
```sh
kadeploy3 debian11-min
```
3. Log in the node via ssh
```sh
ssh root@rennes.<node_name>.g5k
```

4. create Venv and Install dependencies
```sh
git clone git@github.com:VarBench-SE/VarBench.git
conda create --name varbench python=3.12
conda activate varbench
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-fonts-recommended
sudo apt-get install texlive-fonts-extra
sudo apt-get install texlive-latex-extra
sudo apt-get install poppler-utils
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install coverage coverage-badge
```