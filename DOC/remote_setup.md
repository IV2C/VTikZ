## GRID 5000 setup

> [!WARNING]  
> Not tested


1. Reserve a node
```sh
oarsub -I -i ~/.ssh/id_rsa -q production -p "gpu_mem>=24000" -l host=1,walltime=1 -t deploy 
```
1. Deploy the node
```sh
kadeploy3 debian11-min
```
1. Log in the node via ssh
```sh
ssh root@<node_name>.rennes.grid5000.fr
```
4. From local
```sh
scp .ssh/id_ed25519 root@<node_name>.rennes.g5k:~/.ssh/
```

5. create Venv and Install dependencies
6. gpu install => https://www.grid5000.fr/w/GPUs_on_Grid5000#Reserving_full_nodes_with_GPUs
```sh
apt update
echo "deb http://deb.debian.org/debian/ bullseye main contrib non-free" | sudo tee -a /etc/apt/sources.list
apt install -y make libssl-dev libghc-zlib-dev libcurl4-gnutls-dev libexpat1-dev unzip git-all
#nvidia install 
apt install  -y linux-headers-amd64 make g++
wget https://download.nvidia.com/XFree86/Linux-x86_64/470.82.01/NVIDIA-Linux-x86_64-470.82.01.run
rmmod nouveau
sh NVIDIA-Linux-x86_64-470.82.01.run -s --no-install-compat32-libs


git clone git@github.com:VarBench-SE/VarBench.git
git checkout dev

#conda installation
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

#repo setup
conda create --name varbench python=3.12
conda activate varbench
apt-get install -y texlive-latex-base
apt-get install -y texlive-fonts-recommended
apt-get install -y texlive-fonts-extra
apt-get install -y texlive-latex-extra
apt-get install -y poppler-utils
python -m pip install --upgrade pip
pip install -r requirements.txt
```
6. check number of gpus:
```sh
nvidia-smi
```
7. launch whatever script you need

