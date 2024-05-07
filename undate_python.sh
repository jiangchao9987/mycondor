wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3 -s
rm Miniconda3-latest-Linux-x86_64.sh
conda=miniconda3/bin/conda
$conda create -y -n pytorch_new python=3.10.13 pip
$conda install -n pytorch_new -y pytorch torchvision cudatoolkit=10.1 -c pytorch 



