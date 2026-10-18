pip install icetk
pip install Pillow
pip install diffusers
pip install "SwissArmyTransformer==0.2.8"
pip install accelerator
pip install torchvision
pip install icetk
pip install "protobuf==3.20.0"
pip install "numpy==1.24.1"

git clone https://github.com/Sleepychord/Image-Local-Attention
cd Image-Local-Attention
export PATH=/usr/local/cuda/bin:$PATH
python setup.py install

/root/.sat_models/
/home/.sat_models

# Datacenter setup
1. sharef.tar scratch에 scp로 옮기고 ln -s
2. conda create -n cogview python=3.10.8
3. pip install icetk Pillow diffusers SwissArmyTransformer==0.2.8 accelerator torchvision
4. pip install icetk protobuf==3.20.0

# If numpy error occurs
pip install numpy==1.24.1
