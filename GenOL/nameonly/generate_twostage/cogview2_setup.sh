#!/bin/bash

# Install required Python packages
pip install icetk
pip install Pillow
pip install diffusers
pip install "SwissArmyTransformer==0.2.8"
pip install accelerator
pip install torchvision
pip install icetk
pip install "protobuf==3.20.0"
pip install "numpy==1.24.1"

# Clone the GitHub repository
git clone https://github.com/Sleepychord/Image-Local-Attention
cd Image-Local-Attention

# Set CUDA path
export PATH=/usr/local/cuda/bin:$PATH

# Install the package
python setup.py install
cd ../

# Run the Python program
python - <<EOF
from CogView2.generator import Cogview2
model = Cogview2(img_size=224, style='photo', batch_size=1, max_inference_batch_size=1)
image = model.generate_images("Cat")
image.save("cat.jpg")
EOF