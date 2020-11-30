# move to home
cd ~
pip3 install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html

# build CUDA Layers
cd votenet/pointnet2
sudo python3 setup.py install
cd ..
pip3 install -r requirements.txt


cd ~ # move to home
wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-314.0.0-darwin-x86_64.tar.gz
tar xvzf google-cloud-sdk-314.0.0-darwin-x86_64.tar.gz
./google-cloud-sdk/install.sh
source .bashrc
gcloud init