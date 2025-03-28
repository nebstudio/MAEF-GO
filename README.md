# MAEF-GO

Multi-Stage Attention-Based Extraction and Fusion of Protein Sequence and Structural Features for Protein Function Prediction

## Setup Environment

### Clone the Repository

```bash
git clone https://github.com/nebstudio/MAEF-GO
Create Conda Environment

conda env create -f environment.yml
Install PyTorch

conda install pytorch==1.7.0 cudatoolkit=10.2 -c pytorch
Install PyTorch Geometric (PYG)
Download the necessary wheel files:


wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
Install the downloaded wheel files:


pip install *.whl
Install torch_geometric:


pip install torch_geometric==1.6.3
Additional Packages
You also need to install the relative packages to run the ESM-1b protein language model.

Model Testing
Download the dataset from here.

Extract the dataset:


tar -zxvf dataset.tar.gz
Run the testing script:


python test.py --device 0 --task bp --batch_size 64
