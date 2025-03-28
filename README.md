

# MAEF-GO
Multi-Stage Attention-Based Extraction and Fusion of Protein Sequence and Structural Features for Protein Function Prediction.  

Most of the codes in this study are obtained from [HEAL](https://github.com/ZhonghuiGu/HEAL) 
##  Clone the Repository
```bash
git clone https://github.com/nebstudio/MAEF-GO
cd MAEF-GO
```
## Set Up the Environment
```bash
conda env create -f environment.yml
conda activate MAEF-GO
conda install pytorch==1.7.0 cudatoolkit=10.2 -c pytorch
```
## Install PyTorch Geometric and Dependencies
 ```bash
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install *.whl
pip install torch_geometric==1.6.3
```
## Download and Prepare Data

```bash
cd data
```
Data set can be downloaded from [here](https://pan.webos.cloud/?share=ebdMbB8p&password=8Eh2&).
```bash
tar -zxvf processed.tar.gz
```
## Run the testing script
```bash
python test.py --device 0 
               --task bp 
               --batch_size 64
```
             
