Packages that are needed including `pytorch-geometric`, `pytorch`, and `pybind11`.
The code are tested under `cuda113` and `cuda116` environment. Please consider download these packages with the following commands:

```
# pytorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# pytorch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

# pybind11 (used for c++ sampler)
pip install pybind11
```



## Run the code

Step 1: Compile C++ sampler (from https://github.com/amazon-science/tgl).
```
python setup.py build_ext --inplace
```

Step 2: Download data by using `DATA/down.sh`. To create the sub-sampled version of GDELT dataset, please use `DATA/GDELT_lite/gen_dataset.py`.

Step 3: Preprocess data (from https://github.com/amazon-research/tgl)
```
python gen_graph.py --data REDDIT
```
Please replace `REDDIT` to other datasets, e.g., `WIKI`, `MOOC`, `LASTFM`,

Step 3: Run experiment
```
python train.py --data REDDIT     --num_neighbors 50 --use_cached_subgraph --use_onehot_node_feats
python train.py --data WIKI       --num_neighbors 50 --use_cached_subgraph --use_onehot_node_feats
python train.py --data MOOC       --num_neighbors 50 --use_cached_subgraph --use_onehot_node_feats
python train.py --data LASTFM     --num_neighbors 50 --use_cached_subgraph --use_onehot_node_feats
python train.py --data SX     --num_neighbors 50 --use_cached_subgraph
```

If you are running this dataset for the first, it need to take sometime pre-processing the input data. But it will only do it once.


