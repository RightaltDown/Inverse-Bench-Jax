
![](assets/logo.png)

[**InverseBench: Benchmarking Plug-and-Play Diffusion Priors for Inverse Problems in Physical Sciences**](https://openreview.net/pdf?id=U3PBITXNG6) (ICLR 2025 spotlight)

## Operations
Main training is run through `nnx_train.py` through `train_and_evaluate()`. Configuration is temperorarily in configs/default.py. 

## JAX related

Data paralleism is done through `data_sharding` which specifies how the trainining data is sharded onto devices. 

The `util.parallelism.setup_initial_state()` performs model replication across devices which ensures data parallelism. Later can be adapted to be fully sharded data paralleism (FSDP). 

`input_pipeline` loads the data 


## How to run

Install Jax:
```
pip install -U "jax[cuda12]"
```

Install Flax + dependencies:

```
pip install -r requirements.txtYoussef Marzouk  
```
or for conda env:

```
conda env create -f environment.yml
```
`environment_full.yml` contains all the conda packages if there are still missing packages when running with `environment.yml`.  \

And finally start the training:

```
python3 nnx_main.py --workdir=$HOME/logs/navier_stokes \
    --config.per_device_batch_size=4
```


## License
This project is licensed under the MIT License - see the LICENSE file for details.