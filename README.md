# Common-mode noise corrections
Collection of scripts to test different ways of correcting for common-mode noise in CMS HGCAL modules

# lxplus
All of the following assumes the user has access to CERN `lxplus`. Always use `lxplus9` when possible. For the actual training, use `lxplus-gpu`

# First-time setup
The code will be cloned into the current directory, and a virtual environment `torch-env` will be created in the `eos` home directory.

```bash
git clone git@github.com:reimersa/CMCorrection.git
cd CMCorrection
python3.9 -m venv /eos/user/${USER:0:1}/${USER}/torch-env
```
and then
```bash
source /eos/user/${USER:0:1}/${USER}/torch-env/bin/activate
pip install -r requirements.txt
```
# Every-time setup
Every time a new shell is opened, activate the virtual environment and `cd` into the repo (make an alias!)
```bash
cd <PATH/TO/CMCORRECTION/REPO>
source /eos/user/${USER:0:1}/${USER}/torch-env/bin/activate
```

# Running the scripts
## Extracting inputs and targets from ROOT files
To convert NANO-like ROOT files into files that can be used for training and performance evaluation, run the following on `lxplus9`:

```bash
python prepare_inputs.py
```
Inside that file, simply give a list of modules whose data you would like to convert. 

This will create a folder `/eos/user/<FIRSTLETTER>/$USER/hgcal/dnn_inputs/<MODULENAME>` for each module in the list of modules. This folder will hold all you need.

Also, a set of plots will be created in a folder `plots/inputs/<MODULENAME>` in the current directory.


## Training the DNN
To train a DNN, run the following on `lxplus-gpu`:

```bash
python train_dnn.py
```

There is a couple of settings to adjust in that file:
- `modulenames_for_training`: `list[str]` of the modules whose data should be used to train the DNN.
- `nodes_per_layer`: `list[int]` with the number of nodes in each hidden layer. The length of the list defines the number of hidden layers.
- `dropout_rate`: `float` that gives the fraction (`0 <= dropout_rate <= 1`) of hidden nodes that are randomly disabled during training
- `modeltag`: `str` that will be prepended to each model name. By default, each model will receive a name based on its architecture and dropout rate. Can be left empty (`""`)
- `override_full_model_name`: `bool` that decides if the automatically generated name of modeltag+architecture is used (`False`) or the model is renamed to `new_model_name` (next option)
- `new_model_name`: `str` that replaces the full default model name and `modeltag` if `override_full_model_name == True` (see previous option)

This will create a folder `/eos/user/<FIRSTLETTER>/$USER/hgcal/dnn_models/'_'.join(modulenames_for_training)/<MODELNAME>` with all necessary files. If the folder exists, the program will crash to avoid overwriting existing models. See options above for `<MODELNAME>`.


## Evaluate the DNN performance

To evaluate the performance of a DNN, run the following on `lxplus9`:

```bash
python plot_performance.py
```

There is a couple of settings to adjust in that file, largely identical with those used when training the DNN:
- `modulenames_used_for_training`: `list[str]` of the modules that were used to train the DNN
- `modulename_for_evaluation`: `str` of the module whose data should be used to evaluate the DNN
- `username_load_model_from`: `str` that is the username of the user from whose `/eos` area the DNN model should be loaded.
- `modelname`: `str` that can be either `regression_dnn` or `linreg` to distinguish between types of regression. 
- `nodes_per_layer`: `list[int]` with the number of nodes in each hidden layer. The length of the list defines the number of hidden layers.
- `dropout_rate`: `float` that gives the fraction (`0 <= dropout_rate <= 1`) of hidden nodes that are randomly disabled during training
- `modeltag`: `str` that will be prepended to each model name. By default, each model will receive a name based on its architecture and dropout rate. Can be left empty (`""`)
- `override_full_model_name`: `bool` that decides if the automatically generated name of modeltag+architecture is used (`False`) or the model is renamed to `new_model_name` (next option)
- `new_model_name`: `str` that replaces the full default model name and `modeltag` if `override_full_model_name == True` (see previous option)

This will create a set of plots in a folder `plots/performance/<modulenames_used_for_training>/<MODELNAME>/inputs_from_<modulename_for_evaluation>` in the current directory.