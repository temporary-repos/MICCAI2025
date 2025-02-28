# MICCAI2025
Repository to hold code for the MICCAI 2025 work "X"

### To Run
The main training and evaluation script is held within the <code>main_meta-pinn.py</code> script, containing implementations for the proposed model and neural baselines considered.

Included are configuration files within <code>configs/</code> which contains hyperparameters to run a model on. To switch between training and evaluation logic of the scripts, the config hparam <code>train_stage</code> should be set between 1 (training) or 2 (evaluation).

It contains a few argparse arguments:
<ul>
  <li>--cuda_device: which GPU ID to run on</li>
  <li>--generate_data: whether to generate synthetic data for the run or load in real data</li>
  <li>--model_type: which of the neural models to run (multi_pinn, meta_pinn, meta_base)</li>
</ul>
