#! /eos/user/a/areimers/torch-env/bin/python
import os
import subprocess
from itertools import product

import models


# Locate train_dnn.py in the same folder as this script using os.path
current_dir = os.path.dirname(os.path.realpath(__file__))
train_script = os.path.join(current_dir, "train_dnn.py")
if not os.path.exists(train_script):
    raise FileNotFoundError(f"Could not find training script at {train_script}")

# Directory to store condor submit files and logs (relative to script)
workdir = os.path.join(current_dir, "workdir_condor")
os.makedirs(workdir, exist_ok=True)

# Define your sweep of hyper‐parameters here:
modules_list    = [["ML_F3W_WXIH0190"], ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191"]]
# modules_list    = [["ML_F3W_WXIH0190", "ML_F3W_WXIH0191", "ML_F3W_WXIH0192", "ML_F3W_WXIH0193"]]

nodes_choices   = [[512, 512, 512, 512, 64]]

dropout_choices = [0., 0.05, 0.1, 0.2, 0.3]
# dropout_choices = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

epoch_choices   = [1000]

# inputfoldertag  = ""
inputfoldertag  = "_nochadc"

# modeltag        = "submittest"
modeltag        = ""

# jobflavor   = "microcentury"
jobflavor   = "testmatch"

override_name = False
new_name = "NOTHING"

def main():

    # make workdir and copy voms proxy
    os.makedirs(workdir, exist_ok=True)

    proxy_filename_orig = f"/tmp/x509up_u{os.getuid()}"
    proxy_filename_forjob = f"{workdir}/voms_proxy"
    if not os.path.exists(proxy_filename_orig):
        raise ValueError(f"VOMS proxy {proxy_filename_orig} does not exist, please set it up.")
    copycommand = f"cp {proxy_filename_orig} {proxy_filename_forjob}"
    os.system(copycommand)
    print(f"Copied proxy file to {proxy_filename_forjob}.")


    for modules, nodes, dropout, epochs in product(
        modules_list, nodes_choices, dropout_choices, epoch_choices
    ):

        model = models.DNNFlex(input_dim=20, nodes_per_layer=nodes, dropout_rate=dropout, tag=modeltag)
        name = f"{'_'.join(modules)}{inputfoldertag}_{model.get_model_string()}".replace("in20__", "")
        # Build the training‐script args string:
        args = []
        args += ["-m"] + modules
        args += ["-n"] + [str(n) for n in nodes]
        args += ["-d", str(dropout)]
        args += ["-e", str(epochs)]
        if modeltag:
            args += ["-t", modeltag]
        if inputfoldertag:
            args += ["--inputfoldertag", inputfoldertag]
        if override_name:
            args += ["--override-name", "--new-name", new_name]

        args_str = " ".join(args)

        # Make a working directory for logs
        logdir = os.path.join(workdir, name)
        os.makedirs(logdir, exist_ok=True)

        # Read wrapper template:
        with open("wrapper_train_dnn.sh.template") as f:
            tmpl_wrap = f.read()

        
        # Fill in placeholders:
        content_wrap = tmpl_wrap.format(
            VENV=f"/eos/user/{os.getenv('USER')[0]}/{os.getenv('USER')}/torch-env",
            EXE=train_script,
            ARGS=args_str,
        )

        # Write the wrapper
        wrapper_thisjob = os.path.join(workdir, name, f"wrapper.sh")
        with open(wrapper_thisjob, "w") as f:
            f.write(content_wrap)



        # Read sub template:
        with open("train_dnn.sub.template") as f:
            tmpl_sub = f.read()

        # Fill in placeholders:
        content_sub = tmpl_sub.format(
            # VENV=venv_path,
            # EXE=train_script,
            # ARGS=args_str,
            GPUS=1,
            LOGDIR=logdir,
            NAME=name,
            JOBFLAVOR=jobflavor,
            MEMGB=f"{4*len(modules)}",
            VOMS=proxy_filename_forjob,
            WRAPPER=wrapper_thisjob
        )

        # Write the subfile
        subfile = os.path.join(workdir, f"{name}.sub")
        with open(subfile, "w") as f:
            f.write(content_sub)

        # submit
        subprocess.run(['condor_submit', subfile], check=True)
        print(f"Submitted job: {name}")

if __name__ == "__main__":
    main()