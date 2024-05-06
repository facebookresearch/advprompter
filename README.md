# AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs

This repo is an official implementation of **AdvPrompter** ([arxiv:2404.16873](https://arxiv.org/abs/2404.16873)).

Please 🌟star🌟 this repo and cite our paper 📜 if you like (and/or use) our work, thank you!

## 0. Installation

- Option 1. Use Singularity to install the advprompter.def container
- Option 2. Install python3.11 and requirements.txt manually:
```bash
  conda create -n advprompter python=3.11.4
  conda activate advprompter
  pip install -r requirements.txt
```

## 1. Running AdvPrompter

We use hydra as a configuration management tool.
Main config files: ```./conf/{train,eval,eval_suffix_dataset,base}.yaml```
The AdvPrompter and the TargetLLM are specified in conf/base.yaml, various options are already implemented.

The codebase optionally supports wandb by setting the corresponding options in conf/base.yaml.

#### Note on hardware specifications: all the experiments we conducted utilized two NVIDIA A100 GPUs, one for AdvPrompter and one for TargetLLM. You can manage devices in conf/target_llm/base_target_llm.yaml and conf/prompter/base_prompter.yaml

### 1.1 Evaluation
Run
> python3 main.py --config-name=eval

to test the performance of the specified AdvPrompter against the TargetLLM on a given dataset. You'll have to specify TargetLLM and AdvPrompter in conf/base.yaml. Also, you may want to specify a path to peft_checkpoint if AdvPrompter was finetuned before:
>   // see conf/prompter/llama2.yaml\
>   lora_params: \
>   warmstart: true \
>   lora_checkpoint: "path_to_peft_checkpoint"

The suffixes generated during evaluation are saved to a new dataset under the run-directory in ```./exp/.../suffix_dataset``` for later use.
Such a dataset can also be useful for evaluating baselines or hand-crafted suffixes against a TargetLLM, and it can be evaluated by running
> python3 main.py --config-name=eval_suffix_dataset 

after populating the ```suffix_dataset_pth_dct``` in ```eval_suffix_dataset.yaml```

### 1.2. Training
Run
> python3 main.py --config-name=train

to train the specified AdvPrompter against the TargetLLM. It automatically performs the evaulation specified above in regular intervals, and it also saves intermediate versions of the AdvPrompter to the run-directory under ```./exp/.../checkpoints``` for later warmstart. Checkpoint can be specified with the ```lora_checkpoint``` parameter in the model configs (as illustrated in 1.1 Evaluation).
Training also saves for each epoch the target suffixes generated with AdvPrompterOpt to ```./exp/.../suffix_opt_dataset```.
This allows pretraining on such a dataset of suffixes by specifying the corresponding path under pretrain in ```train.yaml```

Some important hyperparameters to consider in conf/train.yaml: ```[epochs, lr, top_k, num_chunks, lambda_val]```

#### Examples

Note: you may want to replace target_llm.llm_params.checkpoint with a local path.

+ **Example 1:** AdvPrompter on Vicuna-7B:
  ```bash
   python3 main.py --config-name=train target_llm=vicuna_chat target_llm.llm_params.model_name=vicuna-7b-v1.5
   ```
+ **Example 2:** AdvPrompter on Vicuna-13B:
  ```bash
   python3 main.py --config-name=train target_llm=vicuna_chat target_llm.llm_params.model_name=vicuna-13b-v1.5 target_llm.llm_params.checkpoint=lmsys/vicuna-13b-v1.5 train.q_params.num_chunks=2
   ```
+ **Example 3:** AdvPrompter on Mistral-7B-chat:
  ```bash
   python3 main.py --config-name=train target_llm=mistral_chat
   ```

+ **Example 4:** AdvPrompter on Llama2-7B-chat:
  ```bash
   python3 main.py --config-name=train target_llm=llama2_chat train.q_params.lambda_val=150
   ```
  
## 2. Contributors

<a href="https://scholar.google.com/citations?user=njZL5CQAAAAJ">Anselm Paulus*</a>,
<a href="https://arman-z.github.io/">Arman Zharmagambetov*</a>,
<a href="https://sites.google.com/view/chuanguo">Chuan Guo</a>,
<a href="http://bamos.github.io/">Brandon Amos**</a>,
<a href="https://yuandong-tian.com/">Yuandong Tian**</a>

<font color="#7F7F7F">(* = Equal 1st authors, ** = Equal advising)</font>

## 3. License
Our source code is under [CC-BY-NC 4.0 license](./LICENSE).