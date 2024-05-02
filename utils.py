# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import csv
import json
from functools import wraps
from tqdm import tqdm

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FalconForCausalLM,
    GemmaForCausalLM,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
)


def hit_rate_at_n(jb_stat, n):
    jb_sum_at_n = np.sum(jb_stat[:, :n], axis=1)
    return np.where(jb_sum_at_n > 0, 1.0, jb_sum_at_n).mean()


def apply_repetition_penalty(logits, prev_ids, penalty):
    _logits = torch.gather(input=logits, dim=1, index=prev_ids)

    # if score < 0 then repetition penalty has to be multiplied to reduce the previous
    # token probability
    _logits = torch.where(_logits < 0, _logits * penalty, _logits / penalty)

    logits_penalized = torch.scatter(input=logits, dim=1, index=prev_ids, src=_logits)
    return logits_penalized


def get_nonascii_toks(tokenizer, device="cpu"):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)


def compute_perplexity(id_seq, likelihood_seq):
    logprobs = torch.gather(
        likelihood_seq.logprobs, dim=2, index=id_seq.ids.unsqueeze(2)
    ).squeeze(2)

    perplexity_per_token_masked = torch.exp(-logprobs) * id_seq.mask
    perplexity = torch.exp(
        -torch.sum(logprobs * id_seq.mask, dim=1)
        / (torch.sum(id_seq.mask, dim=1) + 1e-8)
    )

    return perplexity, perplexity_per_token_masked


def add_dummy_dim_to_slice(slice_obj):
    # First, check if slice_obj is a single slice or a tuple of slices
    if not isinstance(slice_obj, tuple):
        slice_obj = (slice_obj,)

    # Modify the slice_obj to add a new axis where necessary
    new_slice = []
    for sl in slice_obj:
        # Check if it is a single index (int) and add new axis after this dimension
        if isinstance(sl, int):
            new_slice.append(sl)
            new_slice.append(None)
        else:
            new_slice.append(sl)

    return tuple(new_slice)


class ReturnStruct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def clone(self):
        new_kwargs = {}
        for k, v in self.__dict__.items():
            try:
                new_kwargs[k] = v.clone()
            except:
                new_kwargs[k] = v
        return ReturnStruct(**new_kwargs)

    def detach(self):
        new_kwargs = {}
        for k, v in self.__dict__.items():
            try:
                new_kwargs[k] = v.detach()
            except:
                new_kwargs[k] = v
        return ReturnStruct(**new_kwargs)

    def _detach(self):
        for k, v in self.__dict__.items():
            try:
                v._detach()
            except:
                pass

    def to(self, device):
        new_kwargs = {}
        for k, v in self.__dict__.items():
            try:
                new_kwargs[k] = v.to(device)
            except:
                new_kwargs[k] = v
        return ReturnStruct(**new_kwargs)


def ce_loss(pred_seq, target_seq, hard_labels, reweight_loss=False, **kwargs):
    if hard_labels:
        loss = F.cross_entropy(
            pred_seq.logits.transpose(1, 2), target_seq.ids, reduction="none", **kwargs
        )
    else:
        loss = F.cross_entropy(
            pred_seq.logits.transpose(1, 2),
            target_seq.probs.transpose(1, 2),
            reduction="none",
            **kwargs,
        )
    if reweight_loss:
        factor = torch.arange(loss.shape[1], dtype=loss.dtype, device=loss.device) + 1
        loss = loss / factor[None, :]
    return loss


def loss_seqs(pred_seq, target_seq, **loss_params):
    if torch.isnan(pred_seq.logits).any():
        raise ValueError(f"Nan in logits: {pred_seq.logits}")
    _loss = ce_loss(pred_seq, target_seq, **loss_params)
    mask = target_seq.mask
    loss_masked = _loss * mask
    loss_batch = torch.sum(loss_masked, dim=1) / (mask.sum(dim=1) + 1e-10)
    loss = loss_batch.mean()

    ce_return = ReturnStruct(
        loss=loss,
        loss_masked=loss_masked,
        loss_batch=loss_batch,
        pred=pred_seq,
        label=target_seq,
    )
    return ce_return


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    tqdm.write(
        f" trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


def check_jailbroken(seq, test_prefixes):
    jailbroken_list = [
        all([prefix not in text for prefix in test_prefixes]) for text in seq.text
    ]
    jailbroken_avg = list_avg(jailbroken_list)
    return jailbroken_avg, jailbroken_list


def check_success(seq, target_seq):
    success_list = [
        target_seq.text[i].lower() in text.lower() for i, text in enumerate(seq.text)
    ]
    success_avg = list_avg(success_list)
    return success_avg, success_list


def check_affirmative(seq, affirmative_prefixes):
    affirmative_list = [
        any(
            [
                text[: len(prefix)].lower() == prefix.lower()
                for prefix in affirmative_prefixes
            ]
        )
        for text in seq.text
    ]
    affirmative_avg = list_avg(affirmative_list)
    return affirmative_avg, affirmative_list


def list_avg(_list):
    return sum(_list) / len(_list)


column_names = [
    "step",
    "split",
    "batch_idx",
    "sample_idx",
    # Prompt prediction
    "prompter/ar/query",
    "prompter/ar/response",  # auto-regressive prompter generation
    "prompter/ar/response_perplexity_basemodel",
    #
    # --- Evaluation of predicted prompt ---
    "target_llm/ar/query",
    "target_llm/ar/response",
    "target_llm/ar/jailbroken",
]


def log_data(
    log_table,
    metrics,
    step,
    split,
    batch_idx,
    test_prefixes,
    affirmative_prefixes,
    log_sequences_to_wandb,
    log_metrics_to_wandb,
    batch_size=None,
    target_llm_tf=None,
    target_llm_ar=None,
    prompter_ar=None,
    basemodel_tf=None,
    prompter_tf_opt=None,
):
    if batch_size is None and prompter_ar is None:
        raise ValueError("either batch_size or prompter_ar must be provided")
    bs = batch_size if batch_size is not None else prompter_ar.response_sample.bs
    log_dct = {}
    log_seqs = {
        "step": [step] * bs,
        "split": [split] * bs,
        "batch_idx": [batch_idx] * bs,
        "sample_idx": list(range(bs)),
    }

    if prompter_ar is not None:
        log_seqs["prompter/ar/query"] = prompter_ar.query.to_html()
        if basemodel_tf is not None:
            log_dct["prompter/ar/response_perplexity_basemodel"] = (
                basemodel_tf.perplexity.mean().item()
            )

            log_seqs["prompter/ar/response"] = prompter_ar.response_sample.to_html(
                colors=basemodel_tf.loss_masked, normalize=True, color_scheme=2
            )
            log_seqs["prompter/ar/response_perplexity_basemodel"] = (
                basemodel_tf.perplexity
            )
        else:
            log_seqs["prompter/ar/response"] = prompter_ar.response_sample.to_html()

    if target_llm_tf is not None:
        target_llm_tf_affirmative_avg, target_llm_tf_affirmative_list = (
            check_affirmative(
                seq=target_llm_tf.response_dist,
                affirmative_prefixes=affirmative_prefixes,
            )
        )

        log_dct["target_llm/tf/response_entropy"] = (
            target_llm_tf.response_dist.get_entropy().item()
        )
        log_dct["target_llm/tf/affirmative"] = target_llm_tf_affirmative_avg
        log_dct["target_llm/tf/loss"] = target_llm_tf.loss.item()

    if target_llm_ar is not None:
        target_llm_ar_jailbroken_avg, target_llm_ar_jailbroken_list = check_jailbroken(
            seq=target_llm_ar.response_sample, test_prefixes=test_prefixes
        )

        # log_dct["target_llm/ar/jailbroken"] = target_llm_ar_jailbroken_avg
        log_dct["target_llm/ar/jailbroken_sum"] = sum(target_llm_ar_jailbroken_list)

        log_seqs["target_llm/ar/query"] = target_llm_ar.query.to_html()
        log_seqs["target_llm/ar/response"] = target_llm_ar.response_sample.to_html()
        log_seqs["target_llm/ar/jailbroken"] = target_llm_ar_jailbroken_list

    if prompter_tf_opt is not None:
        log_dct["prompter/tf/opt/response_dist_entropy"] = (
            prompter_tf_opt.response_dist.get_entropy().item()
        )
        log_dct["prompter/tf/opt/loss"] = prompter_tf_opt.loss.item()

    metrics.log_dict(log_dct, step=step, log_to_wandb=log_metrics_to_wandb)
    if log_sequences_to_wandb:
        log_data_to_table(log_table, bs, log_seqs)


def log_data_to_table(log_table, bs, log_seqs):
    log_list = []

    for column_name in column_names:
        if column_name in log_seqs:
            log_list.append(log_seqs[column_name])
        else:
            log_list.append([None] * bs)

    for bi in range(bs):
        log_l = [x[bi] for x in log_list]
        log_table.add_data(*log_l)


def autocast_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if "cuda" in self.device:
            device = "cuda"
        else:
            device = self.device

        with torch.autocast(device):
            return func(self, *args, **kwargs)

    return wrapper


def get_total_allocated_memory():
    devices = torch.cuda.device_count()
    total_allocated_memory = 0
    for i in range(devices):
        total_allocated_memory += torch.cuda.memory_allocated(f"cuda:{i}")
    return total_allocated_memory / 1e9


def expand_for_broadcast_tensor(list_of_tensors, dim=0):
    sizes = {tensor.shape[dim] for tensor in list_of_tensors}
    max_size = max(sizes)
    sizes.discard(1)
    assert len(sizes) <= 1
    shape = [-1 for _ in list_of_tensors[0].shape]
    shape[dim] = max_size
    expanded_tensors = [tensor.expand(*shape) for tensor in list_of_tensors]
    return expanded_tensors


def expand_for_broadcast_list(list_of_lists):
    sizes = {len(_list) for _list in list_of_lists}
    max_size = max(sizes)
    sizes.discard(1)
    assert len(sizes) <= 1
    expanded_lists = [
        _list if len(_list) == max_size else [_list[0] for _ in range(max_size)]
        for _list in list_of_lists
    ]
    return expanded_lists


class Metrics:
    def __init__(self, prefix=""):
        self.metrics = {}
        self.prefix = prefix

    def log(self, key, value, step=None, log_to_wandb=False):
        key = self.prefix + key
        if key in self.metrics:
            self.metrics[key].append(value)
        else:
            self.metrics[key] = [value]
        if log_to_wandb:
            assert step is not None
            wandb.log(dict({key: value}), step=step)

    def get_combined(self, fn, prefix="", step=None, log_to_wandb=False):
        average_metrics = {}
        for key, values in self.metrics.items():
            average_metrics[f"{prefix}{key}"] = fn(values)
        if log_to_wandb:
            assert step is not None
            wandb.log(average_metrics, step=step)
        return average_metrics

    def get_avg(self, prefix="avg/", step=None, log_to_wandb=False):
        return self.get_combined(
            fn=list_avg, prefix=prefix, step=step, log_to_wandb=log_to_wandb
        )

    def get_max(self, prefix="max/", step=None, log_to_wandb=False):
        return self.get_combined(
            fn=max, prefix=prefix, step=step, log_to_wandb=log_to_wandb
        )

    def get_min(self, prefix="min/", step=None, log_to_wandb=False):
        return self.get_combined(
            fn=min, prefix=prefix, step=step, log_to_wandb=log_to_wandb
        )

    def log_dict(self, metrics_dict, step=None, log_to_wandb=False):
        for key, value in metrics_dict.items():
            self.log(key=key, value=value, step=step, log_to_wandb=log_to_wandb)

    def reset(self):
        self.metrics = {}


def read_csv_file(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        entries = [row[0] for row in reader]
    return entries


def llm_loader(llm_params, verbose=False):
    tqdm.write(
        f" Loading model: {llm_params.model_name} from {llm_params.checkpoint}...",
    )
    mem_before = get_total_allocated_memory()
    if llm_params.dtype == "float32":
        dtype = torch.float32
    elif llm_params.dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"Cannot load model with dtype {llm_params.dtype}")
    if llm_params.checkpoint == "stas/tiny-random-llama-2":
        tokenizer = AutoTokenizer.from_pretrained(
            llm_params.checkpoint,
            padding_side="right",
        )
        model = AutoModelForCausalLM.from_pretrained(
            llm_params.checkpoint, torch_dtype=dtype
        ).to(llm_params.device)
    else:
        use_fast = "pythia" in llm_params.checkpoint
        tokenizer = AutoTokenizer.from_pretrained(
            llm_params.checkpoint,
            model_max_length=1024,
            padding_side="right",
            use_fast=use_fast,
            legacy=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            llm_params.checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            device_map=llm_params.device,
        )
    mem_after = get_total_allocated_memory()

    if verbose:
        tqdm.write(f" Loaded model: {model}")
    tqdm.write(
        f" Mem usage model: {mem_after - mem_before:.2f} GB | Total Mem usage: {get_total_allocated_memory():.2f} GB",
    )

    embedding_matrix = get_embedding_matrix(model).to(llm_params.device)

    if llm_params.freeze:
        tqdm.write(" Freezing model...")
        for k, v in model.named_parameters():
            v.requires_grad = False

    if llm_params.lora_params is not None:
        if llm_params.lora_params.warmstart:
            tqdm.write(
                f" Loading LoRA checkpoint: {llm_params.lora_params.lora_checkpoint}",
            )
            model = PeftModel.from_pretrained(
                model,
                llm_params.lora_params.lora_checkpoint,
                is_trainable=not llm_params.freeze,
            )
        else:
            tqdm.write(" Transforming to LoRA model...")
            lora_config_dct = dict(llm_params.lora_params.lora_config)
            lora_config_dct["target_modules"] = [
                m for m in llm_params.lora_params.lora_config["target_modules"]
            ]
            lora_config = LoraConfig(**lora_config_dct)
            model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)
    return model, tokenizer, embedding_matrix


def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif (
        isinstance(model, LlamaForCausalLM)
        or isinstance(model, MistralForCausalLM)
        or isinstance(model, GemmaForCausalLM)
    ):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    elif isinstance(model, FalconForCausalLM):
        return model.transformer.word_embeddings.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def load_csv(pth):
    with open(pth) as f:
        dict_reader = csv.DictReader(f)
        csv_list = list(dict_reader)
    return csv_list


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class AdvPromptDataset(Dataset):
    def __init__(self, data_pth):
        self.dataset = load_csv(data_pth)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class AugmentDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, augment_target, shuffle):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        self.effective_dataset_size = len(self.dataset)
        self.aufgment_target = augment_target

        self.process_fn = lambda s: s.replace("Sure, here is", "Sure, here's")
        self.process_fn2 = lambda s: s.replace("Sure, h", "H")

    def __iter__(self):
        for batch in super(AugmentDataLoader, self).__iter__():
            if self.aufgment_target:
                targets = []
                for target in batch["target"]:
                    if np.random.random() < 0.5:
                        target = self.process_fn(target)
                    if np.random.random() < 0.5:
                        target = self.process_fn2(target)
                    targets.append(target)
                batch["target"] = targets
            yield batch


def get_dataloader(data_pth, batch_size, shuffle, augment_target):
    dataset = AdvPromptDataset(data_pth=data_pth)
    dataloader = AugmentDataLoader(
        dataset, augment_target=augment_target, shuffle=shuffle, batch_size=batch_size
    )
    return dataloader


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
