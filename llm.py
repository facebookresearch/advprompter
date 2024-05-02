# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from contextlib import nullcontext

import torch
import torch.nn as nn

from sequence import Seq, MergedSeq, msg_to_seq
from utils import (
    ReturnStruct,
    autocast_decorator,
    compute_perplexity,
    get_nonascii_toks,
    llm_loader,
    loss_seqs,
)


class LLM(nn.Module):
    def __init__(self, params, verbose=False) -> None:
        super().__init__()
        self.params = params
        self.verbose = verbose

        self.model, self.tokenizer, self.embedding_matrix = llm_loader(
            llm_params=params.llm_params, verbose=verbose
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                # TODO: This is a hack I added because Falcon-7b-isntruct doe snot have a pad token
                # We might run into trouble here because the Seq class will automatically treat any eos_token as a pad_token and set the padding mask to 0 for this token
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self.params.llm_params.device
        if self.params.allow_non_ascii:
            self.disallowed_ids = None
        else:
            self.disallowed_ids = get_nonascii_toks(self.tokenizer, device=self.device)

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path, save_embedding_layers=True)

    def model_forward(self, query_seq, use_basemodel=False):
        # reorder such that all masked tokens are on the left
        mask = query_seq.mask
        sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)

        with self.model.disable_adapter() if use_basemodel else nullcontext():
            if query_seq.is_hard:
                ids = query_seq.ids
                sorted_ids = ids.gather(1, indices)
                shifted_sorted_pred_logits = self.model(
                    input_ids=sorted_ids, attention_mask=sorted_mask
                ).logits
            else:
                embeds = query_seq.get_embed(self.embedding_matrix)
                indices_extended = indices[:, :, None].repeat(1, 1, embeds.shape[-1])
                sorted_embeds = embeds.gather(1, indices_extended)
                shifted_sorted_pred_logits = self.model(
                    inputs_embeds=sorted_embeds, attention_mask=sorted_mask
                ).logits

        # reverse the sort to get the original order (also account for the shift)
        dummy_pred_logits = torch.zeros_like(shifted_sorted_pred_logits[:, :1, :])
        sorted_pred_logits = torch.cat(
            [dummy_pred_logits, shifted_sorted_pred_logits[:, :-1, :]], dim=1
        )
        reverse_indices = indices.argsort(dim=1)
        reverse_indices_extended = reverse_indices[:, :, None].repeat(
            1, 1, sorted_pred_logits.shape[-1]
        )
        shifted_pred_logits = sorted_pred_logits.gather(1, reverse_indices_extended)
        pred_logits = torch.cat(
            [shifted_pred_logits[:, 1:, :], shifted_sorted_pred_logits[:, -1:, :]],
            dim=1,
        )

        if self.disallowed_ids is not None:
            pred_logits[:, :, self.disallowed_ids] = -1e10
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            for i in range(pred_logits.shape[0]):
                if torch.isnan(pred_logits[i]).any():
                    print(i, "-th logits..........", pred_logits[i])
                    print("shifted_sorted_pred_logits", shifted_sorted_pred_logits[i])
                    print("ids........", ids[i])
                    print("sorted_masks.......", sorted_mask[i])
                    print("sorted_ids", sorted_ids[i])
            raise RuntimeError(f"NaN in pred_logits: {pred_logits}")
        new_mask = torch.ones_like(mask)
        new_mask[:, :-1] = mask[:, 1:]
        seq = Seq(
            logits=pred_logits,
            mask=new_mask,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        return seq

    @autocast_decorator
    def compute_pred_loss_teacher_forced(self, loss_params, label=None, **kwargs):
        gen_seqs = self.generate_teacher_forced(**kwargs)
        if label is None:
            label = gen_seqs.response_teacher
        loss_return = loss_seqs(gen_seqs.response_dist, label, **loss_params)

        pred_loss_return = ReturnStruct(
            loss=loss_return.loss,
            loss_masked=loss_return.loss_masked,
            loss_batch=loss_return.loss_batch,
            query=gen_seqs.query,
            response_teacher=gen_seqs.response_teacher,
            response_dist=gen_seqs.response_dist,
            label=label,
            perplexity=gen_seqs.perplexity,
            perplexity_per_token_masked=gen_seqs.perplexity_per_token_masked,
        )
        return pred_loss_return

    @autocast_decorator
    def generate_teacher_forced(
        self, key, detach_query=False, use_basemodel=False, **context
    ):
        query_seq, response_teacher_seq = self.prepare_prompt(
            context, up_to_key=key, return_key_seq=True
        )
        assert not response_teacher_seq.is_empty
        full_query_seq = MergedSeq([query_seq, response_teacher_seq])
        if detach_query:
            full_query_seq = full_query_seq.clone().detach()

        pred_full_query_seq = self.model_forward(
            query_seq=full_query_seq, use_basemodel=use_basemodel
        )
        response_dist_seq = pred_full_query_seq[
            :, -response_teacher_seq.seq_len - 1 : -1
        ]
        perplexity, perplexity_per_token_masked = compute_perplexity(
            id_seq=response_teacher_seq, likelihood_seq=response_dist_seq
        )

        return_seqs = ReturnStruct(
            query=query_seq,
            response_teacher=response_teacher_seq,
            response_dist=response_dist_seq,
            perplexity=perplexity,
            perplexity_per_token_masked=perplexity_per_token_masked,
        )
        return return_seqs

    def get_next_token(self, key, use_basemodel=False, **context):
        query_seq, key_seq = self.prepare_prompt(
            context, up_to_key=key, return_key_seq=True
        )
        full_query_seq = MergedSeq([query_seq, key_seq])

        pred_dist_seq = self.model_forward(
            query_seq=full_query_seq, use_basemodel=use_basemodel
        )
        next_dist_seq = pred_dist_seq[:, -1:]

        return_seqs = ReturnStruct(query=full_query_seq, response_dist=next_dist_seq)
        return return_seqs

    def generate_autoregressive(
        self, key, use_basemodel=False, max_new_tokens=None, **context
    ):
        query_seq = self.prepare_prompt(context, up_to_key=key)

        mask = query_seq.mask
        ids = query_seq.ids
        sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)
        sorted_ids = ids.gather(1, indices)

        generation_config = self.model.generation_config
        if self.disallowed_ids is not None:
            generation_config.suppress_tokens = self.disallowed_ids.tolist()
        generation_config.renormalize_logits = True

        if max_new_tokens is None:
            max_new_tokens = self.params.gen_params.max_new_tokens

        gen_params = dict(self.params.gen_params)
        gen_params["max_new_tokens"] = max_new_tokens

        with self.model.disable_adapter() if use_basemodel else nullcontext():
            output = self.model.generate(
                input_ids=sorted_ids,
                attention_mask=sorted_mask,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                **gen_params,
            )

        output_ids = output.sequences[:, ids.shape[1] :]

        response_sample_seq = Seq(
            ids=output_ids, tokenizer=self.tokenizer, device=self.device
        )

        return_seqs = ReturnStruct(
            query=query_seq,
            response_sample=response_sample_seq,
        )
        return return_seqs

    def prepare_prompt(self, context, up_to_key=None, return_key_seq=False):
        seqs = []
        for msg_dct in self.params.prompt_manager.prompt_template:
            if (
                up_to_key is not None
                and up_to_key == msg_dct.key
                and not return_key_seq
            ):
                break
            seq = msg_to_seq(
                msg=msg_dct.msg,
                tokenizer=self.tokenizer,
                device=self.device,
                context=context,
            )
            if up_to_key is not None and up_to_key == msg_dct.key and return_key_seq:
                break
            seqs.append(seq)

        merged_prompt_seq = MergedSeq(seqs)
        if return_key_seq:
            return merged_prompt_seq, seq
        else:
            return merged_prompt_seq
