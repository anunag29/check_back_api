from transformers import RobertaTokenizerFast
import string
from typing import Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import TrOCRForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class CustomTrOCRForCausalLM(TrOCRForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.non_digit_token_ids = self.get_non_digit_token_ids()
        self.logits_storage = []
        self.input_ids_storage = []

    def get_non_digit_token_ids(self):
      tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/trocr-base-handwritten")
      non_digit_token_ids = []
      special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id}

      for token_id in range(tokenizer.vocab_size):
          if token_id not in special_tokens:
              token = tokenizer.decode([token_id]).replace(" ", "")
              # Check if the token contains any non-digit character, except for tokens containing "-"
              if not token.isdigit():
                  non_digit_token_ids.append(token_id)

      return non_digit_token_ids

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        self.logits_storage.append(logits)
        self.input_ids_storage.append(input_ids)

        mask = torch.zeros_like(logits).to(logits.device)
        mask[:, :, self.non_digit_token_ids] = float('-inf')

        logits = logits + mask
        outputs.logits = logits

        # loss = None
        # if outputs.loss is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        #     outputs.loss = loss

        return outputs