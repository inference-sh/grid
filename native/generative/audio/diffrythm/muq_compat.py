"""Keep muq 0.1.0 working on transformers 5.

MuQ builds transformers' Wav2Vec2ConformerEncoder from an EasyDict config and
reads out["hidden_states"]. transformers 5 reads config._attn_implementation,
which an EasyDict lacks, and its encoder no longer returns hidden states (the
top-level model records them). This restores the transformers 4 encoder
output on top of the transformers 5 layers. Import before loading MuQMuLan.
"""
import transformers

if int(transformers.__version__.split(".")[0]) >= 5:
    from transformers.masking_utils import create_bidirectional_mask
    from transformers.modeling_outputs import BaseModelOutput
    from transformers.models.wav2vec2_conformer import modeling_wav2vec2_conformer as _w2v

    def _forward(self, hidden_states, attention_mask=None, output_hidden_states=False, **kwargs):
        if getattr(self.config, "_attn_implementation", None) is None:
            self.config._attn_implementation = "eager"
        kwargs.pop("output_attentions", None)
        kwargs.pop("return_dict", None)

        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0.0
        attention_mask = create_bidirectional_mask(
            config=self.config, inputs_embeds=hidden_states, attention_mask=attention_mask
        )

        hidden_states = self.dropout(hidden_states)
        relative_position_embeddings = (
            self.embed_positions(hidden_states) if self.embed_positions is not None else None
        )

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                relative_position_embeddings=relative_position_embeddings,
                **kwargs,
            )

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    _w2v.Wav2Vec2ConformerEncoder.forward = _forward
