
from typing import Callable, Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from .configuration_helpingai import HelpingAIConfig


@use_kernel_forward_from_hub("RMSNorm")
class HelpingAIRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HelpingAIRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class HelpingAISemanticEmotionReasoning(nn.Module):
    """
    Structured Emotional Reasoning (SER) layer for emotional understanding and processing.
    Maps emotions to semantic representations and provides contextual emotion analysis.
    """
    def __init__(self, config: HelpingAIConfig):
        super().__init__()
        self.config = config
        self.emotion_hidden_size = config.emotion_hidden_size
        self.hidden_size = config.hidden_size
        
        # Emotion detection and mapping
        self.emotion_detector = nn.Linear(self.hidden_size, self.emotion_hidden_size)
        self.emotion_mapper = nn.Linear(self.emotion_hidden_size, self.emotion_hidden_size)
        
        # Contextual emotion analysis
        self.emotion_context = nn.MultiheadAttention(
            embed_dim=self.emotion_hidden_size,
            num_heads=min(8, self.emotion_hidden_size // 64),
            batch_first=True
        )
        
        # Emotion classification heads
        self.primary_emotion = nn.Linear(self.emotion_hidden_size, 32)  # Primary emotions
        self.emotion_intensity = nn.Linear(self.emotion_hidden_size, 1)  # Intensity score
        self.emotion_valence = nn.Linear(self.emotion_hidden_size, 1)   # Positive/negative
        
        # Output projection
        self.emotion_output = nn.Linear(self.emotion_hidden_size, self.hidden_size)
        self.emotion_norm = HelpingAIRMSNorm(self.emotion_hidden_size, eps=config.rms_norm_eps)
        
        # Activation
        self.act_fn = ACT2FN[config.hidden_act]
        
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Detect emotional content
        emotion_features = self.act_fn(self.emotion_detector(hidden_states))
        emotion_mapped = self.emotion_mapper(emotion_features)
        emotion_mapped = self.emotion_norm(emotion_mapped)
        
        # Contextual emotion analysis
        emotion_context, attention_weights = self.emotion_context(
            emotion_mapped, emotion_mapped, emotion_mapped
        )
        
        # Emotion analysis outputs
        primary_emotions = self.primary_emotion(emotion_context)
        emotion_intensity = torch.sigmoid(self.emotion_intensity(emotion_context))
        emotion_valence = torch.tanh(self.emotion_valence(emotion_context))
        
        # Project back to hidden size
        emotion_output = self.emotion_output(emotion_context)
        
        # Emotion metadata
        emotion_metadata = {
            "primary_emotions": primary_emotions,
            "intensity": emotion_intensity,
            "valence": emotion_valence,
            "attention_weights": attention_weights
        }
        
        return emotion_output, emotion_metadata


class HelpingAIPerspectiveEmotionThreading(nn.Module):
    """
    Parallel Empathic Threads (PET) layer for multi-threaded emotional reasoning.
    Processes multiple perspective threads: relatable, supportive, motivational, analytical.
    """
    def __init__(self, config: HelpingAIConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.perspective_threads = config.perspective_threads
        self.thread_hidden_size = config.emotion_hidden_size
        
        # Thread-specific processors
        self.thread_projections = nn.ModuleList([
            nn.Linear(self.hidden_size, self.thread_hidden_size)
            for _ in range(self.perspective_threads)
        ])
        
        # Thread names for interpretability
        self.thread_names = ["relatable", "supportive", "motivational", "analytical"][:self.perspective_threads]
        
        # Cross-thread attention for perspective integration
        self.cross_thread_attention = nn.MultiheadAttention(
            embed_dim=self.thread_hidden_size,
            num_heads=min(4, self.thread_hidden_size // 64),
            batch_first=True
        )
        
        # Thread-specific processing layers
        self.thread_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.thread_hidden_size, self.thread_hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.thread_hidden_size * 2, self.thread_hidden_size),
                HelpingAIRMSNorm(self.thread_hidden_size, eps=config.rms_norm_eps)
            )
            for _ in range(self.perspective_threads)
        ])
        
        # Output integration
        self.thread_combiner = nn.Linear(
            self.thread_hidden_size * self.perspective_threads, 
            self.hidden_size
        )
        
        # Thread importance weighting
        self.thread_weights = nn.Parameter(torch.ones(self.perspective_threads))
        
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Process each perspective thread
        thread_outputs = []
        thread_metadata = {}
        
        for i, (projection, processor, thread_name) in enumerate(
            zip(self.thread_projections, self.thread_processors, self.thread_names)
        ):
            # Project to thread space
            thread_input = projection(hidden_states)
            
            # Process thread-specific perspective
            thread_output = processor(thread_input)
            thread_outputs.append(thread_output)
            
            # Store thread metadata
            thread_metadata[f"{thread_name}_activation"] = torch.mean(torch.abs(thread_output))
        
        # Stack threads for cross-thread attention
        stacked_threads = torch.stack(thread_outputs, dim=2)  # [batch, seq_len, num_threads, hidden]
        stacked_threads = stacked_threads.reshape(batch_size * seq_len, self.perspective_threads, self.thread_hidden_size)
        
        # Cross-thread attention for perspective integration
        integrated_threads, cross_attention = self.cross_thread_attention(
            stacked_threads, stacked_threads, stacked_threads
        )
        
        # Apply thread importance weighting
        thread_weights_normalized = torch.softmax(self.thread_weights, dim=0)
        weighted_threads = integrated_threads * thread_weights_normalized.unsqueeze(0).unsqueeze(-1)
        
        # Combine threads - use reshape instead of view for memory layout compatibility
        combined_threads = weighted_threads.reshape(batch_size, seq_len, -1)
        final_output = self.thread_combiner(combined_threads)
        
        # Thread metadata
        thread_metadata.update({
            "thread_weights": thread_weights_normalized,
            "cross_attention": cross_attention,
            "thread_activations": {
                name: torch.mean(output) for name, output in zip(self.thread_names, thread_outputs)
            }
        })
        
        return final_output, thread_metadata


class HelpingAIMultiStageThinking(nn.Module):
    """
    Multi-stage thinking module for internal reasoning and reflection processes.
    Implements cascaded thinking stages with simplified feedback loops.
    """
    def __init__(self, config: HelpingAIConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.thinking_stages = config.num_thinking_stages
        self.thinking_depth = config.thinking_depth
        
        # Thinking stage processors
        self.thinking_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                HelpingAIRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            )
            for _ in range(self.thinking_stages)
        ])
        
        # Simple reflection mechanism without complex attention
        self.reflection_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size)
            for _ in range(self.thinking_stages - 1)
        ])
        
        # Stage transition gates
        self.stage_gates = nn.ModuleList([
            nn.Linear(self.hidden_size, 1) for _ in range(self.thinking_stages - 1)
        ])
        
        # Thinking combination weights
        self.stage_combiner = nn.Linear(self.thinking_stages * self.hidden_size, self.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch_size, seq_len, _ = hidden_states.shape
        thinking_outputs = []
        thinking_metadata = {}
        
        current_thought = hidden_states
        
        # Multi-stage thinking process
        for stage_idx, stage_processor in enumerate(self.thinking_layers):
            # Process current thinking stage
            current_thought = stage_processor(current_thought)
            
            # Store stage output
            thinking_outputs.append(current_thought)
            thinking_metadata[f"stage_{stage_idx}_activation"] = torch.mean(torch.abs(current_thought)).item()
            
            # Apply reflection if not the last stage
            if stage_idx < self.thinking_stages - 1:
                # Simple reflection mechanism
                reflection = self.reflection_layers[stage_idx](current_thought)
                current_thought = current_thought + 0.1 * reflection  # Small reflection influence
                
                # Stage transition gating
                gate_weight = torch.sigmoid(self.stage_gates[stage_idx](current_thought))
                current_thought = gate_weight * current_thought + (1 - gate_weight) * hidden_states
        
        # Combine all thinking stages
        all_thoughts = torch.cat(thinking_outputs, dim=-1)  # Concatenate along hidden dimension
        final_thought = self.stage_combiner(all_thoughts)
        
        thinking_metadata["stage_contributions"] = [
            torch.mean(torch.abs(output)).item() for output in thinking_outputs
        ]
        
        return final_thought, thinking_metadata


class HelpingAIMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        
        # Enhanced MLP with thinking modules
        if hasattr(config, 'use_emotional_reasoning') and config.use_emotional_reasoning:
            self.thinking_module = HelpingAIMultiStageThinking(config)
            self.use_thinking = True
        else:
            self.use_thinking = False
            
        # Reasoning temperature for controlled generation
        self.reasoning_temperature = getattr(config, 'reasoning_temperature', 1.0)

    def forward(self, x):
        # Standard MLP forward pass
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        
        # Apply multi-stage thinking if enabled
        if self.use_thinking:
            thinking_output, thinking_metadata = self.thinking_module(down_proj)
            # Apply reasoning temperature
            down_proj = down_proj + (thinking_output * self.reasoning_temperature)
            
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class HelpingAIAttention(nn.Module):
    """Multi-headed attention with specialized emotional and empathetic reasoning capabilities"""

    def __init__(self, config: HelpingAIConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = HelpingAIRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = HelpingAIRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        # Enhanced emotional and empathetic attention
        if hasattr(config, 'use_emotional_reasoning') and config.use_emotional_reasoning:
            self.num_emotion_heads = getattr(config, 'num_emotion_heads', 4)
            self.empathy_scaling_factor = getattr(config, 'empathy_scaling_factor', 1.2)
            
            # Specialized emotion attention projections
            self.emotion_q_proj = nn.Linear(config.hidden_size, self.num_emotion_heads * self.head_dim, bias=False)
            self.emotion_k_proj = nn.Linear(config.hidden_size, self.num_emotion_heads * self.head_dim, bias=False)
            self.emotion_v_proj = nn.Linear(config.hidden_size, self.num_emotion_heads * self.head_dim, bias=False)
            
            # Empathy enhancement layer
            self.empathy_enhancer = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, config.num_attention_heads),
                nn.Softmax(dim=-1)
            )
            
            self.use_emotional_attention = True
        else:
            self.use_emotional_attention = False

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Standard attention processing
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Enhanced emotional attention processing
        if self.use_emotional_attention:
            # Compute empathy weights
            empathy_weights = self.empathy_enhancer(hidden_states.mean(dim=1))  # [batch, num_heads]
            
            # Emotional query, key, value computation
            emotion_query = self.emotion_q_proj(hidden_states).view(*input_shape, self.num_emotion_heads, self.head_dim).transpose(1, 2)
            emotion_key = self.emotion_k_proj(hidden_states).view(*input_shape, self.num_emotion_heads, self.head_dim).transpose(1, 2)
            emotion_value = self.emotion_v_proj(hidden_states).view(*input_shape, self.num_emotion_heads, self.head_dim).transpose(1, 2)
            
            # Apply rotary embeddings to emotional attention
            emotion_query, emotion_key = apply_rotary_pos_emb(emotion_query, emotion_key, cos, sin)
            
            # Emotional attention computation
            emotion_scaling = (self.head_dim ** -0.5) * self.empathy_scaling_factor
            emotion_attn_weights = torch.matmul(emotion_query, emotion_key.transpose(2, 3)) * emotion_scaling
            
            if attention_mask is not None:
                emotion_causal_mask = attention_mask[:, :, :, :emotion_key.shape[-2]]
                emotion_attn_weights = emotion_attn_weights + emotion_causal_mask
            
            emotion_attn_weights = nn.functional.softmax(emotion_attn_weights, dim=-1, dtype=torch.float32).to(emotion_query.dtype)
            emotion_output = torch.matmul(emotion_attn_weights, emotion_value)
            
            # Integrate emotional attention with standard attention
            # Pad or truncate emotional attention to match standard attention heads
            if self.num_emotion_heads < self.config.num_attention_heads:
                padding_heads = self.config.num_attention_heads - self.num_emotion_heads
                emotion_padding = torch.zeros(
                    *emotion_output.shape[:-3], padding_heads, *emotion_output.shape[-2:],
                    device=emotion_output.device, dtype=emotion_output.dtype
                )
                emotion_output = torch.cat([emotion_output, emotion_padding], dim=1)

        # Standard attention computation
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # Blend standard and emotional attention if emotional reasoning is enabled
        if self.use_emotional_attention:
            # For now, use a simplified approach - just apply empathy scaling
            # This avoids the complex tensor dimension matching issues
            batch_size, num_heads, seq_len, head_dim = attn_output.shape
            
            # Get average empathy weight per batch
            empathy_scale = torch.mean(empathy_weights, dim=1, keepdim=True)  # [batch, 1]
            empathy_scale = empathy_scale.view(batch_size, 1, 1, 1)  # [batch, 1, 1, 1]
            empathy_scale = empathy_scale.expand(batch_size, num_heads, seq_len, head_dim)
            
            # Apply empathy scaling to attention output
            attn_output = attn_output * (1.0 + empathy_scale * 0.1)  # Small empathy influence

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class HelpingAIDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: HelpingAIConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = HelpingAIAttention(config=config, layer_idx=layer_idx)
        self.mlp = HelpingAIMLP(config)
        self.input_layernorm = HelpingAIRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HelpingAIRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

        # Enhanced reasoning layers
        if hasattr(config, 'use_emotional_reasoning') and config.use_emotional_reasoning:
            self.ser_layer = HelpingAISemanticEmotionReasoning(config)
            self.use_ser = True
        else:
            self.use_ser = False
            
        if hasattr(config, 'use_perspective_threading') and config.use_perspective_threading:
            self.pet_layer = HelpingAIPerspectiveEmotionThreading(config)
            self.use_pet = True
        else:
            self.use_pet = False
            
        # Reasoning integration layers
        if self.use_ser or self.use_pet:
            self.reasoning_norm = HelpingAIRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.reasoning_gate = nn.Linear(config.hidden_size, 1)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, attention_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Enhanced reasoning processing
        reasoning_outputs = []
        reasoning_metadata = {}
        
        if self.use_ser:
            # Semantic Emotion Reasoning
            ser_output, ser_meta = self.ser_layer(hidden_states)
            reasoning_outputs.append(ser_output)
            reasoning_metadata['ser'] = ser_meta
            
        if self.use_pet:
            # Perspective Emotion Threading
            pet_output, pet_meta = self.pet_layer(hidden_states)
            reasoning_outputs.append(pet_output)
            reasoning_metadata['pet'] = pet_meta
        
        # Integrate reasoning outputs if any
        if reasoning_outputs:
            # Combine reasoning outputs
            combined_reasoning = torch.stack(reasoning_outputs, dim=0).mean(dim=0)
            combined_reasoning = self.reasoning_norm(combined_reasoning)
            
            # Apply gating to control reasoning influence
            reasoning_gate = torch.sigmoid(self.reasoning_gate(hidden_states))
            hidden_states = hidden_states + (reasoning_gate * combined_reasoning)

        # Fully Connected (MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Store reasoning metadata for analysis (optional)
        if hasattr(hidden_states, '_reasoning_metadata'):
            hidden_states._reasoning_metadata = reasoning_metadata
            
        return hidden_states


@auto_docstring
class HelpingAIPreTrainedModel(PreTrainedModel):
    config: HelpingAIConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HelpingAIDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": HelpingAIDecoderLayer,
        "attentions": HelpingAIAttention,
    }


class HelpingAIRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: HelpingAIConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class HelpingAIModel(HelpingAIPreTrainedModel):
    def __init__(self, config: HelpingAIConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HelpingAIDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HelpingAIRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HelpingAIRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class HelpingAIForCausalLM(HelpingAIPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = HelpingAIModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Enhanced structured output support
        if hasattr(config, 'structured_output_vocab_size') and config.structured_output_vocab_size > 0:
            self.structured_vocab_size = config.structured_output_vocab_size
            self.use_structured_output = True
            # Build structured head depending on config.structured_head_type
            head_type = getattr(config, 'structured_head_type', 'linear')
            act_name = getattr(config, 'structured_head_activation', 'gelu')
            act_layer = nn.GELU() if act_name == 'gelu' else nn.ReLU()
            hidden_dim = getattr(config, 'structured_head_hidden_dim', None)
            if head_type == 'mlp_v1':
                if hidden_dim is None:
                    # Heuristic: pick hidden so params roughly ~ (in+out)*hidden ~ 50M default
                    denom = config.hidden_size + self.structured_vocab_size
                    target = 50_000_000
                    hidden_dim = max(128, int(target / max(1, denom)))
                self.structured_lm_head = nn.Sequential(
                    nn.Linear(config.hidden_size, hidden_dim, bias=True),
                    act_layer,
                    nn.Linear(hidden_dim, self.structured_vocab_size, bias=True),
                )
            else:
                self.structured_lm_head = nn.Linear(config.hidden_size, self.structured_vocab_size, bias=False)

            # Special token embeddings for structured reasoning
            self.structured_token_embeddings = nn.Embedding(self.structured_vocab_size, config.hidden_size)

            # Reasoning mode classifier
            self.reasoning_mode_classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, 4),  # think, ser, pet, normal
                nn.Softmax(dim=-1)
            )
        else:
            self.use_structured_output = False

        # Optional speech output head (predict mel-spectrogram frames)
        self.use_speech_output = getattr(config, "use_speech_output", False)
        if self.use_speech_output:
            self.speech_num_mels = getattr(config, "speech_num_mels", 80)
            self.speech_upsample_factor = getattr(config, "speech_upsample_factor", 1)
            hidden_dim = getattr(config, "speech_head_hidden_dim", None)
            if hidden_dim is None:
                hidden_dim = config.hidden_size // 2
            # Projector from hidden_size -> hidden_dim -> mel bins
            self.speech_proj = nn.Sequential(
                nn.Linear(config.hidden_size, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.speech_num_mels),
            )
            self.speech_loss_type = getattr(config, "speech_loss_type", "l1")

        # Initialize weights and apply final processing
        self.post_init()
        # Register a load-state pre-hook so older checkpoints with saved structured head metadata can be restored
        self._register_load_state_dict_pre_hook(self._structured_head_migration_hook, with_module=True)

    # --- Structured head migration logic ---
    def _structured_head_migration_hook(self, module, state_dict, prefix, *args, **kwargs):
        """Detect mismatched structured head weights and rebuild head if necessary.

        Supports migration from legacy linear -> MLP (saved externally) when config specifies mlp_v1
        but checkpoint only has linear weights OR when state_dict contains sequential weights not
        matching current module shape.
        """
        if not getattr(self, 'use_structured_output', False):
            return
        cfg = self.config
        desired_type = getattr(cfg, 'structured_head_type', 'linear')
        if desired_type != 'mlp_v1':
            return
        # Current module may already be Sequential; if so, nothing to do
        if isinstance(self.structured_lm_head, nn.Sequential):
            return
        # Look for legacy linear weight key
        w_key = prefix + 'structured_lm_head.weight'
        b_key = prefix + 'structured_lm_head.bias'
        if w_key in state_dict and not any(k.startswith(prefix + 'structured_lm_head.0.') for k in state_dict.keys()):
            # Need to rebuild to MLP form
            hidden_dim = getattr(cfg, 'structured_head_hidden_dim', None)
            if hidden_dim is None:
                denom = cfg.hidden_size + cfg.structured_output_vocab_size
                target = 50_000_000
                hidden_dim = max(128, int(target / max(1, denom)))
            act_name = getattr(cfg, 'structured_head_activation', 'gelu')
            act_layer = nn.GELU() if act_name == 'gelu' else nn.ReLU()
            new_head = nn.Sequential(
                nn.Linear(cfg.hidden_size, hidden_dim, bias=True),
                act_layer,
                nn.Linear(hidden_dim, cfg.structured_output_vocab_size, bias=True),
            )
            self.structured_lm_head = new_head.to(next(self.parameters()).device)
            # Legacy linear weights can't be mapped meaningfully; leave new head randomly inited.
            # Remove old unmatched keys so load_state_dict won't warn.
            state_dict.pop(w_key, None)
            state_dict.pop(b_key, None)
        # If partial sequential weights exist but shape mismatch, rely on normal strict=False upstream behavior

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
        
    def get_reasoning_mode_probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get probabilities for different reasoning modes: think, ser, pet, normal"""
        if self.use_structured_output:
            # Use the last token's hidden state for mode classification
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
            mode_probs = self.reasoning_mode_classifier(last_hidden)
            return mode_probs
        return None

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    # Optional supervision for speech frames: float tensor [B, T_frames, n_mels]
        speech_targets: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_reasoning_metadata: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Enhanced HelpingAI forward pass with structured reasoning and speech supervision support.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
                Pre-computed hidden-states that can be used to speed up autoregressive decoding.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Embedded representation of the input tokens. Can be used instead of `input_ids`.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss.
            speech_targets (`torch.FloatTensor` of shape `(batch_size, T_frames, n_mels)`, *optional*):
                Optional ground-truth mel-spectrogram frames for speech head supervision. Used only if `use_speech_output` is enabled.
                - `batch_size`: number of samples in the batch
                - `T_frames`: number of mel frames (may differ from token count)
                - `n_mels`: number of mel bins (should match config.speech_num_mels)
            use_cache (`bool`, *optional*):
                If set to `True`, past key values are returned and can be used to speed up decoding.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input tokens in the sequence.
            logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to 0):
                Number of logits to keep from the end of the sequence.
            return_reasoning_metadata (`bool`, *optional*, defaults to `False`):
                Whether to return reasoning metadata including SER and PET analysis for structured reasoning.

        Returns:
            `CausalLMOutputWithPast`: Model output containing logits, past key values, and optional reasoning metadata.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, HelpingAIForCausalLM

        >>> model = HelpingAIForCausalLM.from_pretrained("HelpingAI/HelpingAI-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("HelpingAI/HelpingAI-8B")

        >>> # Standard generation
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

        >>> # Structured reasoning generation  
        >>> outputs = model(inputs.input_ids, return_reasoning_metadata=True)
        >>> reasoning_modes = model.get_reasoning_mode_probabilities(outputs.hidden_states)

        >>> # Speech head supervision
        >>> mel_targets = torch.randn(batch_size, T_frames, n_mels)
        >>> outputs = model(inputs.input_ids, speech_targets=mel_targets)
        ```
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        
        # Standard language modeling head
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # Enhanced structured output logits
        structured_logits = None
        reasoning_mode_probs = None
        if self.use_structured_output:
            structured_logits = self.structured_lm_head(hidden_states[:, slice_indices, :])
            reasoning_mode_probs = self.get_reasoning_mode_probabilities(hidden_states)

        # Speech output prediction
        speech_mels = None
        if self.use_speech_output:
            token_level = hidden_states  # [B, T_tok, H]
            # Simple temporal upsampling by repetition to approximate frame rate
            if getattr(self, "speech_upsample_factor", 1) > 1:
                token_level = token_level.repeat_interleave(self.speech_upsample_factor, dim=1)
            # Project to mel bins per (upsampled) time-step
            speech_mels = self.speech_proj(token_level)  # [B, T_frames, n_mels]

        loss = None
        if labels is not None:
            # Standard loss computation
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            
            # Add structured output loss if applicable
            if self.use_structured_output and structured_logits is not None:
                # Additional loss term for structured reasoning (if labels include structured tokens)
                structured_loss_weight = 0.1  # Weight for structured output loss
                structured_loss = self.loss_function(
                    logits=structured_logits, 
                    labels=labels, 
                    vocab_size=self.structured_vocab_size, 
                    **kwargs
                )
                loss = loss + (structured_loss_weight * structured_loss)

        # Add speech supervision if provided
        if self.use_speech_output and speech_targets is not None:
            # Ensure time dimension alignment by trimming or padding speech_mels to targets
            B, T_pred, M = speech_mels.shape
            B2, T_tgt, M2 = speech_targets.shape
            if B != B2 or M != M2:
                raise ValueError("speech_targets shape mismatch. Expected [B, T, n_mels] with same B and n_mels as model output.")
            if T_pred > T_tgt:
                speech_mels_aligned = speech_mels[:, :T_tgt, :]
            elif T_pred < T_tgt:
                pad = torch.zeros(B, T_tgt - T_pred, M, device=speech_mels.device, dtype=speech_mels.dtype)
                speech_mels_aligned = torch.cat([speech_mels, pad], dim=1)
            else:
                speech_mels_aligned = speech_mels

            if self.speech_loss_type == "mse":
                speech_loss = nn.functional.mse_loss(speech_mels_aligned, speech_targets)
            else:
                speech_loss = nn.functional.l1_loss(speech_mels_aligned, speech_targets)
            loss = speech_loss if loss is None else (loss + speech_loss)

        # Prepare output with enhanced reasoning metadata
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        # Add custom attributes for reasoning
        if return_reasoning_metadata and self.use_structured_output:
            output.structured_logits = structured_logits
            output.reasoning_mode_probabilities = reasoning_mode_probs
        if self.use_speech_output:
            output.speech_mels = speech_mels
            
        return output


class HelpingAIForSequenceClassification(GenericForSequenceClassification, HelpingAIPreTrainedModel):
    pass


class HelpingAIForTokenClassification(GenericForTokenClassification, HelpingAIPreTrainedModel):
    pass


class HelpingAIForQuestionAnswering(GenericForQuestionAnswering, HelpingAIPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "HelpingAIForCausalLM",
    "HelpingAIForQuestionAnswering",
    "HelpingAIPreTrainedModel",
    "HelpingAIModel",
    "HelpingAIForSequenceClassification",
    "HelpingAIForTokenClassification",
]



HelpingAIConfig.register_for_auto_class()
HelpingAIForCausalLM.register_for_auto_class("AutoModelForCausalLM")