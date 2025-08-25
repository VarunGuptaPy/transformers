from transformers.configuration_utils import PretrainedConfig


class HelpingAIConfig(PretrainedConfig):
    model_type = "helpingai"

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048,
        layer_norm_epsilon=1e-5,
        hidden_act="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        # Structured output head
        use_structured_output=True,
        structured_output_vocab_size=2,
        # Speech head
        use_speech_output=False,
        speech_num_mels=80,
        speech_head_hidden_dim=1024,
        speech_upsample_factor=1,
        speech_loss_type="l1",
        # Misc
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_epsilon = layer_norm_epsilon
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range

        # Structured
        self.use_structured_output = use_structured_output
        self.structured_output_vocab_size = structured_output_vocab_size

        # Speech
        self.use_speech_output = use_speech_output
        self.speech_num_mels = speech_num_mels
        self.speech_head_hidden_dim = speech_head_hidden_dim
        self.speech_upsample_factor = speech_upsample_factor
        self.speech_loss_type = speech_loss_type

"""HelpingAI model configuration"""

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class HelpingAIConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HelpingAIModel`]. It is used to instantiate a
    HelpingAI model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    HelpingAI-8B [HelpingAI/HelpingAI-8B](https://huggingface.co/HelpingAI/HelpingAI-8B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the HelpingAI model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HelpingAIModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_emotional_reasoning (`bool`, *optional*, defaults to `True`):
            Whether to enable Semantic Emotion Reasoning (SER) capabilities for emotional understanding and processing.
        use_perspective_threading (`bool`, *optional*, defaults to `True`):
            Whether to enable Perspective Emotion Threading (PET) for multi-threaded emotional reasoning.
        num_emotion_heads (`int`, *optional*, defaults to 4):
            Number of specialized attention heads dedicated to emotional processing and reasoning.
        num_thinking_stages (`int`, *optional*, defaults to 3):
            Number of thinking stages for multi-stage reasoning and reflection processing.
        emotion_hidden_size (`int`, *optional*, defaults to 512):
            Hidden size for the emotional reasoning layers and SER processing modules.
        perspective_threads (`int`, *optional*, defaults to 4):
            Number of parallel perspective threads for PET processing (relatable, supportive, motivational, analytical).
        thinking_depth (`int`, *optional*, defaults to 2):
            Depth of thinking layers for internal reasoning and reflection processes.
        structured_output_vocab_size (`int`, *optional*, defaults to 100):
            Additional vocabulary size for structured output tokens like <think>, <ser>, <pet>, etc.
        empathy_scaling_factor (`float`, *optional*, defaults to 1.2):
            Scaling factor for empathy-related attention weights and emotional processing.
        reasoning_temperature (`float`, *optional*, defaults to 0.8):
            Temperature parameter for reasoning and thinking processes to balance creativity and coherence.
        use_speech_output (`bool`, *optional*, defaults to `False`):
            Whether to enable an additional text-to-speech head that predicts mel-spectrogram frames from hidden states.
        speech_num_mels (`int`, *optional*, defaults to `80`):
            Number of mel bins to predict for the speech head.
        speech_upsample_factor (`int`, *optional*, defaults to `1`):
            Temporal upsampling factor to expand token-level hidden states to frame-level resolution by simple repetition.
        speech_loss_type (`str`, *optional*, defaults to `"l1"`):
            Loss for speech supervision. One of {"l1", "mse"}.
        speech_head_hidden_dim (`int`, *optional*, defaults to `None`):
            Hidden dimension for the speech head MLP (hidden_size -> speech_head_hidden_dim -> num_mels).
            If None, defaults to hidden_size // 2. Increase to scale speech head params (e.g., ~9.6k for ~50M).

    ```python
    >>> from transformers import HelpingAIModel, HelpingAIConfig

    >>> # Initializing a HelpingAI style configuration with advanced reasoning
    >>> configuration = HelpingAIConfig(
    ...     use_emotional_reasoning=True,
    ...     use_perspective_threading=True,
    ...     num_emotion_heads=4,
    ...     num_thinking_stages=3
    ... )

    >>> # Initializing a model from the HelpingAI-8B style configuration
    >>> model = HelpingAIModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "helpingai"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `HelpingAI`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # Match num_attention_heads for compatibility
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        # Advanced reasoning parameters
        use_emotional_reasoning=False,  # Disable by default for now
        use_perspective_threading=True,
        num_emotion_heads=4,
        num_thinking_stages=3,
        emotion_hidden_size=512,
        perspective_threads=4,
        thinking_depth=2,
        structured_output_vocab_size=100,
        empathy_scaling_factor=1.2,
        reasoning_temperature=0.8,
    # Structured head architecture (new)
    structured_head_type: str = "linear",  # one of: linear, mlp_v1
    structured_head_hidden_dim: int | None = None,
    structured_head_activation: str = "gelu",  # gelu or relu
        # Speech output head options
        use_speech_output=False,
        speech_num_mels=80,
        speech_upsample_factor=1,
        speech_loss_type="l1",
        speech_head_hidden_dim=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        
        # Advanced reasoning capabilities
        self.use_emotional_reasoning = use_emotional_reasoning
        self.use_perspective_threading = use_perspective_threading
        self.num_emotion_heads = num_emotion_heads
        self.num_thinking_stages = num_thinking_stages
        self.emotion_hidden_size = emotion_hidden_size
        self.perspective_threads = perspective_threads
        self.thinking_depth = thinking_depth
        self.structured_output_vocab_size = structured_output_vocab_size
        self.empathy_scaling_factor = empathy_scaling_factor
        self.reasoning_temperature = reasoning_temperature
        # Structured head architecture spec
        self.structured_head_type = structured_head_type
        self.structured_head_hidden_dim = structured_head_hidden_dim
        self.structured_head_activation = structured_head_activation
            # Speech head config
        self.use_speech_output = use_speech_output
        self.speech_num_mels = speech_num_mels
        self.speech_upsample_factor = speech_upsample_factor
        self.speech_loss_type = speech_loss_type
        self.speech_head_hidden_dim = speech_head_hidden_dim
            
        # Validate emotional reasoning parameters
        if self.use_emotional_reasoning and self.num_emotion_heads > self.num_attention_heads:
            raise ValueError(f"num_emotion_heads ({self.num_emotion_heads}) cannot exceed num_attention_heads ({self.num_attention_heads})")
        
        if self.use_perspective_threading and self.perspective_threads < 2:
            raise ValueError(f"perspective_threads ({self.perspective_threads}) must be at least 2 for meaningful threading")
        if self.use_speech_output:
            if not isinstance(self.speech_num_mels, int) or self.speech_num_mels <= 0:
                raise ValueError("speech_num_mels must be a positive integer")
            if not isinstance(self.speech_upsample_factor, int) or self.speech_upsample_factor <= 0:
                raise ValueError("speech_upsample_factor must be a positive integer")
            if self.speech_loss_type not in {"l1", "mse"}:
                raise ValueError("speech_loss_type must be one of {'l1','mse'}")
            if self.speech_head_hidden_dim is not None:
                if not isinstance(self.speech_head_hidden_dim, int) or self.speech_head_hidden_dim <= 0:
                    raise ValueError("speech_head_hidden_dim must be a positive integer when provided")
            
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["HelpingAIConfig"]
