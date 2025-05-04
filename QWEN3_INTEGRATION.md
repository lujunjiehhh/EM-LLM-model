# Qwen3 Integration with EM-LLM

This document describes the changes made to integrate Qwen3 models into the EM-LLM framework.

## Overview of Changes

The integration of Qwen3 into EM-LLM required several modifications to the framework's patch mechanism to handle Qwen3's specific architecture and attention mechanism.

### 1. Added Qwen3ForCausalLM Support in patch_hf.py

- Added Qwen3ForCausalLM to the list of supported model types
- Added import handling for Qwen3ForCausalLM (both from transformers and local file)
- Updated error message to include Qwen3 in the list of supported models

```python
if isinstance(model, LlamaForCausalLM) or isinstance(model, MistralForCausalLM) \
    or isinstance(model, Qwen2ForCausalLM) or isinstance(model, Phi3ForCausalLM) \
    or model.__class__.__name__ == "Phi3ForCausalLM" \
    or model.__class__.__name__ == "MiniCPMForCausalLM" \
    or model.__class__.__name__ == "Qwen3ForCausalLM":
    Attention = model.model.layers[0].self_attn.__class__
    Model = model.model.__class__
```

### 2. Added Qwen3-specific Rotary Embedding Handling

- Added special handling for Qwen3's rotary embedding mechanism
- Extracted head dimension and base parameters from Qwen3's configuration

```python
if model.__class__.__name__ == "Qwen3ForCausalLM":
    # Handle Qwen3 rotary embedding
    head_dim = model.model.layers[0].self_attn.head_dim
    hf_rope.dim = head_dim
    hf_rope.base = getattr(hf_rope.config, "rope_theta", 10000.0)
    base = base if base is not None else hf_rope.base
    distance_scale = distance_scale if distance_scale is not None else 1.0
    ext_factors = torch.tensor(1.0)
```

### 3. Updated huggingface_forward Function

- Modified the huggingface_forward function to handle Qwen3's normalization layers
- Added support for passing additional kwargs to the forward function

```python
# Handle Qwen3 specific normalization if present
if hasattr(self, 'q_norm') and hasattr(self, 'k_norm'):
    # For Qwen3 models with separate normalization layers
    q_norm = self.q_norm
    k_norm = self.k_norm
    kwargs['q_norm'] = q_norm
    kwargs['k_norm'] = k_norm
```

### 4. Updated em_llm_attn_forward Function

- Modified the em_llm_attn_forward function to accept and handle extra kwargs
- Added support for Qwen3's normalization layers in the query and key projection

```python
def forward(
    self, 
    query : torch.Tensor,
    key_value : torch.Tensor,
    position_bias : Optional[torch.Tensor],
    use_cache: bool,
    past_key_value,
    project_q, 
    project_k, 
    project_v, 
    attention_out, 
    dim_head, 
    num_heads, 
    num_heads_kv,
    **extra_kwargs
):
    # ...
    
    # Check if we have Qwen3-specific normalization layers
    q_norm = extra_kwargs.get('q_norm', None)
    k_norm = extra_kwargs.get('k_norm', None)
    
    # Apply Qwen3-specific normalization if available
    if q_norm is not None and k_norm is not None:
        h_q_shape = (batch_size, len_q, num_heads, dim_head)
        h_k_shape = (batch_size, len_k, num_heads_kv, dim_head)
        
        h_q = h_q.view(h_q_shape)
        h_k = h_k.view(h_k_shape)
        
        h_q = q_norm(h_q)
        h_k = k_norm(h_k)
```

## Testing

The integration was tested using a script that verifies all the necessary changes have been made to support Qwen3 models. The test confirms:

1. Qwen3ForCausalLM is included in the model check
2. Qwen3-specific normalization handling is included
3. The huggingface_forward function has been updated to handle extra kwargs
4. The em_llm_attn_forward function has been updated to handle extra kwargs
5. Qwen3-specific normalization is handled in em_llm.py

## Usage

To use Qwen3 with EM-LLM, you can now load a Qwen3 model and apply the EM-LLM patch:

```python
from transformers import Qwen3ForCausalLM, Qwen3Tokenizer
from em_llm.utils.patch_hf import patch_hf

# Load Qwen3 model and tokenizer
model_name = "Qwen/Qwen3-7B"
tokenizer = Qwen3Tokenizer.from_pretrained(model_name)
model = Qwen3ForCausalLM.from_pretrained(model_name)

# Apply EM-LLM patch
patched_model = patch_hf(
    model,
    attn_type="em_llm",
    n_local=32,
    n_init=32,
    max_block_size=32,
    max_cached_block=32,
    exc_block_size=32,
)

# Now you can use the patched model with EM-LLM's memory capabilities
```

## Limitations and Future Work

- The current implementation has been tested with the basic structure of Qwen3 but may require additional adjustments for specific variants or configurations.
- Performance optimization may be needed for production use.
- Additional testing with real Qwen3 models is recommended to ensure full compatibility.