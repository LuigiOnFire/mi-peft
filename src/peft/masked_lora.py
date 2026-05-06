import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadMasker:
    """
    A utility class that translates a list of critical GPT-2 attention heads 
    into binary tensor masks used for protecting those specific dimensions 
    from being updated during parameter-efficient fine-tuning (LoRA).
    """

    def __init__(self, critical_heads, d_model=768, n_heads=12, n_layers=12):
        """
        Args:
            critical_heads: A list of tuples, e.g., [(layer_idx, head_idx)]
            d_model: The hidden size of the transformer (GPT-2 Small = 768)
            n_heads: Total number of attention heads per layer (GPT-2 Small = 12)
            n_layers: Total number of layers (GPT-2 Small = 12)
        """
        self.critical_heads = critical_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_layers = n_layers

    def generate_masks(self):
        """
        Generates the boolean/float masks mapped to their specific layer names.

        Returns:
            dict: Mapping like {'transformer.h.0.attn.c_attn': tensor_mask_c_attn, ...}
                  0.0 means the dimension is protected (frozen).
                  1.0 means the dimension is trainable.
        """
        masks = {}

        # We generate masks for ALL layers so LoRA can still apply everywhere globally,
        # but the specific critical heads will have internal 0.0 zeroes in their layers.
        for layer in range(self.n_layers):
            attn_name = f"transformer.h.{layer}.attn.c_attn"
            proj_name = f"transformer.h.{layer}.attn.c_proj"

            # Base masks: all 1.0s (fully trainable)
            # c_attn weight shape is [d_model, 3 * d_model]
            mask_c_attn = torch.ones((self.d_model, 3 * self.d_model))
            # c_proj weight shape is [d_model, d_model]
            mask_c_proj = torch.ones((self.d_model, self.d_model))

            # Find all heads in THIS layer that need protection
            layer_heads = [h for l, h in self.critical_heads if l == layer]
            
            for head_idx in layer_heads:
                start = head_idx * self.d_head
                end = (head_idx + 1) * self.d_head

                # -------------------------------------------------------------
                # GPT-2 c_attn: Query, Key, Value are concatenated horizontally 
                # across the output dimension (dim 1).
                # -------------------------------------------------------------
                mask_c_attn[:, start : end] = 0.0                                    # Query Slice
                mask_c_attn[:, self.d_model + start : self.d_model + end] = 0.0      # Key Slice
                mask_c_attn[:, 2 * self.d_model + start : 2 * self.d_model + end] = 0.0  # Value Slice

                # -------------------------------------------------------------
                # GPT-2 c_proj: Output from heads is concatenated along the
                # input dimension (rows / dim 0).
                # -------------------------------------------------------------
                mask_c_proj[start : end, :] = 0.0                                    # Input dimension slice

            masks[attn_name] = mask_c_attn
            masks[proj_name] = mask_c_proj

        return masks


class MaskedLoRALinear(nn.Module):
    """
    A custom wrapper that applies a masked Low-Rank Adaptation (LoRA) update.
    Unlike standard LoRA, the forward pass restricts updates across specific matrix dimensions
    by enforcing an element-wise multiplication with a binary mask.
    """

    def __init__(self, base_layer, mask, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Freeze the base layer entirely
        self.base_layer.weight.requires_grad = False
        if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Identify if base_layer is PyTorch nn.Linear or HuggingFace Conv1D
        # HW Conv1D weight shape is [in_features, out_features]
        # nn.Linear weight shape is [out_features, in_features]
        if isinstance(base_layer, nn.Linear):
            self.is_conv1d = False
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
            self.weight_shape = (self.out_features, self.in_features)
        else:
            # Assuming it's HuggingFace's Conv1D used heavily in GPT-2 architectures
            self.is_conv1d = True
            self.in_features = base_layer.weight.shape[0]
            self.out_features = base_layer.weight.shape[1]
            self.weight_shape = (self.in_features, self.out_features)

        # Ensure the mask shape perfectly matches the base layer's weight shape
        if mask.shape != self.weight_shape:
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match base layer weight shape {self.weight_shape}")

        # Register mask as a non-trainable state buffer so it moves properly to device boundaries 
        self.register_buffer("mask", mask)

        # ---------------------------------------------------------------------
        # Initialize LoRA specific weights.
        # We align the math geometrically based on the layer type so that 
        # delta_W directly matches self.weight_shape before mask multiplication.
        # ---------------------------------------------------------------------
        if self.is_conv1d:
            # For Conv1D, Weight is [in, out]. delta_W = A @ B -> [in, out]
            self.lora_A = nn.Parameter(torch.empty(self.in_features, r))
            self.lora_B = nn.Parameter(torch.empty(r, self.out_features))
        else:
            # For nn.Linear, Weight is [out, in]. delta_W = B @ A -> [out, in]
            # (Matches standard LoRA paper math)
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, r))

        # Initialization constraints as per standard LoRA guidelines
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout)

    @property
    def weight(self):
        """
        Dynamically merges the LoRA weight into the base weight for external reads.
        This allows frameworks like TransformerLens to successfully extract the 
        post-intervention weights without crashing!
        """
        if self.is_conv1d:
            delta_W = self.lora_A @ self.lora_B
            masked_delta_W = delta_W * self.mask
            return self.base_layer.weight + masked_delta_W * self.scaling
        else:
            delta_W = self.lora_B @ self.lora_A
            masked_delta_W = delta_W * self.mask
            return self.base_layer.weight + masked_delta_W * self.scaling

    @property
    def bias(self):
        """Pass-through for bias so TransformerLens can read it cleanly."""
        return self.base_layer.bias

    def forward(self, x):
        # The frozen representation from the pretrained weights
        base_output = self.base_layer(x)

        # --------------------------------------------------------
        # Mathematical Construction of Masked Forward Pass
        # masked_delta_W = (B @ A) * mask 
        # --------------------------------------------------------
        if self.is_conv1d:
            # A: [in, r], B: [r, out] -> delta_W is [in, out]
            delta_W = self.lora_A @ self.lora_B
            masked_delta_W = delta_W * self.mask
            
            # x is [batch, seq, in], so x @ [in, out] -> [batch, seq, out]
            lora_output = torch.matmul(self.dropout(x), masked_delta_W) * self.scaling
        else:
            # B: [out, r], A: [r, in] -> delta_W is [out, in]
            delta_W = self.lora_B @ self.lora_A
            masked_delta_W = delta_W * self.mask
            
            # x is [batch, seq, in]. F.linear internally executes x @ W.T
            # So F.linear(x, masked_delta_W) logically equates to x @ masked_delta_W.T
            lora_output = F.linear(self.dropout(x), masked_delta_W) * self.scaling

        return base_output + lora_output


def inject_masked_lora(model, mask_dict, r=8, alpha=16, dropout=0.1):
    """
    Recursively iterates through an instantiated Hugging Face model and wraps
    target layers defined in `mask_dict` with `MaskedLoRALinear`.
    
    This replaces standard huggingface PEFT integrations.
    """
    injected_count = 0

    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            # Reconstruct the full path name of the child module
            full_child_name = f"{name}.{child_name}" if name else child_name
            
            if full_child_name in mask_dict:
                print(f"Injecting Masked LoRA into: {full_child_name}")
                
                mask = mask_dict[full_child_name]
                masked_lora = MaskedLoRALinear(
                    base_layer=child_module,
                    mask=mask,
                    r=r,
                    alpha=alpha,
                    dropout=dropout
                )
                
                # Move the newly manufactured Masked LoRA weights to whatever 
                # device/dtype the base underlying weight was localized on
                masked_lora.to(child_module.weight.device, dtype=child_module.weight.dtype)
                
                # Hot-swap the original module with our custom module
                setattr(module, child_name, masked_lora)
                injected_count += 1
                
    print(f"Successfully injected custom Masked LoRA into {injected_count} layers.")
    return model