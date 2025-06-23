import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig
from detectron2.modeling import Backbone # This is the abstract class from your provided code
from detectron2.layers import ShapeSpec
from typing import List, Dict

class SwinV2Backbone(Backbone):
    def __init__(self, model_name: str, out_features: List[str], freeze_at: int = 0, norm_output: bool = False):
        """
        Args:
            model_name (str): Name of the SwinV2 model from Hugging Face Model Hub
                              (e.g., "microsoft/swinv2-tiny-patch4-window16-256").
            out_features (List[str]): List of stage names to return features from.
                                      Expected values like "s1", "s2", "s3", "s4".
            freeze_at (int):
                - 0: no layers are frozen.
                - 1: freeze patch embedding.
                - 2: freeze patch embedding and stage 1.
                - 3: freeze patch embedding and stages 1, 2.
                - ... and so on.
                - freeze_at = num_stages + 1 will freeze all stages and patch_embed.
            norm_output (bool): If True, apply LayerNorm to the output features.
                                Swin stages already have normalization, so typically False.
        """
        super().__init__()

        self.swin_model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            add_pooling_layer=False  # We want features, not a pooled output
        )
        # Ensure hidden states are reshaped to (B, H, W, C) if model config supports it
        # Most Swin models in HF have this enabled by default (config.reshape_hidden_states = True)

        model_config = self.swin_model.config
        self.patch_size = model_config.patch_size
        embed_dim = model_config.embed_dim
        num_stages = len(model_config.depths) # e.g., 4 for "tiny"

        self._out_features = out_features
        self._out_feature_channels = {}
        self._out_feature_strides = {}

        # Naming convention for out_features: "s1", "s2", "s3", "s4"
        # These map to Hugging Face hidden_states indices:
        # hidden_states[0] is initial patch embedding output (before 1st stage)
        # hidden_states[1] is output of Stage 1 (HF SwinStage 0)
        # hidden_states[2] is output of Stage 2 (HF SwinStage 1)
        # ...
        # hidden_states[num_stages] is output of Stage num_stages (HF SwinStage num_stages-1)

        # Map desired "sX" names to HF hidden_states indices and properties
        # Example: "s1" -> hidden_states[1], "s2" -> hidden_states[2], etc.
        self._feature_map_hf_indices = []
        possible_out_features_config = {}

        current_stride = self.patch_size
        current_channels = embed_dim
        # Stage 1 (output of HF encoder.stages[0], which is hidden_states[1])
        possible_out_features_config["s1"] = {
            "hf_idx": 1, "stride": current_stride, "channels": current_channels
        }
        # Subsequent stages
        for i in range(1, num_stages): # From stage 2 up to num_stages
            current_stride *= 2
            current_channels *= 2
            stage_name = f"s{i+1}" # s2, s3, s4
            possible_out_features_config[stage_name] = {
                "hf_idx": i + 1, "stride": current_stride, "channels": current_channels
            }

        for out_name in self._out_features:
            if out_name not in possible_out_features_config:
                raise ValueError(f"Unknown out_feature {out_name}. Possible values are: {list(possible_out_features_config.keys())}")
            config = possible_out_features_config[out_name]
            self._feature_map_hf_indices.append(config["hf_idx"])
            self._out_feature_strides[out_name] = config["stride"]
            self._out_feature_channels[out_name] = config["channels"]

        # Freezing logic
        if freeze_at > 0:
            print(f"Freezing SwinV2 patch embeddings.")
            for param in self.swin_model.embeddings.parameters():
                param.requires_grad_(False)

        # self.swin_model.encoder.stages is a ModuleList of SwinStage modules
        for i, stage in enumerate(self.swin_model.encoder.stages):
            if freeze_at >= (i + 2): # freeze_at=2 freezes embed + stage 0 (s1)
                print(f"Freezing SwinV2 encoder stage {i} (outputting as s{i+1}).")
                for param in stage.parameters():
                    param.requires_grad_(False)
        
        # Optional: Normalize output features. Swin stages already have norms.
        self.norm_output = norm_output
        if self.norm_output:
            self.output_norms = nn.ModuleDict()
            for name, channels in self._out_feature_channels.items():
                # Using LayerNorm, expecting (B, H, W, C) then permute
                # Or, apply after permuting to (B, C, H, W) using a wrapper
                # For simplicity, let's assume we apply it on (B,C,H,W) via a wrapper
                # Or, just use nn.LayerNorm and handle permutes in forward
                # For now, let's define LayerNorms which expect (..., C)
                self.output_norms[name] = nn.LayerNorm(channels, eps=model_config.layer_norm_eps)


    def forward(self, images_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images_tensor (torch.Tensor): Input tensor of shape (N, C, H, W).
                                          Assumes C=3 and values are appropriately normalized.
        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping feature names (e.g., "s1", "s2")
                                     to feature tensors of shape (N, Channels, H_feat, W_feat).
        """
        # Hugging Face SwinModel expects `pixel_values` as input.
        outputs = self.swin_model(pixel_values=images_tensor)
        
        # `outputs.hidden_states` is a tuple.
        # hidden_states[0] is the output of patch_embed + norm.
        # hidden_states[1] is the output of the first Swin Stage (Stage 1 / s1).
        # hidden_states[i] is the output of the i-th Swin Stage.
        all_hidden_states = outputs.hidden_states

        features = {}
        for i, out_name in enumerate(self._out_features):
            hf_idx = self._feature_map_hf_indices[i]
            
            # Feature tensor from HF SwinModel is (Batch, Height, Width, Channels)
            # This is because config.reshape_hidden_states is typically True for Swin.
            feature_tensor_bhwc = all_hidden_states[hf_idx]

            if self.norm_output:
                # LayerNorm expects (..., C)
                feature_tensor_bhwc = self.output_norms[out_name](feature_tensor_bhwc)

            # Permute to Detectron2's expected (Batch, Channels, Height, Width)
            feature_tensor_bchw = feature_tensor_bhwc.permute(0, 3, 1, 2).contiguous()
            features[out_name] = feature_tensor_bchw
            
        return features

    @property
    def size_divisibility(self) -> int:
        """
        The maximum stride among output features. For SwinV2, this is typically
        patch_size * 2^(num_downsampling_stages). For a 4-stage Swin-T with patch_size=4,
        it's 4 * 2^3 = 32.
        """
        max_stride = 0
        if self._out_feature_strides:
            max_stride = max(self._out_feature_strides.values())
        return max_stride if max_stride > 0 else 32 # Fallback for safety