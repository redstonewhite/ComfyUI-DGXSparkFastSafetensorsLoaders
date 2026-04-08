# https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader
# Copyright (c) 2025 Phaserblast. All rights reserved.
# Released under the Apache 2.0 license.
from .nodes import *

NODE_CLASS_MAPPINGS = {
	"DGXSparkSafetensorsLoader": DGXSparkSafetensorsLoader,
	"DGXSparkCheckpointLoader": DGXSparkCheckpointLoader,
	"DGXSparkCLIPLoader": DGXSparkCLIPLoader,
	"DGXSparkVAELoader": DGXSparkVAELoader,
	"DGXSparkDualCLIPLoader": DGXSparkDualCLIPLoader,
	"DGXSparkLatentUpscaleModelLoader": DGXSparkLatentUpscaleModelLoader,
	"DGXSparkUnloader": DGXSparkUnloader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
	"DGXSparkSafetensorsLoader": "DGX Spark Safetensors Loader",
	"DGXSparkCheckpointLoader": "DGX Spark Checkpoint Loader",
	"DGXSparkCLIPLoader": "DGX Spark CLIP Loader",
	"DGXSparkVAELoader": "DGX Spark VAE Loader",
	"DGXSparkDualCLIPLoader": "DGX Spark Dual CLIP Loader",
	"DGXSparkLatentUpscaleModelLoader": "DGX Spark Latent Upscale Model Loader",
	"DGXSparkUnloader": "DGX Spark Model Unloader",
}
