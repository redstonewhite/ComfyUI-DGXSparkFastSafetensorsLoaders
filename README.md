# ComfyUI-DGXSparkFastSafetensorsLoaders

> **Use at your own risk. No stability guarantee**

An extended version of [ComfyUI-DGXSparkSafetensorsLoader](https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader) by [Phaserblast](https://github.com/phaserblast). This repo adds more nodes to support drop-in alternatives for Checkpoint Loader, CLIP Loader and VAE Loader, as well as a Model Unloader to free VRAM without the need to restart the server.

This custom node collection loads `.safetensors` model files into ComfyUI using the [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) library, which performs fast, zero-copy transfers from storage directly to VRAM via NVIDIA GPUDirect. It is optimized for the NVIDIA DGX Spark's unified memory architecture, where the standard Hugging Face `safetensors` library can be slow and may transiently use up to 2× the model's size in RAM during loading.

In my workflows with Qwen Image Edit, it reduces loading time to ~5s total.

## Nodes

| Node | Description |
|---|---|
| **DGX Spark Safetensors Loader** | Loads a diffusion model (UNet / DiT) from `diffusion_models/` |
| **DGX Spark Checkpoint Loader** | Loads a full checkpoint (model + CLIP + VAE) from `checkpoints/` |
| **DGX Spark CLIP Loader** | Loads a CLIP / text encoder from `text_encoders/` |
| **DGX Spark VAE Loader** | Loads a VAE from `vae/` |
| **DGX Spark Model Unloader** | Explicitly frees fastsafetensors GPU memory for a loaded model |

All loader nodes cache loaded models in a global registry so re-running a workflow does not reload the file from disk. The **DGX Spark Model Unloader** node allows explicit VRAM reclamation without restarting ComfyUI. This addresses the memory-management limitation present in the original implementation.

## Usage

![nodes](images/image.png)

`Safetensors Loader`, `Checkpoints Loader`, `CLIP Loader` and `VAE Loader` behave similarly to their official counterparts, except that their memory are not managed by ComfyUI. Thus, you can directly replace the original loader(s) in your workflow with them as needed.

If you want to unload models loaded by these fastsafetensors loaders, add a `Model Unloader` to your workflow, enable `confirm` and run the unloader node. To unload certain model, you should change `mode` to `selected` and choose your target model in `target`. You may need to refresh your page for loaded models to appear in the drop-down menu.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/redstonewhite/ComfyUI-DGXSparkFastSafetensorsLoaders.git
```

Install the dependency (activate your venv first if applicable):

```bash
pip install fastsafetensors
```

Restart ComfyUI. The nodes appear in the **loaders** category.

## Notes

- Tested with `--disable-mmap` and `--gpu-only` flags, though they might be unnecessary.

## Acknowledgements

Original node by [Phaserblast](https://github.com/phaserblast) — [ComfyUI-DGXSparkSafetensorsLoader](https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader), licensed under the Apache License 2.0.
