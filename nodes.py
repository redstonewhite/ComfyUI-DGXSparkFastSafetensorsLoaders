# Copyright (c) 2025 Phaserblast
# Licensed under the Apache License, Version 2.0
# https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader

# Modifications Copyright (c) 2026 RedstoneWhite
# This file has been substantially modified and extended.

# Project repository:
# https://github.com/redstonewhite/ComfyUI-DGXSparkFastSafetensorsLoaders

import gc
import logging
from contextlib import contextmanager
import torch
import folder_paths
import comfy
import comfy.model_management
import comfy.sd

# https://github.com/foundation-model-stack/fastsafetensors
from fastsafetensors import fastsafe_open, SafeTensorsFileLoader, SingleGroup

# ---------------------------------------------------------------------------
# Global registry tracking every model loaded by a DGX Spark node.
# key   = "<category>:<filename>"  (e.g. "diffusion_models:flux-dev.safetensors")
# value = {fb, loader, objects, load_id}
#   objects  – list of ModelPatcher-like objects whose params reference fb
# ---------------------------------------------------------------------------
_dgx_registry = {}
_load_counter = 0


def _registry_key(category, name):
    return f"{category}:{name}"


def _fastsafe_load(file_path, device):
    """Load a .safetensors file with fastsafetensors, returning (sd, metadata, fb, loader).
    The caller MUST keep fb and loader alive as long as the tensors are in use."""
    dev = torch.device(device)
    loader = SafeTensorsFileLoader(SingleGroup(), dev)
    loader.add_filenames({0: [file_path]})
    metadata = loader.meta[file_path][0].metadata or {}
    fb = loader.copy_files_to_device()
    sd = {}
    for k in fb.key_to_rank_lidx.keys():
        sd[k] = fb.get_tensor(k)
    return sd, metadata, fb, loader


@contextmanager
def _force_assign_true():
    """Temporarily make ModelPatcher.is_dynamic() return True so that all
    load_state_dict calls inside ComfyUI's loading pipeline use assign=True.
    This prevents tensor duplication when fastsafetensors already placed
    the data in VRAM (unified memory on DGX Spark)."""
    orig = comfy.model_patcher.ModelPatcher.is_dynamic
    comfy.model_patcher.ModelPatcher.is_dynamic = lambda self: True
    try:
        yield
    finally:
        comfy.model_patcher.ModelPatcher.is_dynamic = orig


def _fix_patcher_for_dgx(patcher, dev):
    """Adjust a ModelPatcher for DGX Spark unified memory: set offload
    device equal to load device so ComfyUI never tries to move weights."""
    if patcher is not None:
        patcher.load_device = dev
        patcher.offload_device = dev


def _clear_nn_params(module):
    """Replace every parameter/buffer in *module* with an empty CPU tensor
    so that all references to fastsafetensors memory are broken."""
    for _name, param in list(module.named_parameters()):
        try:
            param.data = torch.empty(0, device="cpu")
        except Exception:
            pass
    for _name, buf in list(module.named_buffers()):
        try:
            buf.data = torch.empty(0, device="cpu")
        except Exception:
            pass


def _remove_from_comfyui(patchers):
    """Remove the given ModelPatcher objects from ComfyUI's loaded-model list."""
    patcher_set = set(id(p) for p in patchers)
    to_remove = []
    for i, loaded in enumerate(comfy.model_management.current_loaded_models):
        lm = loaded.model
        if lm is None:
            continue
        if id(lm) in patcher_set or id(getattr(lm, "parent", None)) in patcher_set:
            to_remove.append(i)
    for i in reversed(to_remove):
        try:
            entry = comfy.model_management.current_loaded_models[i]
            if entry.model_finalizer is not None:
                entry.model_finalizer.detach()
                entry.model_finalizer = None
            entry.real_model = None
        except Exception:
            pass
        comfy.model_management.current_loaded_models.pop(i)


def _cleanup_model(key):
    """Free all memory for a model tracked under *key* in the registry."""
    if key not in _dgx_registry:
        return False

    entry = _dgx_registry.pop(key)
    patchers = entry.get("objects", [])

    # 1. Remove from ComfyUI tracking
    _remove_from_comfyui(patchers)

    # 2. Wipe tensor data in every tracked nn.Module
    for obj in patchers:
        model = getattr(obj, "model", None)
        if model is None:
            continue
        # Diffusion model (UNet / DiT)
        dm = getattr(model, "diffusion_model", None)
        if dm is not None:
            _clear_nn_params(dm)
        # CLIP text encoder
        csm = getattr(obj, "cond_stage_model", None)
        if csm is None:
            csm = model  # the patcher.model could be the TE itself
        if csm is not None and hasattr(csm, "named_parameters"):
            _clear_nn_params(csm)
        # VAE
        fsm = getattr(model, "first_stage_model", None) or getattr(
            obj, "first_stage_model", None
        )
        if fsm is not None:
            _clear_nn_params(fsm)

    # 3. Release fastsafetensors GPU memory
    for h in ("fb", "loader"):
        handle = entry.get(h)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    del entry, patchers
    gc.collect()
    torch.cuda.empty_cache()

    logging.info(f"[DGXSpark] Unloaded: {key}")
    return True


# ===================================================================
#  Loader nodes
# ===================================================================


class DGXSparkSafetensorsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "The filename of the .safetensors model to load."},
                ),
                "device": (
                    ["cuda:0"],
                    {
                        "default": "cuda:0",
                        "tooltip": "The device to which the model will be copied.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model .safetensors file directly into memory using NVIDIA GPUDirect on DGX Spark."

    @classmethod
    def IS_CHANGED(cls, model_name, device):
        key = _registry_key("diffusion_models", model_name)
        if key not in _dgx_registry:
            return float("nan")
        return _dgx_registry[key]["load_id"]

    def load_model(self, model_name, device):
        global _load_counter
        key = _registry_key("diffusion_models", model_name)

        if key in _dgx_registry:
            return (_dgx_registry[key]["objects"][0],)

        dev = torch.device(device)
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)

        sd, metadata, fb, loader = _fastsafe_load(model_path, device)

        # Init the model to pass to ComfyUI
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        temp_sd = comfy.utils.state_dict_prefix_replace(
            sd, {diffusion_model_prefix: ""}, filter_keys=True
        )
        if len(temp_sd) > 0:
            sd = temp_sd

        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

        model_config = comfy.model_detection.model_config_from_unet(
            sd, "", metadata=metadata
        )
        if model_config is None:
            fb.close()
            loader.close()
            raise RuntimeError("Couldn't detect model type.")

        parameters = comfy.utils.calculate_parameters(sd)
        weight_dtype = comfy.utils.weight_dtype(sd, "")

        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        if model_config.quant_config is not None:
            weight_dtype = None

        unet_dtype = comfy.model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=weight_dtype,
        )

        if model_config.quant_config is not None:
            manual_cast_dtype = comfy.model_management.unet_manual_cast(
                None, dev, model_config.supported_inference_dtypes
            )
        else:
            manual_cast_dtype = comfy.model_management.unet_manual_cast(
                unet_dtype, dev, model_config.supported_inference_dtypes
            )
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

        if model_config.optimizations.get("fp8", False):
            model_config.optimizations["fp8"] = True

        model = model_config.get_model(sd, "", device=None)
        sd = model_config.process_unet_state_dict(sd)
        model.diffusion_model.load_state_dict(sd, strict=False, assign=True)

        model_patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=dev, offload_device=dev
        )

        _load_counter += 1
        _dgx_registry[key] = {
            "fb": fb,
            "loader": loader,
            "objects": [model_patcher],
            "load_id": _load_counter,
        }

        return (model_patcher,)


class DGXSparkCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {
                        "tooltip": "The name of the checkpoint to load.",
                    },
                ),
                "device": (
                    ["cuda:0"],
                    {
                        "default": "cuda:0",
                        "tooltip": "The device to load to.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a checkpoint .safetensors file (model + CLIP + VAE) using NVIDIA GPUDirect on DGX Spark."

    @classmethod
    def IS_CHANGED(cls, ckpt_name, device):
        key = _registry_key("checkpoints", ckpt_name)
        if key not in _dgx_registry:
            return float("nan")
        return _dgx_registry[key]["load_id"]

    def load_checkpoint(self, ckpt_name, device):
        global _load_counter
        key = _registry_key("checkpoints", ckpt_name)

        if key in _dgx_registry:
            cached = _dgx_registry[key]["outputs"]
            return cached

        dev = torch.device(device)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        sd, metadata, fb, loader = _fastsafe_load(ckpt_path, device)

        # Use ComfyUI's own config guesser which splits sd into model/clip/vae.
        # Wrap with _force_assign_true so all internal load_state_dict calls
        # use assign=True, preventing tensor duplication on unified memory.
        with _force_assign_true():
            out = comfy.sd.load_state_dict_guess_config(
                sd,
                output_vae=True,
                output_clip=True,
                output_clipvision=False,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                output_model=True,
                metadata=metadata,
            )
        if out is None:
            fb.close()
            loader.close()
            raise RuntimeError(f"Could not detect model type of: {ckpt_path}")

        model_patcher, clip, vae, _clipvision = out

        # Fix patcher devices for DGX Spark unified memory
        _fix_patcher_for_dgx(model_patcher, dev)
        if clip is not None:
            _fix_patcher_for_dgx(clip.patcher, dev)
        if vae is not None:
            _fix_patcher_for_dgx(vae.patcher, dev)

        # Track all patcher objects for cleanup
        tracked = []
        if model_patcher is not None:
            tracked.append(model_patcher)
        if clip is not None:
            tracked.append(clip.patcher)
        if vae is not None:
            tracked.append(vae.patcher)

        _load_counter += 1
        _dgx_registry[key] = {
            "fb": fb,
            "loader": loader,
            "objects": tracked,
            "outputs": (model_patcher, clip, vae),
            "load_id": _load_counter,
        }

        return (model_patcher, clip, vae)


class DGXSparkCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (
                    folder_paths.get_filename_list("text_encoders"),
                    {
                        "tooltip": "The CLIP / text encoder model to load.",
                    },
                ),
                "type": (
                    [
                        "stable_diffusion",
                        "stable_cascade",
                        "sd3",
                        "stable_audio",
                        "mochi",
                        "ltxv",
                        "pixart",
                        "cosmos",
                        "lumina2",
                        "wan",
                        "hidream",
                        "chroma",
                        "ace",
                        "omnigen2",
                        "qwen_image",
                        "hunyuan_image",
                        "flux2",
                        "ovis",
                        "longcat_image",
                    ],
                ),
                "device": (
                    ["cuda:0"],
                    {
                        "default": "cuda:0",
                        "tooltip": "The device to load to. On DGX Spark, always use cuda:0 (unified memory).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a CLIP / text encoder .safetensors file using NVIDIA GPUDirect on DGX Spark."

    @classmethod
    def IS_CHANGED(cls, clip_name, type, device):
        key = _registry_key("text_encoders", clip_name)
        if key not in _dgx_registry:
            return float("nan")
        return _dgx_registry[key]["load_id"]

    def load_clip(self, clip_name, type="stable_diffusion", device="cuda:0"):
        global _load_counter
        key = _registry_key("text_encoders", clip_name)

        if key in _dgx_registry:
            return (_dgx_registry[key]["outputs"][0],)

        dev = torch.device(device)
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        sd, metadata, fb, loader = _fastsafe_load(clip_path, device)

        # Convert old quant formats
        sd, metadata = comfy.utils.convert_old_quants(
            sd, model_prefix="", metadata=metadata
        )

        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )

        with _force_assign_true():
            clip = comfy.sd.load_text_encoder_state_dicts(
                state_dicts=[sd],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
            )

        _fix_patcher_for_dgx(clip.patcher, dev)

        _load_counter += 1
        _dgx_registry[key] = {
            "fb": fb,
            "loader": loader,
            "objects": [clip.patcher],
            "outputs": (clip,),
            "load_id": _load_counter,
        }

        return (clip,)


class DGXSparkVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (
                    folder_paths.get_filename_list("vae"),
                    {
                        "tooltip": "The VAE model to load.",
                    },
                ),
                "device": (
                    ["cuda:0"],
                    {
                        "default": "cuda:0",
                        "tooltip": "The device to load to.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a VAE .safetensors file using NVIDIA GPUDirect on DGX Spark."

    @classmethod
    def IS_CHANGED(cls, vae_name, device):
        key = _registry_key("vae", vae_name)
        if key not in _dgx_registry:
            return float("nan")
        return _dgx_registry[key]["load_id"]

    def load_vae(self, vae_name, device="cuda:0"):
        global _load_counter
        key = _registry_key("vae", vae_name)

        if key in _dgx_registry:
            return (_dgx_registry[key]["outputs"][0],)

        dev = torch.device(device)
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, metadata, fb, loader = _fastsafe_load(vae_path, device)

        with _force_assign_true():
            vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()

        _fix_patcher_for_dgx(vae.patcher, dev)

        _load_counter += 1
        _dgx_registry[key] = {
            "fb": fb,
            "loader": loader,
            "objects": [vae.patcher],
            "outputs": (vae,),
            "load_id": _load_counter,
        }

        return (vae,)


# ===================================================================
#  Unloader node
# ===================================================================


def _loaded_model_choices():
    """Return a list of currently loaded registry keys for the COMBO widget.
    Always includes a stable placeholder so stale frontend values never
    fail server-side validation."""
    return ["(none)"] + sorted(_dgx_registry.keys())


class DGXSparkUnloader:
    """Frees GPU/RAM memory for models loaded by any DGX Spark loader node.
    Set 'confirm' to True and queue the workflow (or click play) to unload.
    Safe no-op when confirm is False.

    The 'target' dropdown lists models currently held by DGX Spark loaders.
    Refresh the browser page to update the list after loading new models."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "confirm": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Safety toggle. Must be True to unload. When False the node is a no-op.",
                    },
                ),
                "mode": (
                    ["all", "selected"],
                    {
                        "default": "all",
                        "tooltip": "'all' unloads every model loaded by any DGX Spark loader. 'selected' unloads only the model chosen in 'target'.",
                    },
                ),
            },
            "optional": {
                "target": (
                    _loaded_model_choices(),
                    {
                        "default": "(none)",
                        "tooltip": "The model to unload (only used when mode is 'selected'). Refresh the page to update this list after loading models.",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "unload_model"
    OUTPUT_NODE = True
    CATEGORY = "loaders"
    DESCRIPTION = "Frees GPU/RAM memory for models loaded by DGX Spark loaders. Set 'confirm' to True and queue to trigger. Safe no-op when confirm is False."

    @classmethod
    def IS_CHANGED(cls, confirm, mode, target="(none)"):
        if not confirm:
            return False
        if mode == "all" and not _dgx_registry:
            return False
        if mode == "selected" and target not in _dgx_registry:
            return False
        return float("nan")

    def unload_model(self, confirm, mode, target="(none)"):
        loaded = sorted(_dgx_registry.keys())
        status_line = (
            f"Currently loaded ({len(loaded)}): {', '.join(loaded)}"
            if loaded
            else "No models loaded."
        )

        if not confirm:
            return {"ui": {"text": [status_line, "Action: none (confirm is False)"]}}

        if mode == "all":
            if not loaded:
                return {"ui": {"text": [status_line, "Nothing to unload."]}}
            count = len(loaded)
            for k in loaded:
                _cleanup_model(k)
            return {
                "ui": {
                    "text": [
                        f"Unloaded {count} model(s): {', '.join(loaded)}",
                        "Tip: set confirm back to False before next run.",
                    ]
                }
            }

        # mode == "selected"
        if target in _dgx_registry:
            _cleanup_model(target)
            remaining = sorted(_dgx_registry.keys())
            remain_line = (
                f"Still loaded ({len(remaining)}): {', '.join(remaining)}"
                if remaining
                else "No models remain."
            )
            return {
                "ui": {
                    "text": [
                        f"Unloaded: {target}",
                        remain_line,
                        "Tip: set confirm back to False before next run.",
                    ]
                }
            }

        return {"ui": {"text": [status_line, f"Target not found: {target}"]}}
