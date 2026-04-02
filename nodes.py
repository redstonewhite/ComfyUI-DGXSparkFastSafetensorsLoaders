# Copyright (c) 2025 Phaserblast
# Licensed under the Apache License, Version 2.0
# https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader

# Modifications Copyright (c) 2026 RedstoneWhite
# This file has been substantially modified and extended.

# Project repository:
# https://github.com/redstonewhite/ComfyUI-DGXSparkFastSafetensorsLoaders

import gc
import json
import logging
import os
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
    loader = None
    fb = None
    try:
        loader = SafeTensorsFileLoader(SingleGroup(), dev)
        loader.add_filenames({0: [file_path]})
        metadata = loader.meta[file_path][0].metadata or {}
        fb = loader.copy_files_to_device()
        sd = {}
        for k in fb.key_to_rank_lidx.keys():
            sd[k] = fb.get_tensor(k)
        _move_aux_tensors_to_cpu(sd)
        return sd, metadata, fb, loader
    except KeyError as e:
        if str(e) != "'data_offsets'":
            raise
        logging.warning(
            "[DGXSpark] fastsafetensors could not parse %s (%s). Falling back to ComfyUI safetensors loading for this file.",
            file_path,
            e,
        )
        if fb is not None:
            try:
                fb.close()
            except Exception:
                pass
        if loader is not None:
            try:
                loader.close()
            except Exception:
                pass
        sd, metadata = comfy.utils.load_torch_file(
            file_path, safe_load=True, return_metadata=True
        )
        _move_aux_tensors_to_cpu(sd)
        return sd, metadata, None, None


def _move_aux_tensors_to_cpu(sd):
    for key, value in list(sd.items()):
        if not torch.is_tensor(value):
            continue
        if (
            key.endswith("spiece_model")
            or key.endswith("tekken_model")
            or key.endswith("comfy_quant")
        ):
            sd[key] = value.detach().cpu()


def _resolve_device(device):
    if device in ("default", "main_device"):
        return comfy.model_management.get_torch_device()
    if device == "cpu":
        return torch.device("cpu")
    return torch.device(device)


def _load_torch_or_fastsafe(file_path, device):
    ext = os.path.splitext(file_path)[1].lower()
    dev = device if isinstance(device, torch.device) else _resolve_device(device)
    if ext in {".safetensors", ".sft"} and dev.type != "cpu":
        return _fastsafe_load(file_path, str(dev))

    sd, metadata = comfy.utils.load_torch_file(
        file_path, safe_load=True, return_metadata=True
    )
    return sd, metadata, None, None


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
            if hasattr(obj, "named_parameters"):
                _clear_nn_params(obj)
            continue
        cleared = False
        # Diffusion model (UNet / DiT)
        dm = getattr(model, "diffusion_model", None)
        if dm is not None:
            _clear_nn_params(dm)
            cleared = True
        # CLIP text encoder
        csm = getattr(obj, "cond_stage_model", None)
        if csm is None:
            csm = model  # the patcher.model could be the TE itself
        if csm is not None and hasattr(csm, "named_parameters"):
            _clear_nn_params(csm)
            cleared = True
        # VAE
        fsm = getattr(model, "first_stage_model", None) or getattr(
            obj, "first_stage_model", None
        )
        if fsm is not None:
            _clear_nn_params(fsm)
            cleared = True
        if not cleared and hasattr(model, "named_parameters"):
            _clear_nn_params(model)

    # 3. Release fastsafetensors GPU memory
    handles = list(entry.get("handles", []))
    for h in ("fb", "loader"):
        handle = entry.get(h)
        if handle is not None:
            handles.append(handle)
    for handle in handles:
        try:
            handle.close()
        except Exception:
            pass

    del entry, patchers
    gc.collect()
    if torch.cuda.is_available():
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
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1"]

    @staticmethod
    def vae_list(s):
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
            else:
                for tae in s.video_taes:
                    if v.startswith(tae):
                        vaes.append(v)

        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        vaes.append("pixel_space")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith(f"{name}_encoder."), approx_vaes))
        decoder = next(filter(lambda a: a.startswith(f"{name}_decoder."), approx_vaes))

        enc = comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise("vae_approx", encoder)
        )
        for k in enc:
            sd[f"taesd_encoder.{k}"] = enc[k]

        dec = comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise("vae_approx", decoder)
        )
        for k in dec:
            sd[f"taesd_decoder.{k}"] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (
                    s.vae_list(s),
                    {
                        "tooltip": "The VAE model to load.",
                    },
                ),
                "device": (
                    ["cuda:0", "main_device", "cpu"],
                    {
                        "default": "cuda:0",
                        "tooltip": "The device to load to.",
                    },
                ),
                "weight_dtype": (
                    ["bf16", "fp16", "fp32"],
                    {
                        "default": "bf16",
                        "tooltip": "Weight dtype used when constructing standard ComfyUI VAE objects.",
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
    def IS_CHANGED(cls, vae_name, device, weight_dtype):
        key = _registry_key("vae", f"{vae_name}|{device}|{weight_dtype}")
        if key not in _dgx_registry:
            return float("nan")
        return _dgx_registry[key]["load_id"]

    def load_vae(self, vae_name, device="cuda:0", weight_dtype="bf16"):
        global _load_counter
        key = _registry_key("vae", f"{vae_name}|{device}|{weight_dtype}")

        if key in _dgx_registry:
            return (_dgx_registry[key]["outputs"][0],)

        dev = _resolve_device(device)
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[weight_dtype]
        metadata = None
        fb = None
        loader = None

        if vae_name == "pixel_space":
            sd = {"pixel_space_vae": torch.tensor(1.0)}
        elif vae_name in self.image_taes:
            sd = self.load_taesd(vae_name)
        else:
            if os.path.splitext(vae_name)[0] in self.video_taes:
                vae_path = folder_paths.get_full_path_or_raise("vae_approx", vae_name)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd, metadata, fb, loader = _load_torch_or_fastsafe(vae_path, dev)

        if "vocoder.conv_post.weight" in sd or "vocoder.vocoder.conv_post.weight" in sd:
            from comfy.ldm.lightricks.vae.audio_vae import AudioVAE

            vae = AudioVAE(sd, metadata)
            tracked_objects = [vae]
        else:
            with _force_assign_true():
                vae = comfy.sd.VAE(sd=sd, device=dev, dtype=dtype, metadata=metadata)
            vae.throw_exception_if_invalid()
            _fix_patcher_for_dgx(vae.patcher, dev)
            tracked_objects = [vae.patcher]

        _load_counter += 1
        _dgx_registry[key] = {
            "fb": fb,
            "loader": loader,
            "objects": tracked_objects,
            "outputs": (vae,),
            "load_id": _load_counter,
        }

        return (vae,)


class DGXSparkDualCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
                "type": (
                    [
                        "sdxl",
                        "sd3",
                        "flux",
                        "hunyuan_video",
                        "hidream",
                        "hunyuan_image",
                        "hunyuan_video_15",
                        "kandinsky5",
                        "kandinsky5_image",
                        "ltxv",
                        "newbie",
                        "ace",
                    ],
                ),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "Loads two text encoders with fastsafetensors and combines them like ComfyUI's Dual CLIP Loader."

    @classmethod
    def IS_CHANGED(cls, clip_name1, clip_name2, type, device="default"):
        key = _registry_key(
            "dual_text_encoders", f"{clip_name1}|{clip_name2}|{type}|{device}"
        )
        if key not in _dgx_registry:
            return float("nan")
        return _dgx_registry[key]["load_id"]

    def load_clip(self, clip_name1, clip_name2, type, device="default"):
        global _load_counter
        key = _registry_key(
            "dual_text_encoders", f"{clip_name1}|{clip_name2}|{type}|{device}"
        )

        if key in _dgx_registry:
            return (_dgx_registry[key]["outputs"][0],)

        dev = _resolve_device(device)
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)

        sd1, metadata1, fb1, loader1 = _load_torch_or_fastsafe(clip_path1, dev)
        sd2, metadata2, fb2, loader2 = _load_torch_or_fastsafe(clip_path2, dev)
        sd1, metadata1 = comfy.utils.convert_old_quants(
            sd1, model_prefix="", metadata=metadata1
        )
        sd2, metadata2 = comfy.utils.convert_old_quants(
            sd2, model_prefix="", metadata=metadata2
        )

        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )

        model_options = {}
        if dev.type == "cpu":
            model_options["load_device"] = dev
            model_options["offload_device"] = dev

        with _force_assign_true():
            clip = comfy.sd.load_text_encoder_state_dicts(
                state_dicts=[sd1, sd2],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options=model_options,
            )

        _fix_patcher_for_dgx(clip.patcher, dev)

        handles = [h for h in (fb1, loader1, fb2, loader2) if h is not None]

        _load_counter += 1
        _dgx_registry[key] = {
            "handles": handles,
            "objects": [clip.patcher],
            "outputs": (clip,),
            "load_id": _load_counter,
        }

        return (clip,)


class DGXSparkLatentUpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("latent_upscale_models"),
                    {
                        "tooltip": "The latent upscale model to load.",
                    },
                ),
                "device": (
                    ["cuda:0", "main_device", "cpu"],
                    {
                        "default": "cuda:0",
                        "tooltip": "The device to load to.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT_UPSCALE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a latent upscale model using NVIDIA GPUDirect on DGX Spark when the checkpoint format supports it."

    @classmethod
    def IS_CHANGED(cls, model_name, device):
        key = _registry_key("latent_upscale_models", f"{model_name}|{device}")
        if key not in _dgx_registry:
            return float("nan")
        return _dgx_registry[key]["load_id"]

    def load_model(self, model_name, device="cuda:0"):
        global _load_counter
        key = _registry_key("latent_upscale_models", f"{model_name}|{device}")

        if key in _dgx_registry:
            return (_dgx_registry[key]["outputs"][0],)

        dev = _resolve_device(device)
        model_path = folder_paths.get_full_path_or_raise(
            "latent_upscale_models", model_name
        )
        sd, metadata, fb, loader = _load_torch_or_fastsafe(model_path, dev)
        tracked_objects = []

        if "blocks.0.block.0.conv.weight" in sd:
            from comfy.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel

            config = {
                "in_channels": sd["in_conv.conv.weight"].shape[1],
                "out_channels": sd["out_conv.conv.weight"].shape[0],
                "hidden_channels": sd["in_conv.conv.weight"].shape[0],
                "num_blocks": len(
                    [
                        k
                        for k in sd.keys()
                        if k.startswith("blocks.")
                        and k.endswith(".block.0.conv.weight")
                    ]
                ),
                "global_residual": False,
            }
            with _force_assign_true():
                model = HunyuanVideo15SRModel("720p", config)
                model.load_sd(sd)
            _fix_patcher_for_dgx(model.patcher, dev)
            tracked_objects = [model.patcher]
        elif "up.0.block.0.conv1.conv.weight" in sd:
            from comfy.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel

            sd = {
                key.replace("nin_shortcut", "nin_shortcut.conv", 1): value
                for key, value in sd.items()
            }
            block_count = len(
                [
                    k
                    for k in sd.keys()
                    if k.startswith("up.") and k.endswith(".block.0.conv1.conv.weight")
                ]
            )
            config = {
                "z_channels": sd["conv_in.conv.weight"].shape[1],
                "out_channels": sd["conv_out.conv.weight"].shape[0],
                "block_out_channels": tuple(
                    sd[f"up.{i}.block.0.conv1.conv.weight"].shape[0]
                    for i in range(block_count)
                ),
            }
            with _force_assign_true():
                model = HunyuanVideo15SRModel("1080p", config)
                model.load_sd(sd)
            _fix_patcher_for_dgx(model.patcher, dev)
            tracked_objects = [model.patcher]
        elif "post_upsample_res_blocks.0.conv2.bias" in sd:
            from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler

            config = json.loads(metadata["config"])
            dtype = comfy.model_management.vae_dtype(
                dev, allowed_dtypes=[torch.bfloat16, torch.float32]
            )
            model = LatentUpsampler.from_config(config).to(device=dev, dtype=dtype)
            model.load_state_dict(sd, assign=True)
            tracked_objects = [model]
        else:
            if fb is not None:
                fb.close()
            if loader is not None:
                loader.close()
            raise RuntimeError(f"Unsupported latent upscale model format: {model_name}")

        handles = [h for h in (fb, loader) if h is not None]

        _load_counter += 1
        _dgx_registry[key] = {
            "handles": handles,
            "objects": tracked_objects,
            "outputs": (model,),
            "load_id": _load_counter,
        }

        return (model,)


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
