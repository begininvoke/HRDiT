"""HRDiT inference: training-free high-resolution generation with FLUX.1-dev.

    python inference.py --prompt "A textbook shows a large group of people."
"""
import argparse
import json
import os

import torch

from hrdit import FluxPipeline, FluxTransformer2DModel


def parse_args():
    p = argparse.ArgumentParser(description="HRDiT high-resolution generation")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt_file", type=str, default=None,
                   help="text file with one prompt per line, or a JSON list of strings")
    p.add_argument("--num_prompts", type=int, default=1)
    p.add_argument("--prompt_offset", type=int, default=0)
    p.add_argument("--model_path", type=str, default="./pretrained_models/FLUX.1-dev")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--resolutions", type=int, nargs="+", default=[2048, 4096],
                   help="target resolutions, generated in order after the 1024 base")
    p.add_argument("--scope_plan", type=str, default="configs/scope_plan_flux.json",
                   help="per-head attention scopes used by HAP")
    p.add_argument("--pipeline_config", type=str, default="configs/pipeline.json")
    p.add_argument("--spa_steps", type=int, nargs="+", default=[3, 0],
                   help="leading denoising steps that use SPA, per stage")
    p.add_argument("--group_num", type=int, default=80, help="SPA bundle granularity")
    p.add_argument("--global_anchor", type=int, default=32,
                   help="HAP keeps every N-th token block globally visible (0 = off)")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    return p.parse_args()


def load_prompts(args):
    if args.prompt:
        return [args.prompt]
    if not args.prompt_file:
        raise SystemExit("give either --prompt or --prompt_file")
    with open(args.prompt_file) as f:
        if args.prompt_file.endswith(".json"):
            prompts = json.load(f)
        else:
            prompts = [line.strip() for line in f if line.strip()]
    return prompts[args.prompt_offset: args.prompt_offset + args.num_prompts]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    n_stages = len(args.resolutions)
    spa_steps = (args.spa_steps + [0] * n_stages)[:n_stages]
    with open(args.pipeline_config) as f:
        cfg = json.load(f)["pipe_params"]
    with open(args.scope_plan) as f:
        scope_plan = json.load(f)

    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=dtype)
    pipe = FluxPipeline.from_pretrained(args.model_path, transformer=None, torch_dtype=dtype)
    pipe.transformer = transformer
    pipe.scheduler.use_dynamic_shifting = False
    pipe.to("cuda")

    for index, prompt in enumerate(load_prompts(args)):
        print(f"[{index + 1}] {prompt}")
        images = pipe(
            prompt=prompt,
            height=1024, width=1024,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            ntk_factor=cfg["ntk_factor"][:n_stages],
            proportional_attention=True,
            text_duplication=False,
            swin_pachify=False,
            target_heights=list(args.resolutions),
            target_widths=list(args.resolutions),
            num_inference_steps_highres=cfg["num_inference_steps_highres"][:n_stages],
            filter_ratio=[0.2] * n_stages,
            guidance_scale_highres=[4.5, 6][:n_stages],
            structure_guidance="fft",
            alphas=[1.0, 0.25][:n_stages],
            betas=[0.5] * n_stages,
            upsampling_choice="latent",
            flow_choice="v_theta",
            generator=torch.Generator("cuda").manual_seed(args.seed),
            # HAP on every high-resolution step, SPA on the leading steps
            scope_plan=scope_plan,
            hap_step_ranges=[[0, 999]] * n_stages,
            hap_anchor_stride=args.global_anchor,
            spa_step_ranges=[[0, s] for s in spa_steps],
            group_nums=[args.group_num] * n_stages,
            save_dir=args.output_dir,
            img_index=args.prompt_offset + index,
        )[0]
        for offset, image in enumerate(images):
            path = os.path.join(args.output_dir,
                                f"{args.prompt_offset + index + offset}.png")
            image.save(path)
            print(f"    saved {path}  ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
