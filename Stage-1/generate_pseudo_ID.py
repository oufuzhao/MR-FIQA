import argparse
import sys
import os
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import torch as th
import torch.distributed as dist

from diffusion import dist_util, logger
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from diffusion.image_datasets import load_data
from torchvision import utils
import math


def main(batch_size, num_samples, save_dir):
    if len(sys.argv) == 1:
        sys.argv += [
            "--batch_size", str(batch_size),
            "--num_samples", str(num_samples),
            "--save_dir", save_dir,
            "--attention_resolutions", "16",
            "--class_cond", "False",
            "--diffusion_steps", "1000",
            "--image_size", "256",
            "--learn_sigma", "True",
            "--noise_schedule", "linear",
            "--num_channels", "128",
            "--num_head_channels", "64",
            "--num_res_blocks", "1",
            "--resblock_updown", "True",
            "--use_fp16", "False",
            "--use_scale_shift_norm", "True",
            "--timestep_respacing", "100",
            "--down_N", "32",
            "--range_t", "20"
        ]
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    logger.log(f"------------------------------------------")
    logger.log("# Creating model")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/Pretrained-Models/Unc-Diff-Stage1.pth', map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    assert math.log(args.down_N, 2).is_integer()
    logger.log("# Generating samples...")
    logger.log(f"------------------------------------------")
    count = 0
    while count * args.batch_size < args.num_samples:
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs={},
            range_t=args.range_t
        )

        for i in range(args.batch_size):
            tmp_num = count * args.batch_size + i
            save_name = f"{str(tmp_num).zfill(5)}.png"
            out_path = os.path.join(logger.get_dir(), save_name)
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True
            )
        count += 1

        logger.log(f"> Generated {count * args.batch_size} samples")

    dist.barrier()
    logger.log("Complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=0,
        batch_size=0,
        down_N=32,
        range_t=0,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_dir="",
        save_latents=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    batch_size = 24
    num_samples = 200000
    save_dir = "Stage-1/samples"
    main(batch_size, num_samples, save_dir)
