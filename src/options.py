import argparse
import os
import pathlib
from typing import Tuple

# Helpers
def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def base_parser():
    parser = argparse.ArgumentParser()

    # Basic training settings
    parser.add_argument('--model', type=str, required=True, help='Model to use: MoCE_IR or MoCE_IR_S.')
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--de_type', nargs='+', help='Degradation types for training/testing.')
    parser.add_argument('--trainset', default="standard", help=["standard", "CDD11_all", "CDD11_single", "CDD11_double", "CDD11_triple"])
    parser.add_argument('--loss_type', default="L1", help='Loss type.')
    parser.add_argument('--patch_size', type=int, default=128, help='Input patch size.')
    parser.add_argument('--balance_loss_weight', type=float, default=0.01, help='Balance loss weight.')
    parser.add_argument('--fft_loss_weight', type=float, default=1.0, help='FFT loss weight.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers.')
    parser.add_argument('--accum_grad', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint.')
    parser.add_argument('--fine_tune_from', type=str, default=None, help='Fine-tune from checkpoint.')
    parser.add_argument('--checkpoint_id', type=str, help='checkpoint id')
    parser.add_argument('--benchmarks', nargs='+', help='which benchmarks to test on.')
    parser.add_argument('--save_results', action="store_true", help="Save restored outputs.")

    # Paths
    parser.add_argument('--data_file_dir', type=str, default=os.path.join(pathlib.Path.home(), "datasets"), help='Path to datasets.')
    parser.add_argument('--output_path', type=str, default="output/", help='Output save path.')
    parser.add_argument('--wblogger', action="store_true", help='Log to Weights & Biases.')
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints", help='Checkpoint directory.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for training.')

    return parser

def moce_ir_s(parser):
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8])
    parser.add_argument('--num_dec_blocks', nargs='+', type=int, default=[2, 4, 4])
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--num_exp_blocks', type=int, default=4)
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--stage_depth', nargs='+', type=int, default=[1, 1, 1])
    parser.add_argument('--with_complexity', action="store_true")
    parser.add_argument('--complexity_scale', type=str, default="max")
    parser.add_argument('--rank_type', default="spread")
    parser.add_argument('--depth_type', default="constant")
    parser.add_argument('--topk', type=int, default=1)
    return parser

def moce_ir(parser):
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8])
    parser.add_argument('--num_dec_blocks', nargs='+', type=int, default=[2, 4, 4])
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--num_exp_blocks', type=int, default=4)
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--stage_depth', nargs='+', type=int, default=[1, 1, 1])
    parser.add_argument('--with_complexity', action="store_true")
    parser.add_argument('--complexity_scale', type=str, default="max")
    parser.add_argument('--rank_type', default="spread")
    parser.add_argument('--depth_type', default="constant")
    parser.add_argument('--topk', type=int, default=1)
    return parser

def train_options():
    base_args = base_parser().parse_known_args()[0]
    
    if base_args.model == "MoCE_IR_S":
        parser = moce_ir_s(base_parser())
    elif base_args.model == "MoCE_IR":
        parser = moce_ir(base_parser())
    else:
        raise NotImplementedError(f"Model '{base_args.model}' not found.")

    options = parser.parse_args()

    # Adjust batch size if gradient accumulation is used
    if options.accum_grad > 1:
        options.batch_size = options.batch_size // options.accum_grad

    return options