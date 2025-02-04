"""
A script for training a diffusion model to unconditional image generation.
"""

import argparse
import numpy as np
import random
import sys
import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.lidcloader import LIDCVolumes
from guided_diffusion.metabreast import NCDataset
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          args_to_dict,
                                          add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# ============================================================================================

# Non-Conditional 3D Diffusion Model Parser Initialization
if True:
    ncdiff_parser = argparse.ArgumentParser(
        description = "Non-Conditional 3D Diffusion Model")
    ncdiff_parser.add_argument('--model_type', type = str,            # Chosen Model / Diffusion
                                choices =  {'video_diffusion',
                                            'blackout_diffusion',
                                            'gamma_diffusion'},
                                default = 'video_diffusion')
    ncdiff_parser.add_argument('--model_version', type = int,         # Model Version Index
                                default = 0)
    ncdiff_parser.add_argument('--data_version', type = int,          # Dataset Version Index
                                default = 4)
    ncdiff_parser.add_argument('--noise_type', type = str,            # Diffusion Noise Distribution
                                default = 'gaussian')
    settings = ncdiff_parser.parse_args("")

    # ============================================================================================

    # Directories and Path Arguments
    ncdiff_parser.add_argument('--reader_folderpath', type = str,         # Path for Dataset Reader Directory
                                default = '../MetaBreast/data/non_cond')

    ncdiff_parser.add_argument('--public_data_folderpath', type = str,    # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
                                default = "../../../datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
    ncdiff_parser.add_argument('--private_data_folderpath', type = str,   # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/METABREST/T1W_Breast")
                                default = '../../../datasets/private/METABREST/T1W_Breast')
    ncdiff_parser.add_argument( '--lung_data_folderpath', type = str,     # Path for LUCAS Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")
                                default = "../../../datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")

    # Directory | Model-Related Path Arguments
    ncdiff_parser.add_argument('--model_folderpath', type = str,          # Path for Model Architecture Directory
                                default = f'../../models/{settings.model_type}')
    ncdiff_parser.add_argument('--script_folderpath', type = str,         # Path for Model Training & Testing Scripts Directory
                                default = f'../../scripts/{settings.model_type}')
    ncdiff_parser.add_argument('--logs_folderpath', type = str,           # Path for Model Saving Directory
                                default = f'../../logs/{settings.model_type}')
        
    # ============================================================================================

    # Dataset | Dataset General Arguments
    ncdiff_parser.add_argument('--data_format', type = str,           # Chosen Dataset Format for Reading
                                choices =  {'mp4', 'dicom'},
                                default = 'dicom')
    ncdiff_parser.add_argument('--img_size', type = int,              # Generated Image Resolution
                                default = 128)
    ncdiff_parser.add_argument('--num_slice', type = int,             # Number of 2D Slices in MRI
                                default = 32)
    ncdiff_parser.add_argument('--slice_spacing', type = bool,        # Usage of Linspace for Slice Spacing
                                default = True)
    ncdiff_parser.add_argument('--slice_bottom_margin', type = int,   # Number of 2D Slices to be Discarded in Bottom Margin
                                default = 0)
    ncdiff_parser.add_argument('--slice_top_margin', type = int,      # Number of 2D Slices to be Discarded in Top Margin
                                default = 0)
    ncdiff_parser.add_argument('--data_prep', type = bool,            # Usage of Dataset Pre-Processing Control Value
                                default = True)
    ncdiff_parser.add_argument('--h_flip', type = int,                # Percentage of Horizontally Flipped Subjects
                                default = 0)

    # Dataset | Dataset Splitting Arguments
    ncdiff_parser.add_argument('--train_subj', type = int,            # Number of Random Subjects in Training Set
                                default = 0)                          # PS: Input 0 for all Subjects in the Dataset
    ncdiff_parser.add_argument('--val_subj', type = int,              # Number of Random Subjects in Validation Set
                                default = 0)
    ncdiff_parser.add_argument('--test_subj', type = int,             # Number of Random Subjects in Test Set
                                default = 0)

    # Dataset | DataLoader Arguments
    ncdiff_parser.add_argument('--batch_size', type = int,            # DataLoader Batch Size Value
                                default = 1)
    ncdiff_parser.add_argument('--shuffle', type = bool,              # DataLoader Subject Shuffling Control Value
                                default = False)
    ncdiff_parser.add_argument('--num_workers', type = int,           # Number of DataLoader Workers
                                default = 12)
    ncdiff_parser.add_argument('--num_fps', type = int,               # Number of Video Frames per Second
                                default = 4)

    # ============================================================================================

    # Model | Architecture-Defining Arguments
    ncdiff_parser.add_argument('--seed', type = int,                  # Randomised Generational Seed
                                default = 0)
    ncdiff_parser.add_argument('--dim', type = int,                   # Input Dimensionality (Not Necessary)
                                default = 64)
    ncdiff_parser.add_argument('--num_channel', type = int,           # Number of Input Channels for Dataset
                                default = 1)
    ncdiff_parser.add_argument('--mult_dim', type = tuple,            # Dimensionality for all Conditional Layers
                                default = (1, 2, 4, 8))

    # Model | Training & Diffusion Arguments
    #ncdiff_parser.add_argument('--num_epochs', type = int,            # Number of Training Epochs
    #                            default = 30)
    ncdiff_parser.add_argument('--num_ts', type = int,                # Number of Scheduler Timesteps
                                default = 500)
    ncdiff_parser.add_argument('--num_steps', type = int,             # Number of Diffusion Training Steps
                                default = 500000)
    ncdiff_parser.add_argument('--lr_base', type = float,             # Base Learning Rate Value
                                default = 1e-4)
    ncdiff_parser.add_argument('--save_interval', type = int,         # Number of Training Step Interval inbetween Image Saving
                                default = 1000)
    ncdiff_parser.add_argument('--save_img', type = int,              # Square Root of Number of Images Saved for Manual Evaluation
                                default = 2)

    # ============================================================================================

    settings = ncdiff_parser.parse_args("")
    settings.device = th.device('cuda:0' if th.cuda.is_available() else "cpu")

# ============================================================================================

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    dist_util.setup_dist(devices=args.devices)

    logger.log("Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)
    #model._save_to_state_dict('checkpoint.pth')
    import torchsummary
    #torchsummary.summary(model.cuda(), input_size = (8, 64, 64, 64))


    # logger.log("Number of trainable parameters: {}".format(np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    if args.dataset == 'brats':
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        ds = BRATSVolumes(args.data_dir, test_flag=False,
                          normalize=(lambda x: 2*x - 1) if args.renormalize else None,
                          mode='train',
                          img_size=args.image_size)

    elif args.dataset == 'lidc-idri':
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        ds = LIDCVolumes(args.data_dir, test_flag=False,
                         normalize=(lambda x: 2*x - 1) if args.renormalize else None,
                         mode='train',
                         img_size=args.image_size)
    
    elif args.dataset == 'metabreast':
        #private_ds = NCDataset( settings,
        #                        mode = 'train',
        #                        dataset = 'private')
        #public_ds =
        ds = NCDataset(  settings,
                                mode = 'train',
                                dataset = 'public')
        #ds = ConcatDataset([private_ds, public_ds])
    else:
        print("We currently just support the datasets: brats, lidc-idri")

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=True,
                                     )

    logger.log("Start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode='default',
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=1000,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='metabreast',
        use_tensorboard=True,
        tensorboard_path='',  # set path to existing logdir for resuming
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=12,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=True,
        #use_freq=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
