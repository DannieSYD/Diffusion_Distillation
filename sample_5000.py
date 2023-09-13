import argparse
import importlib
from v_diffusion import make_beta_schedule
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
import os
from train_utils import make_visualization
import cv2


def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--out_file", help="Path to image.", type=str, default="")
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
    parser.add_argument("--clipped_sampling", help="Use clipped sampling mode.", type=bool, default=False)
    parser.add_argument("--clipping_value", help="Noise clipping value.", type=float, default=1.2)
    parser.add_argument("--eta", help="Amount of random noise in clipping sampling mode(recommended non-zero values only for not distilled model).", type=float, default=0)
    return parser

def sample_images_5000(args, make_model):
    device = torch.device("cuda")

    teacher_ema = make_model().to(device)

    def make_diffusion(args, model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("v_diffusion")
        D = getattr(M, args.diffusion)  # 什么类型的diffusion？
        sampler = "ddpm"
        if args.clipped_sampling:
            sampler = "clipped"
        return D(model, betas, time_scale=time_scale, sampler=sampler)

    teacher = make_model().to(device)  # teacher是一个UNet对象，还不是diffusion对象

    ckpt = torch.load(args.checkpoint)  # 载入checkpoint
    teacher.load_state_dict(ckpt["G"])  # G是所有的模型参数
    n_timesteps = ckpt["n_timesteps"]//args.time_scale  # n_timesteps是采样步长数
    time_scale = ckpt["time_scale"]*args.time_scale  # 一个时间步长的大小
    del ckpt  # 参数传完，删掉
    print("Model loaded.")

    teacher_diffusion = make_diffusion(args, teacher, n_timesteps, time_scale, device)  # 这是一个diffusion对象
    image_size = deepcopy(teacher.image_size)
    image_size[0] = 1
    
    for i in range(10):
        print(f"{i+1}/5000")
        img = make_visualization(teacher_diffusion, device, image_size, need_tqdm=True, eta=args.eta, clip_value=args.clipping_value)
        if img.shape[2] == 1:
            img = img[:, :, 0]
        filename = os.path.join("images/celeba/base_0", f"celeba_full_0_{i:04}.png")
        cv2.imwrite(filename, img)

    print("Finished.")

parser = make_argument_parser()  # 创建一个ArgumentParser对象

args = parser.parse_args()  # 解析parser对象中的命令行参数，参数存在args对象中

M = importlib.import_module(args.module)  # 导入命令行参数module指定的模块
make_model = getattr(M, "make_model")  # 从celeba_u.py中获取make_model属性，里面定义了网络的架构

sample_images_5000(args, make_model)  # 把其他命令行参数和celeba_u对应的网络架构相关的参数传入采样函数
