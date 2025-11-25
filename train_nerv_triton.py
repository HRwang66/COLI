from __future__ import print_function

import argparse
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm
import warnings

from model_nerv import CustomDataSet, Generator
from utils import *

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def main():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--vid', default=[None], type=int, nargs='+', help='video id list for training')
    parser.add_argument('--scale', type=int, default=1, help='scale-up facotr for data transformation,  added to suffix!!!!')
    parser.add_argument('--frame_gap', type=int, default=1, help='frame selection gap')
    parser.add_argument('--augment', type=int, default=0, help='augment frames between frames,  added to suffix!!!!')
    parser.add_argument('--dataset', type=str, default='tiles_36', help='dataset')
    parser.add_argument('--test_gap', default=1, type=int, help='evaluation gap')

    # NERV architecture parameters
    parser.add_argument('--embed', type=str, default='1.25_40', help='base value/embed length for position encoding')

    # FC + Conv parameters（保持与你给的一致）
    parser.add_argument('--stem_dim_num', type=str, default='512_1', help='hidden dimension and length')
    parser.add_argument('--fc_hw_dim', type=str, default='9_16_26', help='out size (h,w) for mlp')
    parser.add_argument('--expansion', type=float, default=3, help='channel expansion from fc to conv')
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 2, 2, 2, 2], help='strides list')
    parser.add_argument('--num-blocks', type=int, default=1)

    parser.add_argument('--norm', default='none', type=str, choices=['none', 'bn', 'in'], help='norm layer for generator')
    parser.add_argument('--act', type=str, default='swish',
                        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower-width', type=int, default=128, help='lowest channel width for output feature maps')
    parser.add_argument("--single_res", action='store_true', default=True, help='single resolution,  added to suffix!!!!')
    parser.add_argument("--conv_type", default='conv', type=str, choices=['conv', 'deconv', 'bilinear'],
                        help='upscale methods')

    # General training setups
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not_resume_epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=15000, help='number of epochs to train for')
    parser.add_argument('--cycles', type=int, default=1, help='epoch cycles for training')
    parser.add_argument('--warmup', type=float, default=0.02, help='warmup epoch ratio compared to the epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_type', type=str, default='cosine', help='learning rate type')
    parser.add_argument('--lr_steps', default=[], type=float, nargs="+", metavar='LRSteps',
                        help='epochs to decay learning rate by 10')
    parser.add_argument('--beta', type=float, default=0.9, help='beta for adam.')
    parser.add_argument('--loss_type', type=str, default='L2', help='loss type')
    parser.add_argument('--lw', type=float, default=1.0, help='loss weight')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency')
    parser.add_argument('--quant_bit', type=int, default=-1, help='bit length for model quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_images', action='store_true', help='dump the prediction images')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')

    # pruning paramaters
    parser.add_argument('--prune_steps', type=float, nargs='+', default=[0., ], help='prune steps')
    parser.add_argument('--prune_ratio', type=float, default=1.0, help='pruning ratio')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:9888', type=str, help='url for distributed')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training')

    # logging, output directory,
    parser.add_argument('--debug', action='store_true', help='debug status, earlier for train/eval')
    parser.add_argument('-p', '--print-freq', default=50, type=int)
    parser.add_argument('--weight', default='./output/tiles_36_ab_triton_4/tiles_36/embed1.25_40_512_1_fc_9_16_26__exp3_reduce2_low128_blk1_cycle1_gap1_e15000_warm300_b1_conv_lr0.0005_cosine_L2_Strd5,2,2,2,2_SinRes_actswish__tcomp_amp/model_train_best.pth', type=str, help='pretrained weights for initialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='tiles_36_ab_triton_4', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")

    # ✅ 新增开关
    parser.add_argument('--amp', action='store_true', help='enable AMP mixed precision')
    parser.add_argument('--triton', action='store_true', help='enable torch.compile (Triton backend)')
    # 可选：梯度裁剪（默认不启用，保持原逻辑）
    parser.add_argument('--grad_clip', type=float, default=0.0, help='max grad norm; 0 for disable')

    # ✅ 新增：强制输出尺寸（默认关闭）
    parser.add_argument('--out_h', type=int, default=808, help='force output height; >0 to enable')
    parser.add_argument('--out_w', type=int, default=1285, help='force output width; >0 to enable')

    args = parser.parse_args()

    args.warmup = int(args.warmup * args.epochs)

    print(args)
    torch.set_printoptions(precision=4)

    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    if args.prune_ratio < 1 and not args.eval_only:
        prune_str = '_Prune{}_{}'.format(args.prune_ratio, ','.join([str(x) for x in args.prune_steps]))
    else:
        prune_str = ''
    extra_str = '_Strd{}_{}Res{}{}'.format(','.join([str(x) for x in args.strides]),
                                           'Sin' if args.single_res else f'_lw{args.lw}_multi',
                                           '_dist' if args.distributed else '', f'_eval' if args.eval_only else '')
    norm_str = '' if args.norm == 'none' else args.norm

    exp_id = f'{args.dataset}/embed{args.embed}_{args.stem_dim_num}_fc_{args.fc_hw_dim}__exp{args.expansion}_reduce{args.reduction}_low{args.lower_width}_blk{args.num_blocks}_cycle{args.cycles}' + \
             f'_gap{args.frame_gap}_e{args.epochs}_warm{args.warmup}_b{args.batchSize}_{args.conv_type}_lr{args.lr}_{args.lr_type}' + \
             f'_{args.loss_type}{norm_str}{extra_str}{prune_str}'
    exp_id += f'_act{args.act}_{args.suffix}'
    # 标注是否启用 triton/amp（不影响文件夹结构，只是更清晰）
    if args.triton:
        exp_id += '_tcomp'
    if args.amp:
        exp_id += '_amp'

    args.exp_id = exp_id
    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)
    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method = f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2)
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)


# =========================
# 只为“指标计算”增加稳健版本（保持你给的实现）
# =========================
def _infer_data_range(tensors):
    def _max_val(t):
        try:
            return float(t.detach().max().item())
        except Exception:
            return float(t.max().item())
    flat = []
    for x in tensors:
        if isinstance(x, (list, tuple)):
            flat += list(x)
        else:
            flat.append(x)
    m = max(_max_val(t) for t in flat)
    return 255.0 if m > 1.5 else 1.0


def psnr_fn_safe(output_list, target_list, force_data_range=None, eps=1e-10):
    dr = force_data_range if force_data_range is not None else _infer_data_range([output_list, target_list])
    vals = []
    for yhat, y in zip(output_list, target_list):
        yh = yhat.detach().to(torch.float32).clamp(0.0, dr)
        yt = y.detach().to(torch.float32).clamp(0.0, dr)
        mse = (yh - yt).pow(2).mean(dim=(1, 2, 3)).clamp_min(eps)
        psnr = 10.0 * torch.log10((dr ** 2) / mse)
        vals.append(psnr)
    out = torch.stack(vals, dim=1)
    return torch.nan_to_num(out, nan=0.0, posinf=99.0, neginf=0.0)


def msssim_fn_safe(output_list, target_list, force_data_range=None):
    dr = force_data_range if force_data_range is not None else _infer_data_range([output_list, target_list])
    scale = 1.0 / dr
    outs = [yhat.detach().to(torch.float32).mul(scale).clamp(0.0, 1.0) for yhat in output_list]
    tgts = [y.detach().to(torch.float32).mul(scale).clamp(0.0, 1.0) for y in target_list]
    try:
        val = msssim_fn(outs, tgts)
    except Exception:
        b = outs[-1].shape[0]
        val = torch.zeros(b, len(outs), device=outs[-1].device)
    return torch.nan_to_num(val, nan=1.0, posinf=1.0, neginf=0.0)


def maybe_compile(model, args, local_rank):
    """
    只在安全的场景开启 torch.compile：
    - 单卡：OK
    - DDP：OK（先to(device)，后compile，再包DDP）
    - DataParallel：禁用（容易出问题），给出提示
    """
    if not args.triton:
        return model, False

    if args.distributed:
        # 在 train() 里，已经 .to(device) 了
        try:
            compiled = torch.compile(model, backend='inductor', dynamic=True, fullgraph=False)
            print("⚡ Using torch.compile(backend='inductor', dynamic=True, fullgraph=False) [DDP safe]")
            return compiled, True
        except Exception as e:
            warnings.warn(f"[compile] Fallback to eager due to: {e}")
            return model, False
    else:
        if args.ngpus_per_node > 1:
            warnings.warn("DataParallel + torch.compile 不稳定，已自动禁用 --triton。建议改用 --distributed。")
            return model, False
        # 单卡
        try:
            compiled = torch.compile(model, backend='inductor', dynamic=True, fullgraph=False)
            print("⚡ Using torch.compile(backend='inductor', dynamic=True, fullgraph=False) [single GPU]")
            return compiled, True
        except Exception as e:
            warnings.warn(f"[compile] Fallback to eager due to: {e}")
            return model, False


def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]
    is_train_best, is_val_best = False, False

    PE = PositionalEncoding(args.embed)
    args.embed_length = PE.embed_length
    model = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim,
                      expansion=args.expansion, num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias=True,
                      reduction=args.reduction, conv_type=args.conv_type, stride_list=args.strides,
                      sin_res=args.single_res, lower_width=args.lower_width, sigmoid=args.sigmoid)

    # ----- prune (保持原样) -----
    prune_net = args.prune_ratio < 1
    if prune_net:
        param_list = []
        for k, v in model.named_parameters():
            if 'weight' in k:
                if 'stem' in k:
                    stem_ind = int(k.split('.')[1])
                    param_list.append(model.stem[stem_ind])
                elif 'layers' in k[:6] and 'conv' in k:
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.layers[layer_ind].conv.conv)
        param_to_prune = [(ele, 'weight') for ele in param_list]
        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]
        prune_num = 0
        if args.eval_only:
            prune.global_unstructured(param_to_prune, pruning_method=prune.L1Unstructured,
                                      amount=1 - prune_base_ratio ** prune_num)

    # ----- params log -----
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    if local_rank in [0, None]:
        params = total_params
        print(f'{args}\n {model}\n Model Params: {params}M')
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'Params: {params}M\n')
        writer = SummaryWriter(os.path.join(args.outf, f'param_{total_params}M', 'tensorboard'))
    else:
        writer = None

    # ----- move to GPU & parallel -----
    print("Use GPU: {} for training".format(local_rank))
    compiled_flag = False
    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        # 先 to(device)，再 compile，最后包 DDP
        model = model.to(local_rank)
        model, compiled_flag = maybe_compile(model, args, local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    elif args.ngpus_per_node > 1:
        # DataParallel：不编译（上面说明原因）
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
        model, compiled_flag = maybe_compile(model, args, local_rank)

    if compiled_flag and hasattr(model, "_orig_mod"):
        # 打印下真实结构（可选）
        print(model)

    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))

    # AMP scaler
    scaler = GradScaler(enabled=args.amp)

    # ----- resume from args.weight -----
    checkpoint = None
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        orig_ckt = checkpoint['state_dict']
        new_ckt = {k.replace('blocks.0.', ''): v for k, v in orig_ckt.items()}
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt = {k.replace('module.', ''): v for k, v in new_ckt.items()}
            model.load_state_dict(new_ckt)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt)
        else:
            model.load_state_dict(new_ckt)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))

    # ----- resume from model_latest -----
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if prune_net:
            prune.global_unstructured(param_to_prune, pruning_method=prune.L1Unstructured,
                                      amount=1 - prune_base_ratio ** prune_num)
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}')
        # DDP/DP 兼容 load
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except Exception:
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                raise
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    args.start_epoch = 0
    if checkpoint is not None:
        args.start_epoch = checkpoint.get('epoch', 0)
        # 兼容 tensor / 数值
        def _to_device(x):
            return x.to(torch.device(loc)) if torch.is_tensor(x) else torch.tensor(float(x)).to(torch.device(loc))
        train_best_psnr = _to_device(checkpoint.get('train_best_psnr', 0))
        train_best_msssim = _to_device(checkpoint.get('train_best_msssim', 0))
        val_best_psnr = _to_device(checkpoint.get('val_best_psnr', 0))
        val_best_msssim = _to_device(checkpoint.get('val_best_msssim', 0))
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.not_resume_epoch:
        args.start_epoch = 0

    # ----- dataloader -----
    img_transforms = transforms.ToTensor()
    DataSet = CustomDataSet
    train_data_dir = f'./data/{args.dataset.lower()}'
    val_data_dir = f'./data/{args.dataset.lower()}'

    train_dataset = DataSet(train_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.frame_gap)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=worker_init_fn
    )

    val_dataset = DataSet(val_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.test_gap)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchSize, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False,
        worker_init_fn=worker_init_fn
    )
    data_size = len(train_dataset)

    if args.eval_only:
        print('Evaluation ...')
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        print_str = f'{time_str}\t Results for checkpoint: {args.weight}\n'
        if prune_net:
            for param in param_to_prune:
                prune.remove(param[0], param[1])
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print_str += f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}\n'

        val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)
        print_str += f'PSNR/ms_ssim on validate set for bit {args.quant_bit} with axis {args.quant_axis}: {round(val_psnr.item(),2)}/{round(val_msssim.item(),4)}'
        print(print_str)
        with open('{}/eval.txt'.format(args.outf), 'a') as f:
            f.write(print_str + '\n\n')
        return

    # ----- training loop -----
    start = datetime.now()
    total_epochs = args.epochs * args.cycles
    for epoch in range(args.start_epoch, total_epochs):
        model.train()

        if prune_net and epoch in args.prune_steps:
            prune_num += 1
            prune.global_unstructured(param_to_prune, pruning_method=prune.L1Unstructured,
                                      amount=1 - prune_base_ratio ** prune_num)
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{epoch}: {sparisity_num / 1e6 / total_params}')

        epoch_start_time = datetime.now()
        psnr_list = []
        msssim_list = []

        for i, (data, norm_idx) in enumerate(train_dataloader):
            if i > 10 and args.debug:
                break
            embed_input = PE(norm_idx)
            if local_rank is not None:
                data = data.cuda(local_rank, non_blocking=True)
                embed_input = embed_input.cuda(local_rank, non_blocking=True)
            else:
                data, embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)

            # ===== forward (保持原损失逻辑) =====
            with autocast(enabled=args.amp):
                output_list = model(embed_input)
                # ✅ 新增：强制把所有 stage 的输出统一到指定尺寸（若开启）
                if args.out_h > 0 and args.out_w > 0:
                    output_list = [F.interpolate(o, size=(args.out_h, args.out_w),
                                                 mode='bilinear', align_corners=False)
                                   for o in output_list]
                target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
                loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
                loss_list = [loss_list[j] * (args.lw if j < len(loss_list) - 1 else 1) for j in range(len(loss_list))]
                loss_sum = sum(loss_list)

            lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)
            optimizer.zero_grad(set_to_none=True)

            # 若出现 NaN/Inf，跳过这步，避免参数被写坏（无侵入保护）
            if not torch.isfinite(loss_sum):
                warnings.warn(f"[skip step] non-finite loss at epoch {epoch+1}, iter {i+1}: {loss_sum.item()}")
                continue

            # ===== backward =====
            if args.amp:
                scaler.scale(loss_sum).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_sum.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            # ===== metrics（与你的安全版一致） =====
            psnr_list.append(psnr_fn_safe(output_list, target_list, force_data_range=1.0))
            msssim_list.append(msssim_fn_safe(output_list, target_list, force_data_range=1.0))

            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                train_psnr = torch.cat(psnr_list, dim=0)
                train_psnr = torch.mean(train_psnr, dim=0)
                train_msssim = torch.cat(msssim_list, dim=0)
                train_msssim = torch.mean(train_msssim.float(), dim=0)
                time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}'.format(
                    time_now_string, local_rank, epoch + 1, args.epochs, i + 1, len(train_dataloader), lr,
                    RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')

        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            train_psnr = all_reduce([train_psnr.to(local_rank)])
            train_msssim = all_reduce([train_msssim.to(local_rank)])

        if local_rank in [0, None]:
            h, w = output_list[-1].shape[-2:]
            is_train_best = train_psnr[-1] > train_best_psnr
            train_best_psnr = train_psnr[-1] if train_psnr[-1] > train_best_psnr else train_best_psnr
            train_best_msssim = train_msssim[-1] if train_msssim[-1] > train_best_msssim else train_best_msssim
            writer.add_scalar(f'Train/PSNR_{h}X{w}_gap{args.frame_gap}', train_psnr[-1].item(), epoch + 1)
            writer.add_scalar(f'Train/MSSSIM_{h}X{w}_gap{args.frame_gap}', train_msssim[-1].item(), epoch + 1)
            writer.add_scalar(f'Train/best_PSNR_{h}X{w}_gap{args.frame_gap}', train_best_psnr.item(), epoch + 1)
            writer.add_scalar(f'Train/best_MSSSIM_{h}X{w}_gap{args.frame_gap}', train_best_msssim, epoch + 1)
            print_str = '\t{}p: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(
                h, train_psnr[-1].item(), train_best_psnr.item(), train_best_msssim.item())
            print(print_str, flush=True)
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
            writer.add_scalar('Train/lr', lr, epoch + 1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format(
                (epoch_end_time - epoch_start_time).total_seconds(),
                (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch)))

        # save
        state_dict = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
        save_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'train_best_psnr': train_best_psnr,
            'train_best_msssim': train_best_msssim,
            'val_best_psnr': val_best_psnr,
            'val_best_msssim': val_best_msssim,
            'optimizer': optimizer.state_dict(),
        }

        # eval
        if (epoch + 1) % args.eval_freq == 0 or epoch > total_epochs - 10:
            val_start_time = datetime.now()
            val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)
            val_end_time = datetime.now()
            if args.distributed and args.ngpus_per_node > 1:
                val_psnr = all_reduce([val_psnr.to(local_rank)])
                val_msssim = all_reduce([val_msssim.to(local_rank)])
            if local_rank in [0, None]:
                h, w = output_list[-1].shape[-2:]
                print_str = f'Eval best_PSNR at epoch{epoch + 1}:'
                is_val_best = val_psnr[-1] > val_best_psnr
                val_best_psnr = val_psnr[-1] if is_val_best else val_best_psnr
                val_best_msssim = val_msssim[-1] if val_msssim[-1] > val_best_msssim else val_best_msssim
                writer.add_scalar(f'Val/PSNR_{h}X{w}_gap{args.test_gap}', val_psnr[-1], epoch + 1)
                writer.add_scalar(f'Val/MSSSIM_{h}X{w}_gap{args.test_gap}', val_msssim[-1], epoch + 1)
                writer.add_scalar(f'Val/best_PSNR_{h}X{w}_gap{args.test_gap}', val_best_psnr, epoch + 1)
                writer.add_scalar(f'Val/best_MSSSIM_{h}X{w}_gap{args.test_gap}', val_best_msssim, epoch + 1)
                print_str += '\t{}p: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}\t Time/epoch: {:.2f}'.format(
                    h, val_psnr[-1].item(), val_best_psnr.item(), val_best_msssim.item(),
                    (val_end_time - val_start_time).total_seconds())
                print(print_str)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
                if is_val_best:
                    torch.save(save_checkpoint, '{}/model_val_best.pth'.format(args.outf))

        if local_rank in [0, None]:
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if is_train_best:
                torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))

    print("Training complete in: " + str(datetime.now() - start))


@torch.no_grad()
def evaluate(model, val_dataloader, pe, local_rank, args):
    # quant（保持原逻辑）
    if args.quant_bit != -1:
        cur_ckt = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
        from dahuffman import HuffmanCodec
        quant_weitht_list = []
        for k, v in cur_ckt.items():
            large_tf = (v.dim() in {2, 4} and 'bias' not in k)
            quant_v, new_v = quantize_per_tensor(v, args.quant_bit, args.quant_axis if large_tf else -1)
            valid_quant_v = quant_v[v != 0]
            quant_weitht_list.append(valid_quant_v.flatten())
            cur_ckt[k] = new_v
        cat_param = torch.cat(quant_weitht_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))
        codec = HuffmanCodec.from_data(input_code_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        avg_bits = total_bits / len(input_code_list)
        encoding_efficiency = avg_bits / args.quant_bit
        print_str = f'Entropy encoding efficiency for bit {args.quant_bit}: {encoding_efficiency}'
        print(print_str)
        if local_rank in [0, None]:
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
        if hasattr(model, 'module'):
            model.module.load_state_dict(cur_ckt)
        else:
            model.load_state_dict(cur_ckt)

    psnr_list = []
    msssim_list = []
    if args.dump_images:
        from torchvision.utils import save_image
        visual_dir = f'{args.outf}/visualize'
        print(f'Saving predictions to {visual_dir}')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)

    time_list = []
    model.eval()
    for i, (data, norm_idx) in enumerate(val_dataloader):
        if i > 10 and args.debug:
            break
        embed_input = pe(norm_idx)
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            embed_input = embed_input.cuda(local_rank, non_blocking=True)
        else:
            data, embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)

        fwd_num = 10 if args.eval_fps else 1
        for _ in range(fwd_num):
            start_time = datetime.now()
            with autocast(enabled=args.amp):
                output_list = model(embed_input)
                # ✅ 新增：验证同样按需要插值到固定尺寸
                if args.out_h > 0 and args.out_w > 0:
                    output_list = [F.interpolate(o, size=(args.out_h, args.out_w),
                                                 mode='bilinear', align_corners=False)
                                   for o in output_list]
            torch.cuda.synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

        if args.dump_images:
            from torchvision.utils import save_image
            for batch_ind in range(args.batchSize):
                full_ind = i * args.batchSize + batch_ind
                # 保存时转回 fp32
                save_image(output_list[-1][batch_ind].float(), f'{visual_dir}/pred_{full_ind}.png')
                save_image(data[batch_ind].float(), f'{visual_dir}/gt_{full_ind}.png')

        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        psnr_list.append(psnr_fn_safe(output_list, target_list, force_data_range=1.0))
        msssim_list.append(msssim_fn_safe(output_list, target_list, force_data_range=1.0))

        val_psnr = torch.cat(psnr_list, dim=0)
        val_psnr = torch.mean(val_psnr, dim=0)
        val_msssim = torch.cat(msssim_list, dim=0)
        val_msssim = torch.mean(val_msssim.float(), dim=0)
        if i % args.print_freq == 0:
            fps = fwd_num * (i + 1) * args.batchSize / sum(time_list)
            print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {}'.format(
                local_rank, i + 1, len(val_dataloader),
                RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False), round(fps, 2))
            print(print_str)
            if local_rank in [0, None]:
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
    model.train()

    return val_psnr, val_msssim


if __name__ == '__main__':
    main()