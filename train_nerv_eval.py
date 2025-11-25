from __future__ import print_function

import argparse
import os
import random
import shutil
from datetime import datetime
from lib.Compress_Params_Standard_uint_i_ClassCenterHyper_Multi_VersionC_matrix import compress_params, \
    decompress_params, find_nth_occurrence
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import time
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # ✅ AMP 混合精度
from model_nerv import CustomDataSet, Generator
from utils import *


def main():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--vid', default=[None], type=int, nargs='+', help='video id list for training')
    parser.add_argument('--scale', type=int, default=1,
                        help='scale-up facotr for data transformation,  added to suffix!!!!')
    parser.add_argument('--frame_gap', type=int, default=1, help='frame selection gap')
    parser.add_argument('--augment', type=int, default=0, help='augment frames between frames,  added to suffix!!!!')
    parser.add_argument('--dataset', type=str, default='tiles_36', help='dataset', )
    parser.add_argument('--test_gap', default=1, type=int, help='evaluation gap')

    # NERV architecture parameters
    # embedding parameters
    parser.add_argument('--embed', type=str, default='1.25_40', help='base value/embed length for position encoding')

    # FC + Conv parameters
    parser.add_argument('--stem_dim_num', type=str, default='512_1', help='hidden dimension and length')
    parser.add_argument('--fc_hw_dim', type=str, default='9_16_26', help='out size (h,w) for mlp')
    parser.add_argument('--expansion', type=float, default=3, help='channel expansion from fc to conv')
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 2, 2, 2, 2], help='strides list')
    parser.add_argument('--num-blocks', type=int, default=1)

    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator',
                        choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='swish', help='activation to use',
                        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower-width', type=int, default=128, help='lowest channel width for output feature maps')
    parser.add_argument("--single_res", action='store_true', default=True,
                        help='single resolution,  added to suffix!!!!')
    parser.add_argument("--conv_type", default='conv', type=str,
                        help='upscale methods, can add bilinear and deconvolution methods',
                        choices=['conv', 'deconv', 'bilinear'])

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not_resume_epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=15000, help='number of epochs to train for')
    parser.add_argument('--cycles', type=int, default=1, help='epoch cycles for training')
    parser.add_argument('--warmup', type=float, default=0.02,
                        help='warmup epoch ratio compared to the epochs, default=0.2,  added to suffix!!!!')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine', help='learning rate type, default=cosine')
    parser.add_argument('--lr_steps', default=[], type=float, nargs="+", metavar='LRSteps',
                        help='epochs to decay learning rate by 10,  added to suffix!!!!')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam. default=0.5,  added to suffix!!!!')
    parser.add_argument('--loss_type', type=str, default='L2', help='loss type, default=L2')
    parser.add_argument('--lw', type=float, default=1.0, help='loss weight,  added to suffix!!!!')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_only', default=True, action='store_true', help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_bit', type=int, default=-1, help='bit length for model quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_images', default=True, action='store_true', help='dump the prediction images')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')

    parser.add_argument('--out_h', type=int, default=808, help='force output height; >0 to enable')
    parser.add_argument('--out_w', type=int, default=1285, help='force output width; >0 to enable')

    # pruning paramaters
    parser.add_argument('--prune_steps', type=float, nargs='+', default=[0., ], help='prune steps')
    parser.add_argument('--prune_ratio', type=float, default=1.0, help='pruning ratio')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:9888', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('-d', '--distributed', action='store_true', default=False,
                        help='distributed training,  added to suffix!!!!')

    # logging, output directory,
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')
    parser.add_argument('-p', '--print-freq', default=50, type=int, )
    parser.add_argument('--weight',
                        default='./output/tiles_36_ab_triton_4/tiles_36/embed1.25_40_512_1_fc_9_16_26__exp3_reduce2_low128_blk1_cycle1_gap1_e15000_warm300_b1_conv_lr0.0005_cosine_L2_Strd5,2,2,2,2_SinRes_actswish__tcomp_amp/model_train_best.pth',
                        type=str,
                        help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='tiles_36_ab_triton_4', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")

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
                      expansion=args.expansion,
                      num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias=True, reduction=args.reduction,
                      conv_type=args.conv_type,
                      stride_list=args.strides, sin_res=args.single_res, lower_width=args.lower_width,
                      sigmoid=args.sigmoid)

    ##### prune model params and flops #####
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
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

    ##### get model params and flops #####
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    if local_rank in [0, None]:
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6

        print(f'{args}\n {model}\n Model Params: {params}M')
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'Params: {params}M\n')
        writer = SummaryWriter(os.path.join(args.outf, f'param_{total_params}M', 'tensorboard'))
    else:
        writer = None

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
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
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], \
                                                          output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))

    #######################
    model_name = 'celllll'

    rect_l = 0.2  # !!!!!!!!

    # encoding save path
    Save_Param_Path = f'./compressed_result/{model_name}_l_{str(rect_l)[0] + str(rect_l)[2:]}/'

    # decoding files path
    Decode_Param_Path = Save_Param_Path

    num_inner_list = np.fromfile(Decode_Param_Path + '/Compressed_Dir/num_inner_list.bin', dtype=np.uint64)
    rect_l_str = Decode_Param_Path[find_nth_occurrence(Decode_Param_Path, "/", 2):].split('_')[2]
    rect_l = float(rect_l_str[:1] + '.' + rect_l_str[1:-1])

    # 通过文件夹decode出来的model的所有参数张量 [tensor1, tensor2, ..... , tensorN]
    decode_params_list = decompress_params(model, Decode_Param_Path + '/Compressed_Dir/', rect_l, num_inner_list)

    params_list = list(model.parameters())
    # 将model params 换掉
    with torch.no_grad():
        for i in tqdm(range(len(params_list))):
            ori_param = params_list[i].data  # 原参数
            new_param = decode_params_list[i]  # 还原的新参数
            params_list[i].copy_(torch.tensor(new_param).float().cuda())  # 用还原参数替换原模型参数

    state_dict_back = model.state_dict()

    ########################

    # resume from args.weight
    checkpoint = None
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint['state_dict'] = state_dict_back
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

        # resume from model_latest
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if prune_net:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    args.start_epoch = 0
    if checkpoint is not None:
        args.start_epoch = checkpoint['epoch']
        train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
        train_best_msssim = checkpoint['train_best_msssim'].to(torch.device(loc))
        val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
        val_best_msssim = checkpoint['val_best_msssim'].to(torch.device(loc))
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.not_resume_epoch:
        args.start_epoch = 0

    # setup dataloader
    img_transforms = transforms.ToTensor()
    DataSet = CustomDataSet
    train_data_dir = f'./data/{args.dataset.lower()}'
    val_data_dir = f'./data/{args.dataset.lower()}'

    train_dataset = DataSet(train_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.frame_gap)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchSize,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    val_dataset = DataSet(val_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.test_gap)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
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
        print_str += f'PSNR/ms_ssim on validate set for bit {args.quant_bit} with axis {args.quant_axis}: {round(val_psnr.item(), 2)}/{round(val_msssim.item(), 4)}'
        print(print_str)
        with open('{}/eval.txt'.format(args.outf), 'a') as f:
            f.write(print_str + '\n\n')
        return

    # ======= Training =======
    scaler = GradScaler()

    start = datetime.now()
    total_epochs = args.epochs * args.cycles
    for epoch in range(args.start_epoch, total_epochs):
        model.train()

        if prune_net and epoch in args.prune_steps:
            prune_num += 1
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            sparisity_num = sum((param.weight == 0).sum() for param in param_list)
            print(f'Model sparsity at Epoch{epoch}: {sparisity_num / 1e6 / total_params}')

        epoch_start_time = datetime.now()
        psnr_list = []
        msssim_list = []
        batch_size = 1
        all_data = []
        all_norm_idx = []

        for data, norm_idx in train_dataloader:
            all_data.append(data)
            all_norm_idx.append(norm_idx)
            if args.debug and len(all_data) > 10:
                break

        num_patches = len(all_data)
        for i in range(0, num_patches, batch_size):
            batch_start = time.time()

            data_batch = all_data[i:i + batch_size]
            norm_idx_batch = all_norm_idx[i:i + batch_size]

            norm_idx_batch_tensor = torch.cat(norm_idx_batch, dim=0)
            if local_rank is not None:
                norm_idx_batch_tensor = norm_idx_batch_tensor.cuda(local_rank, non_blocking=True)
            else:
                norm_idx_batch_tensor = norm_idx_batch_tensor.cuda(non_blocking=True)

            pe_start = time.time()
            embed_input = PE(norm_idx_batch_tensor)
            torch.cuda.synchronize()
            pe_time = time.time() - pe_start

            if local_rank is not None:
                data_batch = [x.cuda(local_rank, non_blocking=True) for x in data_batch]
            else:
                data_batch = [x.cuda(non_blocking=True) for x in data_batch]

            # ==== AMP mixed precision training ====
            fwd_start = time.time()
            with autocast():
                output_list = model(embed_input)

                # ✅ 可选：强制输出尺寸（训练阶段）
                if args.out_h > 0 and args.out_w > 0:
                    output_list = [
                        F.interpolate(x, size=(args.out_h, args.out_w),
                                      mode='bilinear', align_corners=False)
                        for x in output_list
                    ]

                patch_output_list = [[] for _ in range(len(data_batch))]
                for stage_output in output_list:
                    for b in range(len(data_batch)):
                        patch_output_list[b].append(stage_output[b])

                loss_sum = 0
                for patch_outputs, data in zip(patch_output_list, data_batch):
                    target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in patch_outputs]
                    loss_list = [loss_fn(x.unsqueeze(0), y, args) for x, y in zip(patch_outputs, target_list)]
                    loss_list = [l * (args.lw if i < len(loss_list) - 1 else 1) for i, l in enumerate(loss_list)]
                    loss_sum += sum(loss_list)
            torch.cuda.synchronize()
            fwd_time = time.time() - fwd_start

            backward_start = time.time()
            lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)
            optimizer.zero_grad()
            scaler.scale(loss_sum).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            backward_time = time.time() - backward_start

            batch_total = time.time() - batch_start
            print(
                f"[TIMING] PE: {pe_time:.3f}s | Forward+Loss: {fwd_time:.3f}s | Backward: {backward_time:.3f}s | Total: {batch_total:.3f}s")

            for patch_outputs, data in zip(patch_output_list, data_batch):
                target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in patch_outputs]
                patch_outputs = [x.unsqueeze(0) for x in patch_outputs]
                psnr_list.append(psnr_fn(patch_outputs, target_list))
                msssim_list.append(msssim_fn(patch_outputs, target_list))

            if i % args.print_freq == 0 or i + batch_size >= num_patches:
                train_psnr = torch.cat(psnr_list, dim=0).mean(dim=0)
                train_msssim = torch.cat(msssim_list, dim=0).mean(dim=0)
                time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}'.format(
                    time_now_string, local_rank, epoch + 1, args.epochs, i + 1, len(train_dataloader), lr,
                    RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')

        if args.distributed and args.ngpus_per_node > 1:
            train_psnr = all_reduce([train_psnr.to(local_rank)])
            train_msssim = all_reduce([train_msssim.to(local_rank)])

        if local_rank in [0, None]:
            h, w = patch_output_list[0][-1].shape[-2:]
            train_best_psnr = max(train_best_psnr, train_psnr[-1])
            train_best_msssim = max(train_best_msssim, train_msssim[-1])
            writer.add_scalar(f'Train/PSNR_{h}X{w}_gap{args.frame_gap}', train_psnr[-1].item(), epoch + 1)
            writer.add_scalar(f'Train/MSSSIM_{h}X{w}_gap{args.frame_gap}', train_msssim[-1].item(), epoch + 1)
            writer.add_scalar(f'Train/best_PSNR_{h}X{w}_gap{args.frame_gap}', train_best_psnr.item(), epoch + 1)
            writer.add_scalar(f'Train/best_MSSSIM_{h}X{w}_gap{args.frame_gap}', train_best_msssim, epoch + 1)
            print_str = '\t{}p: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(
                h, train_psnr[-1].item(),
                train_best_psnr.item(),
                train_best_msssim.item())
            print(print_str, flush=True)
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
            writer.add_scalar('Train/lr', lr, epoch + 1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format(
                (epoch_end_time - epoch_start_time).total_seconds(),
                (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch)))

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'train_best_psnr': train_best_psnr,
            'train_best_msssim': train_best_msssim,
            'val_best_psnr': val_best_psnr,
            'val_best_msssim': val_best_msssim,
            'optimizer': optimizer.state_dict(),
        }

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
                    h, val_psnr[-1].item(),
                    val_best_psnr.item(),
                    val_best_msssim.item(),
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
    # Model Quantization
    if args.quant_bit != -1:
        cur_ckt = model.state_dict()
        from dahuffman import HuffmanCodec
        quant_weitht_list = []
        for k, v in cur_ckt.items():
            large_tf = (v.dim() in {2, 4} and 'bias' not in k)
            quant_v, new_v = quantize_per_tensor(v, args.quant_bit, args.quant_axis if large_tf else -1)
            valid_quant_v = quant_v[v != 0]  # only include non-zero weights
            quant_weitht_list.append(valid_quant_v.flatten())
            cur_ckt[k] = new_v
        cat_param = torch.cat(quant_weitht_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
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

        # compute psnr and msssim
        fwd_num = 10 if args.eval_fps else 1
        for _ in range(fwd_num):
            start_time = datetime.now()
            output_list = model(embed_input)

            # ✅ 可选：强制输出尺寸（验证阶段）
            if args.out_h > 0 and args.out_w > 0:
                output_list = [
                    F.interpolate(x, size=(args.out_h, args.out_w),
                                  mode='bilinear', align_corners=False)
                    for x in output_list
                ]

            torch.cuda.synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

        # dump predictions
        if args.dump_images:
            # ✅ GT 也 resize 到同一尺寸，方便对比查看
            if args.out_h > 0 and args.out_w > 0:
                data_to_save = F.interpolate(
                    data, size=(args.out_h, args.out_w),
                    mode='bilinear', align_corners=False
                )
            else:
                data_to_save = data

            for batch_ind in range(args.batchSize):
                full_ind = i * args.batchSize + batch_ind
                save_image(output_list[-1][batch_ind], f'{visual_dir}/pred_{full_ind}.png')
                save_image(data_to_save[batch_ind], f'{visual_dir}/gt_{full_ind}.png')

        # compute psnr and ms-ssim
        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        psnr_list.append(psnr_fn(output_list, target_list))
        msssim_list.append(msssim_fn(output_list, target_list))
        val_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
        val_psnr = torch.mean(val_psnr, dim=0)  # (num_stage)
        val_msssim = torch.cat(msssim_list, dim=0)  # (batchsize, num_stage)
        val_msssim = torch.mean(val_msssim.float(), dim=0)  # (num_stage)
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
