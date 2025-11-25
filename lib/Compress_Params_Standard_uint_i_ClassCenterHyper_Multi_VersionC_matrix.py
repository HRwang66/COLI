import numpy as np
import time
import os
from tqdm import tqdm
import shutil
import multiprocessing
from .param_compress_new2_uint_i_ClassCenterHyper_Multi_VersionC_matrix import decompress_v3, compress_decom_v3, Save_Num_Zero, Decode_Params
import torch
import torch.nn as nn
import pickle
import os
import threading
import subprocess

def dummy_task():
    while True:
        pass

def save_uintn_list_to_bin(int_list, n, file_path):
    # 确定每个整数需要多少字节
    num_bytes = (n + 7) // 8  # 向上取整到完整的字节数

    with open(file_path, 'wb') as f:
        for number in int_list:
            # 将每个数字限制在 n 位以内
            masked_number = int(number) & ((1 << n) - 1)
            # 将数字转换为字节并写入文件
            for i in range(num_bytes):
                byte = (masked_number >> (8 * (num_bytes - 1 - i))) & 0xFF
                f.write(byte.to_bytes(1, byteorder='big'))

def trans2byte(bitstring):
    output = int(bitstring, 2)
    return output




def save_uintn_list_to_bits(Result, uint_i, save_path):
    bitstring = ''.join(format(number, f'0{uint_i}b') for number in Result)
    if len(bitstring) % 8 == 0:
        n_zero = 0
    else:
        n_zero = (8 - len(bitstring) % 8) # 前面
        bitstring = '0'*n_zero + bitstring # 前面补n_zero个0

    '''''
    num_core = os.cpu_count()
    with multiprocessing.Pool(processes=1) as pool:
        # 构造参数列表
        trans_uint_list = pool.starmap(trans2byte, [[bitstring[i:i+8]] for i in range(0, len(bitstring), 8)])
    '''''

    trans_uint_list = []
    for i in [bitstring[i:i+8] for i in range(0, len(bitstring), 8)]:
        trans_uint_list.append(trans2byte(i))

    np.array(trans_uint_list, dtype=np.uint8).tofile(save_path+f'_{n_zero}.bin')




def get_max_threads():
    # 获取CPU核心数
    cpu_cores = os.cpu_count()

    # 获取用户进程限制（适用于Linux）
    try:
        max_processes = int(subprocess.check_output(['ulimit', '-u']).strip())
    except Exception as e:
        print(f"Error fetching max processes: {e}")
        max_processes = cpu_cores * 10  # 默认值

    # 根据系统资源设置线程数
    max_threads = min(cpu_cores * 2, max_processes)

    return max_threads



def save_new_param_uncompress(root_dir, ori_param, num_ori):
    """
    root_dir : “ .../Compressed_Dir/ ”
    param : param
    num_ori : i-layer
    """

    name_size = ''
    for i in ori_param.shape:
        name_size += '_' + str(i)

    save_name = f"{str(num_ori)}{name_size}_0_0.bin"
    save_path = root_dir + save_name
    ori_param_flatten = ori_param.flatten()
    ori_param_flatten.astype(np.float32).tofile(save_path)  # save as float32 array



def save_new_param_compress(root_dir, output_idx, num_ori, best_class, num_inner, ori_shape):
    """
        root_dir : “ .../Compressed_Dir/ ”
        output_idx : 压缩后的index
        num_ori : i-layer
        best_class : param的outer的class
        num_inner : inner中的nodes数量 (不包含center node)
        ori_shape : ori_param的shape
        """

    name_size = ''
    for i in ori_shape:
        name_size += '_' + str(i)


    if int(np.max(output_idx))==0:
        uint_i = 1
    else:
        uint_i = int(np.max(output_idx)).bit_length()

    save_name = f"{str(num_ori)}{name_size}_{uint_i}"
    save_path = root_dir + save_name

    output_idx = output_idx.astype(np.int64)

    save_uintn_list_to_bits(output_idx, uint_i, save_path)






#### Only Multiprocessing ########
def multi_compress_decom_v3(param, num_ori, rect_l, num_inner_list, root_dir, class_max, loss_max, loss_hope):

    print("\n")
    print(f"num_ori : {num_ori}")

    try:
        if param.flatten().shape[0] >= 10:
            """
            results[0] : param还原后的MAE_loss list
            results[1] ： outer的最佳分类数
            results[2] ：替换后的新参数
            results[3] : 压缩后的index
            results[4] : inner中的nodes数量 (不包含center node)
            results[5] : inner nodes的数量
            results[6] : outer nodes的数量
            results[7] : padding_size, 补0数量
            results[8] : num_inner在num_inner_list中的index
            results[9] : center_node
            results[10] : farthest_node
            """

            results = compress_decom_v3(param, num_ori, rect_l, num_inner_list, class_max, loss_max, loss_hope)
            new_param = results[2]
            mean_MAE = np.mean(results[0])
            if mean_MAE > 0.008: # 0.006
                print(1/0)
            max_loss = np.max(results[0])
            min_loss = np.min(results[0])
            max_index = (results[4]+1) + results[1] * (results[4]+1)
            if_padding = results[7]
            num_inner_index = results[8]
            best_class = results[1]
            center_node = results[9]
            farthest_node = results[10]
            print(f"total_mean_loss : {mean_MAE}")
            print(f"best_class : {results[1]}")
            print(f"max_index : {max_index}")
            print(f"real_max_index : {np.max(results[3])}")
            print(f"num_inside : {results[5]}")
            print(f"num_outside : {results[6]}")
            save_new_param_compress(root_dir, results[3], num_ori, results[1], results[4], param.shape)
        else:
            print(1/0)

    except:
        print(f"There is a BUG, the num_ori is {num_ori}")
        new_param = param
        mean_MAE = 0
        max_loss = 0
        min_loss = 0
        max_index = 0
        save_new_param_uncompress(root_dir, param, num_ori)
        if_padding = 0
        num_inner_index = 0
        best_class = 0
        center_node = np.array([0,0], dtype=np.float32)
        farthest_node = np.array([0,0], dtype=np.float32)

    return new_param, mean_MAE, max_index, if_padding, num_inner_index, best_class, center_node, farthest_node



def compress_params(model, Save_CompressedResult_RootPath, rect_l, num_inner_list, class_max, loss_max, loss_hope, num_cores):  #

    ######### 压缩并保存结果Result #################  ./compressed_result/  MobileNetV3_PruneRatio_0.5_PrunerName_L1NormPruner_codebook_6270/
    #############################################
    #global dimension, Max
    #Max, dimension = define_global(codebook_num, uint_i, K, K_lossless)

    root_dir = Save_CompressedResult_RootPath


    # 如果root_dir ( ./test/Compressed_Dir ) 存在，则删除它重建
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
        os.makedirs(root_dir, exist_ok=True)
    else:
        os.makedirs(root_dir, exist_ok=True)

    np.array([num_inner_list]).astype(np.uint64).tofile(root_dir+'num_inner_list.bin')

    params_list = list(model.parameters())

    # num_cores = os.cpu_count()
    #print(f"Available CPU cores: {num_cores}")


    #################################
    ###### Try multiprocessing ######
    #test_smallest_rect_l = test_rect_l(params_list)  # 测试能使所有layer的inner中没有nodes的rect_l


    t_start = time.perf_counter()
    with multiprocessing.Pool(processes=num_cores) as pool:
        new_params_list_multi = pool.starmap(multi_compress_decom_v3,
                                        [[params_list[num_ori].data.cpu().numpy(), num_ori, rect_l, num_inner_list,
                                          root_dir, class_max, loss_max, loss_hope] for num_ori in range(len(params_list))])



    new_params_list = [t[0] for t in new_params_list_multi]
    MAE = np.array([t[1] for t in new_params_list_multi])
    total_mean_MAE = np.mean(MAE)
    total_max_MAE = np.max(MAE)
    total_min_MAE = np.min(MAE[MAE != 0])
    total_max_index = np.mean(np.array([t[2] for t in new_params_list_multi]))


    if_padding = np.array([t[3] for t in new_params_list_multi])
    if_padding.astype(np.uint8).tofile(root_dir+'if_padding.bin')

    num_inner_index = np.array([t[4] for t in new_params_list_multi])
    num_inner_index.astype(np.uint8).tofile(root_dir+'num_inner_index.bin')

    class_list = np.array([t[5] for t in new_params_list_multi])
    class_list.astype(np.uint8).tofile(root_dir+'class_list.bin')

    center_node_list = np.array([t[6] for t in new_params_list_multi])
    center_node_list.tofile(root_dir+'center_node_list.bin')

    farthest_node_list = np.array([t[7] for t in new_params_list_multi])
    farthest_node_list.tofile(root_dir + 'farthest_node_list.bin')


    t_end = time.perf_counter()
    diff_seconds = t_end - t_start
    hours = int(diff_seconds // 3600)
    minutes = int((diff_seconds % 3600) // 60)
    print("\n")
    print("Time：{}Hour {}Min".format(hours, minutes))
    print(f"rect_l : {rect_l}")
    print(f"max_index : {total_max_index}")
    print(f"MAE_loss : {total_mean_MAE}")
    print(f"max_loss : {total_max_MAE}")
    print(f"min_loss/0 : {total_min_MAE}")

    print("\n")
    print(f"替换前：{list(model.parameters())[0][0][0].detach().cpu().numpy()}")

    #### 将model params 换掉 #####
    with torch.no_grad():
        for i in tqdm(range(len(params_list))):
            ori_param = params_list[i].data  # 原参数
            new_param = new_params_list[i]  # 还原的新参数
            params_list[i].copy_(torch.tensor(new_param).float().cuda())  # 用还原参数替换原模型参数
    print(f"替换后：{list(model.parameters())[0][0][0].detach().cpu().numpy()}")


    ###### Try multiprocessing ######
    #################################

    compressed_size = get_folder_size(Save_CompressedResult_RootPath)

    return compressed_size, model



def read_uintn_list_from_bin_raw(layer_path, uint_i, uint_i_padding):
    """
    layer_path : array文件地址
    uint_i : 数据格式
    uint_i_padding : 前面补0个数
    """
    layer_param_int = []
    bit_save = np.fromfile(layer_path, dtype=np.uint8)
    # print(bit_save)
    t_start = time.perf_counter()
    # 转换成二进制，如不足8位补0至8位
    results8bits = "".join(format(i, f'0{8}b') for i in bit_save)[uint_i_padding:]


    # with open(layer_path, 'rb') as f:
    #     bit_save = f.read()
    #
    # # 将字节数据转换成二进制字符串
    # results8bits = ''.join(format(byte, '08b') for byte in bit_save)

    t_end = time.perf_counter()
    # print(f"part1:{t_end - t_start}秒")

    t_start = time.perf_counter()
    for i in range(0, len(results8bits), uint_i):
        index_bits = results8bits[i:i + uint_i]
        layer_param_int.append(trans2byte(index_bits))
    t_end = time.perf_counter()
    # print(f"part2:{t_end - t_start}秒")

    return np.array(layer_param_int)


def read_uintn_list_from_bin(layer_path, uint_i, uint_i_padding):
    with open(layer_path, 'rb') as f:
        byte_data = np.fromfile(f, dtype=np.uint8)  # 直接读取字节数据
    # 将每个字节转换为二进制字符串，并去掉前面的 '0b' 标记
    binary_data = np.unpackbits(byte_data)  # 将字节转换为二进制位数组
    # 去掉前面指定数量的比特位
    binary_data = binary_data[uint_i_padding:]
    # 按 bit_length 将二进制位数组分组
    binary_data = binary_data.reshape(-1, uint_i)
    # 生成位权重，形如 [2^(bit_length-1), ..., 2^0]
    weights = 2 ** np.arange(uint_i)[::-1]
    # 使用矩阵乘法来计算每个二进制片段的十进制值
    restored_numbers = binary_data.dot(weights)

    return restored_numbers


def read_uintn_list_from_bin00(layer_path, uint_i, uint_i_padding):
    with open(layer_path, 'rb') as f:
        byte_data = np.fromfile(f, dtype=np.uint8)  # 高效读取字节数据

    # 直接进行切片，减少冗余操作
    total_bits = byte_data.size * 8
    useful_bits = total_bits - uint_i_padding
    binary_data = np.unpackbits(byte_data)[:useful_bits]  # 转换并只保留有效位

    # 避免不必要的 reshape 操作
    n_elements = useful_bits // uint_i
    binary_data = binary_data[:n_elements * uint_i].reshape(-1, uint_i)

    # 使用 bit shift 和累加而不是矩阵乘法
    weights = 2 ** np.arange(uint_i)[::-1]
    restored_numbers = np.dot(binary_data, weights)  # 使用np.dot进行矢量化计算

    return restored_numbers


def find_nth_occurrence(string, substring, n):
    start = 0
    for i in range(n):
        start = string.index(substring, start) + 1
    return start - 1

import torch

def decompress_params0(model, Decode_Param_Path, rect_l, num_inner_list, num_streams=16):
    if_padding = np.fromfile(Decode_Param_Path + 'if_padding.bin', dtype=np.uint8)
    num_inner_index = np.fromfile(Decode_Param_Path + 'num_inner_index.bin', dtype=np.uint8)
    class_list = np.fromfile(Decode_Param_Path + 'class_list.bin', dtype=np.uint8)
    center_node_list = np.fromfile(Decode_Param_Path + 'center_node_list.bin', dtype=np.float32)
    center_node_list = center_node_list.reshape(-1,2)
    farthest_node_list = np.fromfile(Decode_Param_Path + 'farthest_node_list.bin', dtype=np.float32)
    farthest_node_list = farthest_node_list.reshape(-1,2)

    file_names = os.listdir(Decode_Param_Path)

    dict_layer_files = {}
    for file_name in file_names:
        try:
            i_layer = int(file_name[:file_name.index('_')])
            split_str = file_name[file_name.index('_') + 1: -4].split('_')
            shape = []
            for j in split_str[:-2]:
                shape.append(int(j))
            shape = np.array(shape)
            uint_i = int(split_str[-2])
            uint_i_padding = int(split_str[-1])
            dict_layer_files[i_layer] = [Decode_Param_Path + file_name, shape, uint_i, uint_i_padding]
        except:
            continue

    num_layers = len(list(dict_layer_files.values()))
    print(f"成功获取{num_layers}个文件")

    decode_params_list = [None] * num_layers
    streams = [torch.cuda.Stream() for _ in range(num_streams)]  # 创建num_streams个流

    # 分批次执行每层解压
    for i in tqdm(range(num_layers)):
        stream_index = i % num_streams  # 将层分配给不同的流
        layer_path, layer_shape, uint_i, uint_i_padding = dict_layer_files[i]
        with torch.cuda.stream(streams[stream_index]):  # 指定在该流中执行
            if uint_i == 0:  # 当时压缩失败的参数直接load
                try:
                    param1 = np.fromfile(layer_path, dtype=np.float32)
                    param1 = np.array(param1).astype(np.float32).reshape(layer_shape)
                    decode_params_list[i] = param1
                except:
                    print("wait")
            else:
                try:
                    if_padding_i = if_padding[i]
                    num_inner_index_i = num_inner_index[i]
                    class_list_i = class_list[i]
                    center_node_i = center_node_list[i]
                    farthest_node_i = farthest_node_list[i]



                    index_array = read_uintn_list_from_bin(layer_path, uint_i, uint_i_padding)

                    param1 = decompress_v3(index_array, num_inner_index_i, class_list_i, if_padding_i, center_node_i, farthest_node_i,
                                           rect_l, num_inner_list)

                    param1 = np.array(param1).astype(np.float32).reshape(layer_shape)
                    decode_params_list[i] = param1
                except:
                    print("wait")

    # 同步所有流
    torch.cuda.synchronize()

    return decode_params_list

# 多进程
def process_single_layer(i, dict_layer_files, if_padding, num_inner_index, class_list, center_node_list,
                         farthest_node_list, rect_l, num_inner_list):
    layer_path, layer_shape, uint_i, uint_i_padding = dict_layer_files[i]

    # 如果压缩失败直接读取float参数
    if uint_i == 0:
        try:
            param1 = np.fromfile(layer_path, dtype=np.float32)
            param1 = np.array(param1).astype(np.float32).reshape(layer_shape)
            return param1
        except:
            print("wait")
            return None

    else:
        try:
            if_padding_i = if_padding[i]
            num_inner_index_i = num_inner_index[i]
            class_list_i = class_list[i]
            center_node_i = center_node_list[i]
            farthest_node_i = farthest_node_list[i]

            # 读取并解压参数
            index_array = read_uintn_list_from_bin(layer_path, uint_i, uint_i_padding)
            param1 = decompress_v3(index_array, num_inner_index_i, class_list_i, if_padding_i, center_node_i,
                                   farthest_node_i, rect_l, num_inner_list)
            param1 = np.array(param1).astype(np.float32).reshape(layer_shape)
            return param1

        except:
            print("wait")
            return None

from concurrent.futures import ProcessPoolExecutor,as_completed


# 多进程
def decompress_params999(model, Decode_Param_Path, rect_l, num_inner_list):
    # 加载必要的数据
    if_padding = np.fromfile(Decode_Param_Path + 'if_padding.bin', dtype=np.uint8)
    num_inner_index = np.fromfile(Decode_Param_Path + 'num_inner_index.bin', dtype=np.uint8)
    class_list = np.fromfile(Decode_Param_Path + 'class_list.bin', dtype=np.uint8)
    center_node_list = np.fromfile(Decode_Param_Path + 'center_node_list.bin', dtype=np.float32).reshape(-1, 2)
    farthest_node_list = np.fromfile(Decode_Param_Path + 'farthest_node_list.bin', dtype=np.float32).reshape(-1, 2)

    # 构建 layer_i 与 文件地址的dict
    file_names = os.listdir(Decode_Param_Path)
    dict_layer_files = {}
    for file_name in file_names:
        try:
            i_layer = int(file_name[:file_name.index('_')])
            split_str = file_name[file_name.index('_') + 1: -4].split('_')
            shape = [int(j) for j in split_str[:-2]]
            shape = np.array(shape)
            uint_i = int(split_str[-2])
            uint_i_padding = int(split_str[-1])
            dict_layer_files[i_layer] = [Decode_Param_Path + file_name, shape, uint_i, uint_i_padding]
        except:
            continue

    num_layers = len(dict_layer_files)
    print(f"成功获取 {num_layers} 个文件")

    # decode_params_list = []

    # 使用 ProcessPoolExecutor 并行处理
    with ProcessPoolExecutor(max_workers=4) as executor:  # 根据硬件调整 max_workers
        # futures = [executor.submit(process_single_layer, i, dict_layer_files, if_padding, num_inner_index, class_list,
        #                            center_node_list, farthest_node_list, rect_l, num_inner_list) for i in
        #            range(num_layers)]

        # 创建一个字典，键是 Future 对象，值是原始索引
        futures = {
            executor.submit(process_single_layer, i, dict_layer_files, if_padding, num_inner_index, class_list,
                            center_node_list, farthest_node_list, rect_l, num_inner_list): i for i in
            range(num_layers)}

        # 初始化一个空的列表，按顺序存储解码后的参数
        decode_params_list = [None] * num_layers

        # 使用 tqdm 和 as_completed 迭代
        for future in tqdm(as_completed(futures), total=num_layers):
            index = futures[future]  # 获取原始索引
            param1 = future.result()
            if param1 is not None:
                decode_params_list[index] = param1  # 按顺序存储结果

    return decode_params_list


# 多线程
def process_single_layer0(i, dict_layer_files, if_padding, num_inner_index, class_list, center_node_list,
                         farthest_node_list, rect_l, num_inner_list):
    layer_path, layer_shape, uint_i, uint_i_padding = dict_layer_files[i]

    # 如果压缩失败直接读取float参数
    if uint_i == 0:
        try:
            param1 = np.fromfile(layer_path, dtype=np.float32)
            param1 = np.array(param1).astype(np.float32).reshape(layer_shape)
            return param1
        except:
            print("wait")
            return None

    else:
        try:
            if_padding_i = if_padding[i]
            num_inner_index_i = num_inner_index[i]
            class_list_i = class_list[i]
            center_node_i = center_node_list[i]
            farthest_node_i = farthest_node_list[i]

            # 读取并解压参数
            index_array = read_uintn_list_from_bin(layer_path, uint_i, uint_i_padding)
            param1 = decompress_v3(index_array, num_inner_index_i, class_list_i, if_padding_i, center_node_i,
                                   farthest_node_i, rect_l, num_inner_list)
            param1 = np.array(param1).astype(np.float32).reshape(layer_shape)
            return param1

        except:
            print("wait")
            return None


# 多线程
def decompress_params0(model, Decode_Param_Path, rect_l, num_inner_list):
    # 加载必要的数据
    if_padding = np.fromfile(Decode_Param_Path + 'if_padding.bin', dtype=np.uint8)
    num_inner_index = np.fromfile(Decode_Param_Path + 'num_inner_index.bin', dtype=np.uint8)
    class_list = np.fromfile(Decode_Param_Path + 'class_list.bin', dtype=np.uint8)
    center_node_list = np.fromfile(Decode_Param_Path + 'center_node_list.bin', dtype=np.float32).reshape(-1, 2)
    farthest_node_list = np.fromfile(Decode_Param_Path + 'farthest_node_list.bin', dtype=np.float32).reshape(-1, 2)

    # 构建 layer_i 与 文件地址的dict
    file_names = os.listdir(Decode_Param_Path)
    dict_layer_files = {}
    for file_name in file_names:
        try:
            i_layer = int(file_name[:file_name.index('_')])
            split_str = file_name[file_name.index('_') + 1: -4].split('_')
            shape = [int(j) for j in split_str[:-2]]
            shape = np.array(shape)
            uint_i = int(split_str[-2])
            uint_i_padding = int(split_str[-1])
            dict_layer_files[i_layer] = [Decode_Param_Path + file_name, shape, uint_i, uint_i_padding]
        except:
            continue

    num_layers = len(dict_layer_files)
    print(f"成功获取 {num_layers} 个文件")

    decode_params_list = []

    # 使用 ThreadPoolExecutor 并行处理
    with ThreadPoolExecutor(max_workers=1) as executor:  # 根据硬件调整 max_workers
        futures = [executor.submit(process_single_layer, i, dict_layer_files, if_padding, num_inner_index, class_list,
                                   center_node_list, farthest_node_list, rect_l, num_inner_list) for i in
                   range(num_layers)]

        for future in tqdm(as_completed(futures), total=num_layers):
            param1 = future.result()
            if param1 is not None:
                decode_params_list.append(param1)

    return decode_params_list



# 原始
def decompress_params(model, Decode_Param_Path, rect_l, num_inner_list):
    t0 = time.perf_counter()
    readtime=0
    compresstime=0
    time1d=0
    if_padding = np.fromfile(Decode_Param_Path + 'if_padding.bin', dtype=np.uint8)
    num_inner_index = np.fromfile(Decode_Param_Path + 'num_inner_index.bin', dtype=np.uint8)
    class_list = np.fromfile(Decode_Param_Path + 'class_list.bin', dtype=np.uint8)
    center_node_list = np.fromfile(Decode_Param_Path + 'center_node_list.bin', dtype=np.float32)
    center_node_list = center_node_list.reshape(-1,2)
    farthest_node_list = np.fromfile(Decode_Param_Path + 'farthest_node_list.bin', dtype=np.float32)
    farthest_node_list = farthest_node_list.reshape(-1,2)

    # 获取 root文件夹中所有文件名中不带"_"的文件名
    file_names = os.listdir(Decode_Param_Path)

    # 构建 layer_i 与 文件地址的dict
    dict_layer_files = {}
    for file_name in file_names:
        try:
            i_layer = int(file_name[:file_name.index('_')])  # 第几层的参数
            split_str = file_name[file_name.index('_') + 1: -4].split('_')
            shape = []
            for j in split_str[:-2]:
                shape.append(int(j))
            shape = np.array(shape)  # 参数的shape
            uint_i = int(split_str[-2])
            uint_i_padding = int(split_str[-1])
            dict_layer_files[i_layer] = [Decode_Param_Path + file_name, shape, uint_i, uint_i_padding]
        except:
            # except 说明是“非layer params”文件
            #print("构建 layer_i 与 文件地址的dict BUG!!!!!!!!!")
            continue

    num_layers = len(list(dict_layer_files.values()))
    print(f"成功获取{num_layers}个文件")

    decode_params_list = []
    # 遍历到每一个文件名和shape
    for i in tqdm(range(num_layers)):

        layer_path, layer_shape, uint_i, uint_i_padding = dict_layer_files[i]

        if uint_i == 0:  # 当时压缩失败的参数直接load
            try:
                param1 = np.fromfile(layer_path, dtype=np.float32)
                param1 = np.array(param1).astype(np.float32).reshape(layer_shape)
                decode_params_list.append(param1)
            except:
                print("wait")

        else:
            try:
                if_padding_i = if_padding[i]
                num_inner_index_i = num_inner_index[i]
                class_list_i = class_list[i]
                center_node_i = center_node_list[i]
                farthest_node_i = farthest_node_list[i]

                t_start = time.perf_counter()
                # read_uintn_list_from_bin适合在CPU执行
                index_array = read_uintn_list_from_bin(layer_path, uint_i, uint_i_padding)
                t_end = time.perf_counter()
                # print(f"read时间：{t_end - t_start}")
                readtime+=t_end - t_start
                t_start = time.perf_counter()

                param1,time1 = decompress_v3(index_array, num_inner_index_i, class_list_i, if_padding_i, center_node_i, farthest_node_i,
                                       rect_l, num_inner_list)
                time1d+=time1
                t_end = time.perf_counter()
                # print(f"decompress时间：{t_end - t_start}")
                compresstime += t_end - t_start
                # param1 : 将整数index结果decode为新的float参数
                param1 = np.array(param1).astype(np.float32).reshape(layer_shape)

                decode_params_list.append(param1)
            except:
                print("wait")
    t1 = time.perf_counter()
    print(f"all：{t1 - t0}")
    print(f"readtime：{readtime}")
    print(f"compresstime：{compresstime}")
    print(f"1d：{time1d}")
    return decode_params_list


def param_split(param):
    param_flatten = param.flatten()  #将param展开成1维

    param_decimal = np.array([])
    param_int = np.array([])

    '''''
    t1_start = time.perf_counter()
    index = 0
    for i in param_flatten:
        if i != 0:
            param_decimal = np.append(param_decimal , i)
            param_int = np.append(param_int, index)
        index += 1
    #print(param_int[:5])
    t1_end = time.perf_counter()
    t1 = t1_end - t1_start
    '''''

    t2_start = time.perf_counter()
    # 创建进程池，最大进程数为 8
    with multiprocessing.Pool(processes=8) as pool:
        # 构造参数列表
        args_list = [(index, value) for index, value in enumerate(param_flatten)]

        # 并行处理非零元素的坐标
        param_int = pool.map(find_nonzero_indices, args_list)
        # 过滤掉为 None 的元素
        param_int = [index for index in param_int if index is not None]
        #print("非零元素的坐标：", len(param_int))

        param_decimal = pool.map(find_nonzero_value, args_list)
        param_decimal = [value for value in param_decimal if value is not None]
        #print("非零元素的value：",len(param_decimal))

    t2_end = time.perf_counter()
    t2 = t2_end - t2_start

    #print(f"调整初始点  速度提升{round((t1 - t2) / t1, 4) * 100}%")

    '''''
    param_dis = np.array([param_int[0]])
    for ele in range(1, len(param_int)):
        param_dis = np.append(param_dis, param_int[ele] - param_int[ele-1])
    #print(param_dis[:5])
    '''''

    with multiprocessing.Pool(processes=8) as pool:
        param_dis = pool.starmap(get_param_dis, [[param_int[ele-1], param_int[ele]] for ele in range(1, len(param_int))])
    param_dis = np.append(param_int[0], param_dis)

    return param_decimal, param_dis


def find_nonzero_value(args):
    index, value = args
    if value != 0:
        return value
    else:
        return None



def get_param_dis(param_int_l, param_int_r):
    return param_int_r - param_int_l



def find_nonzero_indices(args):
    index, value = args
    if value != 0:
        return index
    else:
        return None



def flatten_restore(new_param_flatten, param_dis, param_layer_shape):
    param_index = [param_dis[0]]
    for i in range(1, len(param_dis)):
        param_index.append(param_index[i-1]+param_dis[i])

    new_param_back_flatten = np.zeros(param_layer_shape).flatten()

    for index in range(len(param_index)): # param_index : new_param_flatten对应的index
        new_param_back_flatten[int(param_index[index])] = new_param_flatten[index]

    new_param = new_param_back_flatten.reshape(param_layer_shape)

    return new_param


def check_add(S_split_each, add_element): #判断 add_element是否还能加入S_split_each

    S_split_each = S_split_each + [add_element]

    # 判断目前的S_split_each中的最大值所对应的最大倍数 max_dimension
    max_dimension = Max_Dimension(S_split_each)

    if len(S_split_each) <= max_dimension:
        return True
    else:
        return False



def dynamic_split(param_dis):
    S_split = []
    K = []
    while len(param_dis) != 0:
        S_split_each = [param_dis[0]]
        index = 1
        while check_add(S_split_each, param_dis[index]) and index != len(param_dis)-1:
            S_split_each.append(param_dis[index])
            index += 1

        if index != len(param_dis)-1:
            S_split.append(S_split_each)
            K.append(len(S_split_each))
            param_dis = param_dis[index:]

        elif index == len(param_dis)-1:
            S_split.append(param_dis.tolist())
            K.append(len(param_dis))
            param_dis = []
            #到最后补0


    return S_split, K

def Max_Dimension(list):
    # 判断目前的S_split_each中的最大值所对应的最大倍数 max_dimension
    if max(list) == 0 and len(list) <= 64:
        max_dimension = 64
    else:
        for i in range(len(Max)):
            if Max[i] < max(list):
                max_dimension = dimension[i - 1]
                break
    return max_dimension

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Skip if it is a symbolic link
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size