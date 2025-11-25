# Use a pipeline as a high-level helper
#from transformers import pipeline
import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from lib.Compress_Params_Standard_uint_i_ClassCenterHyper_Multi_VersionC_matrix import compress_params, decompress_params, find_nth_occurrence
import contextlib
from tqdm import tqdm
from model_nerv import Generator
import shutil
import time
from optparse import OptionParser





if __name__ == '__main__':

    model_name = 'INR'

    rect_l = 0.2  # 0.1, 0.3
    num_inner_list = [2025]  # 设为平方数，数字越小，压缩倍数越高
    class_max = 3
    loss_max = 0.001
    loss_hope = 0.001
    num_cores = 1  # num_cpu

    mode = 'encode' # 'encode' or 'decode'

    model = Generator(stem_dim_num='512_1',            # stem层维度和数量
    fc_hw_dim='9_16_58',        # 全连接层的高度、宽度和维度
    embed_length=80,               # 嵌入向量的长度
    stride_list=[5,2,2, 2, 2],          # 卷积层的步幅
    expansion=4,                  # 卷积层宽度扩展系数
    reduction=2,                    # 卷积层宽度缩小系数
    lower_width=96,                 # 最小宽度
    num_blocks=1,                   # 每个阶段的模块数量
    bias=True,
    norm='none',                      # 使用批量归一化
    act='swish',                     # 使用ReLU激活函数
    conv_type='conv',               # 使用普通卷积                      # 使用偏置项
    sin_res=True,                  # 不使用sin作为激活函数
    sigmoid=False                    # 对输出使用Sigmoid激活函数
    )



    # encoding save path
    Save_Param_Path = f'./compressed_result/{model_name}_l_{str(rect_l)[0] + str(rect_l)[2:]}/'

    # decoding files path
    # Decode_Param_Path = f'./compressed_result/{model_name}_l_{str(rect_l)[0] + str(rect_l)[2:]}'
    Decode_Param_Path = Save_Param_Path

    if mode == 'encode':

        checkpoint = torch.load('')  #Add ur checkpoint here
        str01 = model.state_dict()
        print(f"str01 : {len(str01)}")
        str02 = checkpoint['state_dict']
        print(f"str02 : {len(str02)}")

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        # 若存在，则删除重建
        if os.path.exists(Save_Param_Path):
            shutil.rmtree(Save_Param_Path)
            os.makedirs(Save_Param_Path, exist_ok=True)
        else:
            os.makedirs(Save_Param_Path, exist_ok=True)


        # 原参数保存地址
        Save_OriParam_Path = Save_Param_Path + 'Origin_Params.pth'
        # SpeedUp + FineTuning 后的params压缩结果的文件夹root地址
        Save_CompressedResult_RootPath = Save_Param_Path + 'Compressed_Dir/'
        # 压缩后再还原的params保存地址
        Save_BackParam_Path = Save_Param_Path + 'Back_Params.pth'


        ## 保存 model的原参数 ##
        torch.save(model.state_dict(), Save_OriParam_Path)

        t1_start = time.perf_counter()
        ### model ：参数复原后的model ###
        size_result, model = compress_params(model, Save_CompressedResult_RootPath, rect_l, num_inner_list, class_max,
                                             loss_max, loss_hope, num_cores)  # 返回 压缩后的文件大小
        t1_end = time.perf_counter()

        # 保存"Back_Params.pth"和"pytorch_model.bin"
        torch.save(model.state_dict(), Save_BackParam_Path)


        print(f"原参数大小为 {os.path.getsize(Save_OriParam_Path)}字节")
        print(f"压缩结果的大小为 {size_result}字节")
        print(f"压缩倍数{os.path.getsize(Save_OriParam_Path) / size_result}倍")

        print("Encoding Finished!!!!!!!")
        print(f"Compression Time : {t1_end-t1_start}秒 = {(t1_end-t1_start)/60}分钟")


    elif mode == 'decode':
        t_start = time.perf_counter()
        num_inner_list = np.fromfile(Decode_Param_Path + '/Compressed_Dir/num_inner_list.bin', dtype=np.uint64)
        rect_l_str = Decode_Param_Path[find_nth_occurrence(Decode_Param_Path, "/", 2):].split('_')[2]
        rect_l = float(rect_l_str[:1] + '.' + rect_l_str[1:-1])

        # 通过文件夹decode出来的model的所有参数张量 [tensor1, tensor2, ..... , tensorN]
        decode_params_list = decompress_params(model, Decode_Param_Path + '/Compressed_Dir/', rect_l, num_inner_list)
        t_end = time.perf_counter()
        print(f"{t_end-t_start}秒")

        # print(f"Decode : {decode_params_list[0][0][0]}")
        # model.load_state_dict(torch.load(Decode_Param_Path + '/Back_Params.pth'))
        # params_list = list(model.parameters())
        # target_params_list = [i.detach().cpu().numpy() for i in params_list]
        # print(f"Target : {target_params_list[0][0][0]}")
        #
        # error = 0
        # for i in range(len(target_params_list)):
        #     if not np.array_equal(target_params_list[i], decode_params_list[i]):
        #         error += 1
        #
        # if error == 0:
        #     print("Decoding Finished!!!!!!!")
        # else:
        #     print(f"There {error} bugs!!!!")

        params_list = list(model.parameters())
        all = []
        print(f"替换前：{list(model.parameters())[0][0][0].detach().cpu().numpy()}")
        #### 将model params 换掉 #####
        with torch.no_grad():
            for i in tqdm(range(len(params_list))):
                ori_param = params_list[i].data # 原参数
                new_param = decode_params_list[i]  # 还原的新参数
                params_list[i].copy_(torch.tensor(new_param).float().cuda())  # 用还原参数替换原模型参数
        print(f"替换后：{list(model.parameters())[0][0][0].detach().cpu().numpy()}")
        # print(model) # 已经加载了decode还原后的带损失参数的model
        state_dict_back = model.state_dict()
        print(f"wait")
