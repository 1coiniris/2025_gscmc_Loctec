from tensorboard.compat.tensorflow_stub.dtypes import double

import function.data_load as data_fun
import torch
import ast
import numpy as np
from scipy.io import savemat, loadmat
import argparse
import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # train_filename = ['data/341c-f0d4-70be/train/341c-f0d4-70be_0.xlsx',
    #             'data/bc98-2983-907b/train/bc98-2983-907b_0.xlsx',
    #             'data/d0c1-bfe4-18f0/train/d0c1-bfe4-18f0_0.xlsx']
    # test_filename = ['data/341c-f0d4-70be/valid/341c-f0d4-70be_0.xlsx',
    #             'data/bc98-2983-907b/valid/bc98-2983-907b_0.xlsx',
    #             'data/d0c1-bfe4-18f0/valid/d0c1-bfe4-18f0_0.xlsx']
    # train_341c,train_bc98,train_d0c1 = data_fun.train_data_get(train_filename)
    # train_341c['csi_state'] = [int(a) for a in train_341c['csi_state']]
    # train_341c['mcs'] = [float(a) for a in train_341c['mcs']]
    # train_341c['dfx_time'] = [float(a) for a in train_341c['dfx_time']]
    # train_341c['csi_time'] = [float(a) for a in train_341c['csi_time']]
    # train_341c['beamforming_en'] = [int(a) for a in train_341c['beamforming_en']]
    # train_341c['noise_floor'] = [float(a) for a in train_341c['noise_floor']]
    #
    # train_bc98['csi_state'] = [int(a) for a in train_bc98['csi_state']]
    # train_bc98['mcs'] = [float(a) for a in train_bc98['mcs']]
    # train_bc98['dfx_time'] = [float(a) for a in train_bc98['dfx_time']]
    # train_bc98['csi_time'] = [float(a) for a in train_bc98['csi_time']]
    # train_bc98['beamforming_en'] = [int(a) for a in train_bc98['beamforming_en']]
    # train_bc98['noise_floor'] = [float(a) for a in train_bc98['noise_floor']]
    #
    # train_d0c1['csi_state'] = [int(a) for a in train_d0c1['csi_state']]
    # train_d0c1['mcs'] = [float(a) for a in train_d0c1['mcs']]
    # train_d0c1['dfx_time'] = [float(a) for a in train_d0c1['dfx_time']]
    # train_d0c1['csi_time'] = [float(a) for a in train_d0c1['csi_time']]
    # train_d0c1['beamforming_en'] = [int(a) for a in train_d0c1['beamforming_en']]
    # train_d0c1['noise_floor'] = [float(a) for a in train_d0c1['noise_floor']]
    #
    # mat_filename = [f"data/341c_train.mat",
    #                 f"data/bc98_train.mat",
    #                 f"data/d0c1_train.mat"]
    # savemat(mat_filename[0], train_341c)
    # savemat(mat_filename[1], train_bc98)
    # savemat(mat_filename[2], train_d0c1)
    #
    # mat_filename = [f"data/341c_val.mat",
    #                 f"data/bc98_val.mat",
    #                 f"data/d0c1_val.mat"]
    # val_341c, val_bc98, val_d0c1 = data_fun.test_data_get(test_filename)
    #
    # val_341c['csi_state'] = [int(a) for a in val_341c['csi_state']]
    # # train_341c['mcs'] = [float(a) for a in train_341c['mcs']]
    # val_341c['dfx_time'] = [float(a) for a in val_341c['dfx_time']]
    # val_341c['csi_time'] = [float(a) for a in val_341c['csi_time']]
    # val_341c['beamforming_en'] = [int(a) for a in val_341c['beamforming_en']]
    # val_341c['noise_floor'] = [float(a) for a in val_341c['noise_floor']]
    #
    # val_bc98['csi_state'] = [int(a) for a in val_bc98['csi_state']]
    # # val_bc98['mcs'] = [float(a) for a in val_bc98['mcs']]
    # val_bc98['dfx_time'] = [float(a) for a in val_bc98['dfx_time']]
    # val_bc98['csi_time'] = [float(a) for a in val_bc98['csi_time']]
    # val_bc98['beamforming_en'] = [int(a) for a in val_bc98['beamforming_en']]
    # val_bc98['noise_floor'] = [float(a) for a in val_bc98['noise_floor']]
    #
    # val_d0c1['csi_state'] = [int(a) for a in val_d0c1['csi_state']]
    # # val_d0c1['mcs'] = [float(a) for a in val_d0c1['mcs']]
    # val_d0c1['dfx_time'] = [float(a) for a in val_d0c1['dfx_time']]
    # val_d0c1['csi_time'] = [float(a) for a in val_d0c1['csi_time']]
    # val_d0c1['beamforming_en'] = [int(a) for a in val_d0c1['beamforming_en']]
    # val_d0c1['noise_floor'] = [float(a) for a in val_d0c1['noise_floor']]
    #
    #
    # mat_filename = [f"data/341c_val.mat",
    #                 f"data/bc98_val.mat",
    #                 f"data/d0c1_val.mat"]
    # savemat(mat_filename[0], val_341c)
    # savemat(mat_filename[1], val_bc98)
    # savemat(mat_filename[2], val_d0c1)

    # 带beamforming
    train_filename = ['data/txbf_com_excels_f4/341c-f0d4-70be/train/341c-f0d4-70be_0.xlsx',
                        'data/txbf_com_excels_f4/508e-4963-d5dc/train/508e-4963-d5dc_0.xlsx',
                        'data/txbf_com_excels_f4/508e-4963-e122/train/508e-4963-e122_0.xlsx',
                        'data/txbf_com_excels_f4/bc98-2983-907b/train/bc98-2983-907b_0.xlsx',
                        'data/txbf_com_excels_f4/d0c1-bfe4-18f0/train/d0c1-bfe4-18f0_0.xlsx',
                        'data/txbf_com_excels_f4/fc02-9668-e2fd/train/fc02-9668-e2fd_0.xlsx']


    test_filename = ['data/txbf_com_excels_f4/341c-f0d4-70be/valid/341c-f0d4-70be_0.xlsx',
                        'data/txbf_com_excels_f4/508e-4963-d5dc/valid/508e-4963-d5dc_0.xlsx',
                        'data/txbf_com_excels_f4/508e-4963-e122/valid/508e-4963-e122_0.xlsx',
                        'data/txbf_com_excels_f4/bc98-2983-907b/valid/bc98-2983-907b_0.xlsx',
                        'data/txbf_com_excels_f4/d0c1-bfe4-18f0/valid/d0c1-bfe4-18f0_0.xlsx',
                        'data/txbf_com_excels_f4/fc02-9668-e2fd/valid/fc02-9668-e2fd_0.xlsx']
    # train_341c_beam,train_508ed5dc_beam,train_508ee122_beam,train_bc98_beam,train_d0c1_beam,train_fc02_beam = data_fun.train_data_get_beam(train_filename)
    # mat_filename = [f"data/341c_beam_train.mat",
    #                 f"data/508ed5dc_beam_train.mat",
    #                 f"data/508ee122_beam_train.mat",
    #                 f"data/bc98_beam_train.mat",
    #                 f"data/d0c1_beam_train.mat",
    #                 f"data/fc02_beam_train.mat"]
    #
    # map_list = [train_341c_beam,train_508ed5dc_beam,train_508ee122_beam,train_bc98_beam,train_d0c1_beam,train_fc02_beam]
    # for name,map in zip(mat_filename,map_list):
    #     savemat(name,map)

    val_341c_beam, val_508ed5dc_beam, val_508ee122_beam, val_bc98_beam, val_d0c1_beam, val_fc02_beam = data_fun.test_data_get_beam(test_filename)

    mat_filename = [f"data/341c_beam_val.mat",
                    f"data/508ed5dc_beam_val.mat",
                    f"data/508ee122_beam_val.mat",
                    f"data/bc98_beam_val.mat",
                    f"data/d0c1_beam_val.mat",
                    f"data/fc02_beam_val.mat"]
    map_list = [val_341c_beam, val_508ed5dc_beam, val_508ee122_beam, val_bc98_beam, val_d0c1_beam, val_fc02_beam]
    for name, map in zip(mat_filename, map_list):
        savemat(name, map)


a = 1


