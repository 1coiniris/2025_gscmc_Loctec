import pandas as pd
from numpy.polynomial.chebyshev import chebpts2
from openpyxl import load_workbook
import numpy as np
import shutil
import ast

def train_data_get(train_filename):
    # 读取Excel文件
    # 假设数据从第一个sheet开始，且数据不需要跳过任何行或列
    # sheet_name = 0  # 或者使用sheet_name='Sheet1'（如果知道具体的sheet名）
    Map_341c = {}
    Map_bc98 = {}
    Map_d0c1 = {}
    for case, filename in enumerate(train_filename):
        df = pd.read_excel(filename,header=None)
        column_names = [
            'csi_state','csi_matrix_r0_c0', 'csi_matrix_r0_c1', 'csi_matrix_r0_c2', 'csi_matrix_r0_c3',
            'csi_matrix_r1_c0', 'csi_matrix_r1_c1', 'csi_matrix_r1_c2', 'csi_matrix_r1_c3',
            'mcs', 'dfx_time','csi_time', 'beamforming_en',  'noise_floor'
        ]

        num_rows = df.shape[0]
        num_cols = df.shape[1]
        df_data = df2list(df,[0,0],[num_rows,num_cols])
        # for csi_state,r0c0,r0c1,r0c2,r0c3,r1c0,r1c1,r1c2,r1c3,mcs,dfx_time,csi_time,beam_en,noise in df_data[:]:
        #     Map_341c[f'csi_state'] = csi_state
        for i, col_name in enumerate(column_names):
            data_list = []
            if i>=1 and i<=8:
                for str in df_data[i][1:]:
                    complex_list = ast.literal_eval(str)
                    data_list.append(np.array(complex_list))
                data_i = data_list
            else:
                data_i = df_data[i][1:]

            if case == 0:
                Map_341c[col_name] = data_i
            if case == 1:
                Map_bc98[col_name] = data_i
            if case == 2:
                Map_d0c1[col_name] = data_i

    return  Map_341c,Map_bc98,Map_d0c1

def test_data_get(test_filename):
    # 读取Excel文件
    # 假设数据从第一个sheet开始，且数据不需要跳过任何行或列
    # sheet_name = 0  # 或者使用sheet_name='Sheet1'（如果知道具体的sheet名）
    Map_341c = {}
    Map_bc98 = {}
    Map_d0c1 = {}
    for case, filename in enumerate(test_filename):
        df = pd.read_excel(filename,header=None)
        column_names = [
            'csi_state','csi_matrix_r0_c0', 'csi_matrix_r0_c1', 'csi_matrix_r0_c2', 'csi_matrix_r0_c3',
            'csi_matrix_r1_c0', 'csi_matrix_r1_c1', 'csi_matrix_r1_c2', 'csi_matrix_r1_c3',
            'dfx_time','csi_time', 'beamforming_en',  'noise_floor'
        ]

        num_rows = df.shape[0]
        num_cols = df.shape[1]
        df_data = df2list(df,[0,0],[num_rows,num_cols])
        # for csi_state,r0c0,r0c1,r0c2,r0c3,r1c0,r1c1,r1c2,r1c3,mcs,dfx_time,csi_time,beam_en,noise in df_data[:]:
        #     Map_341c[f'csi_state'] = csi_state
        for i, col_name in enumerate(column_names):
            data_list = []
            if i>=1 and i<=8:
                for str in df_data[i][1:]:
                    complex_list = ast.literal_eval(str)
                    data_list.append(np.array(complex_list))
                data_i = data_list
            else:
                data_i = df_data[i][1:]

            if case == 0:
                Map_341c[col_name] = data_i
            if case == 1:
                Map_bc98[col_name] = data_i
            if case == 2:
                Map_d0c1[col_name] = data_i

    return  Map_341c,Map_bc98,Map_d0c1

def train_data_get_beam(train_filename):
    # 读取Excel文件
    # 假设数据从第一个sheet开始，且数据不需要跳过任何行或列
    # sheet_name = 0  # 或者使用sheet_name='Sheet1'（如果知道具体的sheet名）
    train_341c_beam={}
    train_508ed5dc_beam={}
    train_508ee122_beam={}
    train_bc98_beam={}
    train_d0c1_beam={}
    train_fc02_beam={}
    # Map_341c = {}
    # Map_bc98 = {}
    # Map_d0c1 = {}
    for case, filename in enumerate(train_filename):
        df = pd.read_excel(filename,header=None)
        column_names = [
            'csi_state','csi_matrix_r0_c0', 'csi_matrix_r0_c1', 'csi_matrix_r0_c2', 'csi_matrix_r0_c3',
            'csi_matrix_r1_c0', 'csi_matrix_r1_c1', 'csi_matrix_r1_c2', 'csi_matrix_r1_c3',
            'mcs', 'dfx_time','csi_time', 'beamforming_en',  'noise_floor'
        ]

        num_rows = df.shape[0]
        num_cols = df.shape[1]
        df_data = df2list(df,[0,0],[num_rows,num_cols])
        # for csi_state,r0c0,r0c1,r0c2,r0c3,r1c0,r1c1,r1c2,r1c3,mcs,dfx_time,csi_time,beam_en,noise in df_data[:]:
        #     Map_341c[f'csi_state'] = csi_state
        for i, col_name in enumerate(column_names):
            data_list = []
            if i>=1 and i<=8:
                for str in df_data[i][1:]:
                    complex_list = ast.literal_eval(str)
                    data_list.append(np.array(complex_list))
                data_i = data_list
            else:
                if i==0:
                    data_i = [int(a) for a in df_data[i][1:]]
                else:
                    data_i = [float(a) for a in df_data[i][1:]]

            if case == 0:
                train_341c_beam[col_name] = data_i
            if case == 1:
                train_508ed5dc_beam[col_name] = data_i
            if case == 2:
                train_508ee122_beam[col_name] = data_i
            if case == 3:
                train_bc98_beam[col_name] = data_i
            if case == 4:
                train_d0c1_beam[col_name] = data_i
            if case == 5:
                train_fc02_beam[col_name] = data_i

    return  train_341c_beam,train_508ed5dc_beam,train_508ee122_beam,train_bc98_beam,train_d0c1_beam,train_fc02_beam

def test_data_get_beam(test_filename):
    # 读取Excel文件
    # 假设数据从第一个sheet开始，且数据不需要跳过任何行或列
    # sheet_name = 0  # 或者使用sheet_name='Sheet1'（如果知道具体的sheet名）
    train_341c_beam={}
    train_508ed5dc_beam={}
    train_508ee122_beam={}
    train_bc98_beam={}
    train_d0c1_beam={}
    train_fc02_beam={}
    for case, filename in enumerate(test_filename):
        df = pd.read_excel(filename,header=None)
        column_names = [
            'csi_state','csi_matrix_r0_c0', 'csi_matrix_r0_c1', 'csi_matrix_r0_c2', 'csi_matrix_r0_c3',
            'csi_matrix_r1_c0', 'csi_matrix_r1_c1', 'csi_matrix_r1_c2', 'csi_matrix_r1_c3',
            'dfx_time','csi_time', 'beamforming_en',  'noise_floor'
        ]

        num_rows = df.shape[0]
        num_cols = df.shape[1]
        df_data = df2list(df,[0,0],[num_rows,num_cols])
        # for csi_state,r0c0,r0c1,r0c2,r0c3,r1c0,r1c1,r1c2,r1c3,mcs,dfx_time,csi_time,beam_en,noise in df_data[:]:
        #     Map_341c[f'csi_state'] = csi_state
        for i, col_name in enumerate(column_names):
            data_list = []
            if i>=1 and i<=8:
                for str in df_data[i][1:]:
                    complex_list = ast.literal_eval(str)
                    data_list.append(np.array(complex_list))
                data_i = data_list
            else:
                if i == 0:
                    data_i = [int(a) for a in df_data[i][1:]]
                else:
                    data_i = [float(a) for a in df_data[i][1:]]

            if case == 0:
                train_341c_beam[col_name] = data_i
            if case == 1:
                train_508ed5dc_beam[col_name] = data_i
            if case == 2:
                train_508ee122_beam[col_name] = data_i
            if case == 3:
                train_bc98_beam[col_name] = data_i
            if case == 4:
                train_d0c1_beam[col_name] = data_i
            if case == 5:
                train_fc02_beam[col_name] = data_i

    return train_341c_beam, train_508ed5dc_beam, train_508ee122_beam, train_bc98_beam, train_d0c1_beam, train_fc02_beam

def unmerge_df(df):
    # 将合并的单元格值还原值
    df0 = df.copy()
    # fillna 这里之前用 '-' 作识别值，但是如果原表中有 '-' 则会出现错误，故用 'add58' 作为识别符
    df0.fillna('add58', inplace=True)  # 缺失值替换
    rows = df0.shape[0]
    cols = df0.shape[1]
    for i in range(rows - 1):  # 循环行
        for j in range(cols):  # 循环列
            if i < rows - 1:  # 不是最后一行
                a = df0.iloc[i, j]  # 某行某列的值
                b = df0.iloc[i + 1, j]  # 某下一行某列的值
                if b == 'add58':  # 是缺失值
                    df0.iloc[i + 1, j] = a  # 用上一行非缺失值替换
    for i in range(cols - 1):  # 循环列
        for j in range(rows):  # 循环行
            if i < cols - 1:  # 不是最后一列
                a = df0.iloc[j, i]  # 某行某列的值
                b = df0.iloc[j, i + 1]  # 某行某下一列的值
                if b == 'add58':  # 是缺失值
                    df0.iloc[j, i + 1] = a  # 用上一行非缺失值替换
    # 以下这行新加，主要是当某列全为空时，会被填充为add58不好看，还原回去
    df0.replace({'add58': np.nan}, inplace=True)
    return df0


def df2list(df,data_start = [0,0],data_end = 'none'):
    data_start_row = data_start[0]  # 实际上从0开始计数，但这里为了说明从第二行开始
    data_start_col = data_start[1]  # B列，从0开始计数则为1
    if data_end == 'none':
        num_rows = df.shape[0]
        num_cols = df.shape[1]
    else:
        num_rows = data_end[0]
        num_cols = data_end[1]
    # 初始化一个空列表来存储数据
    data_list = []
    # 遍历数据的有效范围，并将其添加到列表中
    for col in range(data_start_col, num_cols):
        col_data = []
        for row in range(data_start_row, num_rows):
            cell_value = df.iat[row, col]
            col_data.append(cell_value)
        # 添加到总列表中，每个内部列表代表一行数据
        data_list.append(col_data)
    return  data_list