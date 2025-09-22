#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取当前目录及子目录的 .mat：
  csi_matrix_r{0,1}_c{0..3} : [T,122] complex double
  noise_floor               : 1×T / T×1 / T 或 T×6 char（单位 dBm）
  rate                      : 1×T double 或 T×K char（可选，供拟合）

计算：
  γ_sc(t,k) = P_tx(mW)*||H_{t,k}||_F^2 / P_noise(t,mW)
  γ_eff(t; α,β) = - α * ln( mean_k exp( - γ_sc(t,k)/β ) )
  I_EXP(t) = 1 - exp( - γ_eff(t) )

拟合目标（最小二乘）：
  min_{α>0, β>0}  MSE( I_EXP(t; α,β),  y(t) )
  其中 y(t) 为来自 rate 字段的目标值。

输出：
  - 拟合到的 α、β
  - 保存 numpy：eesm_gamma_eff.npy / eesm_gamma_per_sc.npy
  - CSV 汇总：eesm_results_summary.csv（含每样本 I_EXP 与 y、误差）
"""

import os
import re
import glob
import csv
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ========================
# 固定参数
# ========================
N_RX = 2
N_TX = 4
N_SC = 122
P_TX_W = 1.0                      # 发射功率：1 W
P_TX_MW = P_TX_W                    # 换算至 mW（线性功率统一用 mW）
EPS = 1e-12

SEARCH_DIRS = ["."]
# EXTRA_DIRS  = ["341c_train", "bc98_train", "d0c1_train"]
FILE  = "341c_train.mat"

# ========================
# 工具函数
# ========================

def dbm_to_w(dbm: np.ndarray) -> np.ndarray:
    """dBm -> mW：P[mW] = 10^(dBm/10)。"""
    return np.power(10.0, dbm / 10.0) / 1000

def load_noise_dbm(noise_obj, T: int) -> np.ndarray:
    """将 noise_floor 转为长度 T 的 dBm 向量。"""
    arr = np.asarray(noise_obj)
    noise_dbm = np.squeeze(arr).astype(float).reshape(-1)
    if noise_dbm.size != T:
        raise ValueError(f"noise_floor 长度应为 {T}，实际 {noise_dbm.size}")
    return noise_dbm

def load_rate_target(rate_obj, T: int) -> np.ndarray:
    """
    读取 rate 字段为目标 y(t)：
      - 若为数值：直接拉平成 [T]
      - 若为字符：逐行解析为 float
    """
    arr = np.asarray(rate_obj)
    rate = np.squeeze(arr).astype(float).reshape(-1)
    if rate.size != T:
        raise ValueError(f"rate 长度应为 {T}，实际 {rate.size}")
    return rate

def load_one_mat(path: str):
    """
    读取一个 .mat 并返回：
      H: [T,122,4,2] complex
      noise_dbm: [T]
      rate (可为 None)：[T] 目标（来自 rate）
    """
    md = loadmat(path, squeeze_me=True, struct_as_record=False)

    # --- 信道 8 个矩阵 ---
    mats = []
    for r in range(N_RX):
        for c in range(N_TX):
            key = f"csi_matrix_r{r}_c{c}"
            mat = np.asarray(md[key])
            if mat.ndim != 2 or mat.shape[1] != N_SC:
                raise ValueError(f"{key} 形状应为 [T,{N_SC}]，实际 {mat.shape}")
            if not np.iscomplexobj(mat):
                mat = mat.astype(np.complex128)
            mats.append(mat)
    T = mats[0].shape[0]
    H = np.stack(mats, axis=-1).reshape(T, N_SC, N_RX, N_TX)

    # --- 噪声 ---
    noise_dbm = load_noise_dbm(md["noise_floor"], T)

    # --- 目标---
    rate = None
    if "mcs" in md:
        rate = load_rate_target(md["mcs"], T)

    return H, noise_dbm, rate

def compute_gamma_sc(H: np.ndarray, noise_dbm: np.ndarray) -> np.ndarray:
    """
    先计算并返回所有样本的逐子载波 SINR（线性域），形状 [T,122]。
    这样拟合 α、β 时无需重复计算信号与噪声部分。
    """
    H_power = np.sum(np.abs(H) ** 2, axis=(2,3))      # [T,122], ||H||_F^2
    sig_mw  = H_power                       # 乘 P_tx
    noise_w = dbm_to_w(noise_dbm)                   # [T]
    gamma_sc = sig_mw / noise_w[:,None]              # [T,122]
    return gamma_sc

def gamma_eff_from_gamma_sc(gamma_sc: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """根据给定 α、β，从已算好的逐子载波 SINR 得到 γ_eff（线性域）。"""
    exp_term = np.exp(-(gamma_sc / max(beta, EPS)))   # 防 0
    mean_exp = np.mean(exp_term, axis=1)              # [T]
    mean_exp = np.clip(mean_exp, a_min=EPS, a_max=None)
    gamma_eff = -alpha * np.log(mean_exp)             # [T]
    return gamma_eff

def i_exp(gamma_eff: np.ndarray) -> np.ndarray:
    """I_EXP(gamma) = 1 - exp(-gamma)。"""
    # return 1.0 - np.exp(-gamma_eff)
    return 0.5 * np.log2(1 + gamma_eff)  # 使用 Shannon 公式作为替代

# ========================
# 拟合 α、β（最小二乘）
# ========================
def fit_alpha_beta(gamma_sc_all: np.ndarray, rate_all: np.ndarray,
                   alpha0: float = 1.0, beta0: float = 1.0) -> tuple:
    """
    两段式：粗网格 + Nelder–Mead。
    gamma_sc_all: [N,122]
    rate_all    : [N]（不进行缩放）
    """
    # --- 粗搜网格 ---
    alphas = np.geomspace(0.1, 100.0, 1)    # 0.1~100
    betas  = np.geomspace(0.1, 100.0, 1)    # 0.1~10
    best = (np.inf, alpha0, beta0)

    for a in alphas:
        for b in betas:
            ge = gamma_eff_from_gamma_sc(gamma_sc_all, a, b)
            pred = i_exp(ge)
            mse = np.mean((pred - rate_all) ** 2)
            if mse < best[0]:
                best = (mse, a, b)

    # --- Nelder–Mead 细化 ---
    def obj(x):
        a = float(np.abs(x[0])) + 1e-8   # 约束 a,b > 0（用绝对值替代）
        b = float(np.abs(x[1])) + 1e-8
        ge = gamma_eff_from_gamma_sc(gamma_sc_all, a, b)
        pred = i_exp(ge)
        return np.mean((pred - rate_all) ** 2)

    x0 = np.array([best[1], best[2]])
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6, "disp": False})

    alpha_hat = float(np.abs(res.x[0]) + 1e-8)
    beta_hat  = float(np.abs(res.x[1]) + 1e-8)
    return alpha_hat, beta_hat

# ========================
# 主流程
# ========================
def main(alpha_init=1.0, beta_init=1.0, do_fit=True):
    
    roots = [d for d in SEARCH_DIRS if os.path.isdir(d)]
    if not roots:
        raise SystemExit(f"未找到指定的目录：{SEARCH_DIRS}")

    paths = []
    for r in roots:
        paths += glob.glob(os.path.join(r, "**", FILE), recursive=True)
    paths = sorted(list(set(paths)))
    if not paths:
        raise SystemExit(f"未在 {SEARCH_DIRS} 目录中找到 .mat 文件。")

    # 聚合所有样本（便于全局拟合 α/β）
    Gamma_all, Rate_all, file_marks = [], [], []
    per_file_results = []

    for p in paths:
        H, noise_dbm, rate = load_one_mat(p)
        gamma_sc = compute_gamma_sc(H, noise_dbm)        # [T,122]
        Gamma_all.append(gamma_sc)

        if rate is not None:
            # 不进行缩放，直接使用原始 rate 值
            Rate_all.append(rate)
        else:
            # 若无 rate，填 None；后续只做预测不参与拟合
            Rate_all.append(None)

        file_marks.append((p, gamma_sc.shape[0]))

    # 拼接 gamma_sc 与 rate（只保留有标签的文件）
    Gamma_with_rate = [Gamma_all[i] for i, rate in enumerate(Rate_all) if rate is not None]
    Rate_with_label = [rate for rate in Rate_all if rate is not None]

    if len(Rate_with_label) == 0:
        raise SystemExit("未在任何 .mat 中发现 rate 字段，无法拟合 α/β。")

    # 拼接成整体矩阵/向量
    G_concat = np.concatenate(Gamma_with_rate, axis=0)   # [N_labeled, 122]
    Rate_concat = np.concatenate(Rate_with_label, axis=0)  # [N_labeled]

    # 拟合 α、β
    if do_fit:
        alpha_hat, beta_hat = fit_alpha_beta(G_concat[:Rate_concat.shape[0], :],
                                             Rate_concat, alpha_init, beta_init)
        print(f"[拟合完成] alpha={alpha_hat:.6f}  beta={beta_hat:.6f}")
    else:
        alpha_hat, beta_hat = float(alpha_init), float(beta_init)
        print(f"[使用给定参数] alpha={alpha_hat}  beta={beta_hat}")

    # 用最终 α、β 计算所有样本的 γ_eff 与 I_EXP，并保存结果
    gamma_eff_all = gamma_eff_from_gamma_sc(G_concat, alpha_hat, beta_hat)  # [Total_N]
    i_exp_all     = i_exp(gamma_eff_all)

    # 保存逐子载波 SINR（注意体量可能较大）
    np.save("eesm_gamma_per_sc.npy", G_concat.astype(np.float64))
    np.save("eesm_gamma_eff.npy",     gamma_eff_all.astype(np.float64))

    # 生成 CSV：逐文件逐样本记录（若有 rate，同步写入误差）
    rows = []
    offset = 0
    for (p, T) in file_marks:
        base = os.path.basename(p)
        Iseg = i_exp_all[offset:offset+T]
        Gseg = gamma_eff_all[offset:offset+T]
        # 对应 rate（若文件有标签），否则填空
        rate_seg = None
        idx = paths.index(p)
        if Rate_all[idx] is not None:
            rate_seg = Rate_all[idx]
        for t in range(T):
            if rate_seg is not None:
                rows.append([base, t, float(Gseg[t]), float(Iseg[t]), float(rate_seg[t]), float(Iseg[t]-rate_seg[t])])
            else:
                rows.append([base, t, float(Gseg[t]), float(Iseg[t]), "", ""])
        offset += T

    with open("eesm_results_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mat_file","time_index","gamma_eff","I_EXP","target_rate","I_EXP_minus_target"])
        w.writerows(rows)

    # 绘制时间序列对比图（只需要第三张图）
    if Rate_concat is not None:
        plt.figure(figsize=(12, 6))
        # sample_indices = np.arange(min(200, len(Rate_concat)))
        sample_indices = np.arange(len(Rate_concat))
        # plt.plot(sample_indices, Rate_concat[:200], 'b-', label='Target Rate', linewidth=2)
        # plt.plot(sample_indices, i_exp_all[:200], 'r-', label='I_EXP Prediction', alpha=0.7)
        plt.plot(sample_indices, Rate_concat, 'b-', label='Target Rate', linewidth=2)
        plt.plot(sample_indices, i_exp_all, 'r-', label='I_EXP Prediction', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Time Series Comparison (First 200 samples)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('eesm_time_series_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 计算并显示误差指标
        errors = i_exp_all[:len(Rate_concat)] - Rate_concat
        print(f"MSE: {np.mean(errors**2):.6f}")
        print(f"MAE: {np.mean(np.abs(errors)):.6f}")
        print(f"Max Error: {np.max(np.abs(errors)):.6f}")

    print("结果已保存：eesm_gamma_per_sc.npy / eesm_gamma_eff.npy / eesm_results_summary.csv / eesm_time_series_comparison.png")

if __name__ == "__main__":
    main(alpha_init=1.0, beta_init=1.0, do_fit=True)