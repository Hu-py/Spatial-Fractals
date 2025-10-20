#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:18:30 2025

@author: zha
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import bisect
import statsmodels.api as sm

# 可选 numba 加速
try:
    from numba import njit
    NUMBA = True
except:
    NUMBA = False

# =============================================================
# Utilities
# =============================================================
class Step:
    def __init__(self, angle_deg, ratio):
        self.angle_deg = angle_deg
        self.ratio = ratio

def make_generator(steps):
    return [Step(a,r) for a,r in steps]

def iterate_polyline(points, generator):
    new_pts = []
    for i in range(len(points)-1):
        z0 = complex(points[i,0], points[i,1])
        z1 = complex(points[i+1,0], points[i+1,1])
        seg = z1 - z0
        L = abs(seg)
        if L==0: continue
        heading = np.angle(seg)
        cur = z0
        new_pts.append([cur.real, cur.imag])
        theta = heading
        for s in generator:
            theta += np.deg2rad(s.angle_deg)
            step_len = L * s.ratio
            cur = cur + step_len * np.exp(1j*theta)
            new_pts.append([cur.real, cur.imag])
    return np.array(new_pts)

def build_fractal(generator, iterations, initiator=((0.0,0.0),(1.0,0.0))):
    pts = np.array(initiator, dtype=float)
    for _ in range(iterations):
        pts = iterate_polyline(pts, generator)
    return pts

def similarity_dimension(ratios):
    ratios = [r for r in ratios if r>0]
    f = lambda D: np.sum(np.power(ratios,D))-1.0
    return bisect(f,0,5)

def rasterize_polyline(pts,R=512):
    bb_min = pts.min(axis=0); bb_max = pts.max(axis=0)
    scale = (bb_max - bb_min).max() or 1.0
    pts_n = (pts - bb_min)/scale
    grid = np.zeros((R,R),dtype=np.uint8)
    for i in range(len(pts_n)-1):
        x0,y0 = pts_n[i]; x1,y1 = pts_n[i+1]
        steps = int(max(abs(x1-x0),abs(y1-y0))*R)+1
        for t in range(steps+1):
            u = t/max(1,steps)
            x = (1-u)*x0 + u*x1
            y = (1-u)*y0 + u*y1
            ix = min(R-1,max(0,int(x*R)))
            iy = min(R-1,max(0,int(y*R)))
            grid[iy,ix] = 1
    return grid

def box_counting_dimension(grid, ks=[2,4,8,16,32,64]):
    N = grid.shape[0]
    eps = []
    counts = []
    for k in ks:
        s = N//k
        if s<1: continue
        cnt = 0
        for i in range(k):
            for j in range(k):
                block = grid[i*s:(i+1)*s, j*s:(j+1)*s]
                if block.max()>0: cnt+=1
        eps.append(1.0/k)
        counts.append(cnt)
    x = np.log(1/np.array(eps)); y = np.log(np.array(counts)+1e-9)
    A = np.vstack([x, np.ones_like(x)]).T
    D,c = np.linalg.lstsq(A,y,rcond=None)[0]
    return D, (x,y)

# =============================================================
# Streamlit App
# =============================================================

st.set_page_config(page_title="Spatial Fractals & Urban Scaling", layout="wide")
st.title("Spatial Fractals & Urban Scaling Demo")

tabA, tabB, tabC = st.tabs(["A: Initiator-Generator", "B: Multiplicative Cascades", "C: Urban Scaling"])

# -------- Sidebar 动态控件 --------
# 先在 sidebar 放一个占位 container
sidebar_container = st.sidebar.container()

# =============================================================
# === A: Fractal Generator ===
# =============================================================
with tabA:
    st.subheader("Fractal Generator (Initiator & Generator)")

    PRESETS_A = {
        'Koch (classic)': [(0,1/3),(60,1/3),(-120,1/3),(60,1/3)],
        'V (acute)': [(0,0.5),(60,0.5)],
        'Cantor-like 1D': [(0,1/3),(0,1/3)],
        'Dragon-ish': [(45,1/np.sqrt(2)),(-90,1/np.sqrt(2))],
    }

    # 在 container 里生成 tabA 的 sidebar 控件
    with sidebar_container:
        st.subheader("Fractal Generator Controls")
        preset = st.selectbox("Preset", list(PRESETS_A.keys())+["Custom 2-step"])
        iters = st.slider("Iterations", 1, 7, 4)
        angle = st.slider("Angle (step 2)", -180.0,180.0,60.0)
        ratio = st.slider("Ratio (step 1)",0.05,0.95,0.333)
        gen_button = st.button("Generate Fractal", key="gen_fractal")

    if gen_button:
        if preset=="Custom 2-step":
            steps = [(0.0, ratio), (angle, 1.0-ratio)]
        else:
            steps = PRESETS_A[preset]
        gen = make_generator(steps)
        pts = build_fractal(gen, iters)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(pts[:,0], pts[:,1], lw=1.5)
        ax.set_aspect('equal'); ax.axis('off')
        st.pyplot(fig)

        ratios = [s.ratio for s in gen]
        try:
            D_sim = similarity_dimension(ratios)
        except:
            D_sim = np.nan
        grid = rasterize_polyline(pts,R=512)
        D_box, (x,y) = box_counting_dimension(grid)
        st.write(f"Similarity dimension (theory): {D_sim:.4f}")
        st.write(f"Box-counting dimension (empirical): {D_box:.4f}")

        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.scatter(x,y)
        A = np.vstack([x,np.ones_like(x)]).T
        D_hat, c = np.linalg.lstsq(A,y,rcond=None)[0]
        ax2.plot(x, D_hat*x+c, '--')
        ax2.set_xlabel('log(1/ε)'); ax2.set_ylabel('log N(ε)')
        st.pyplot(fig2)

# =============================================================
# === B: Multiplicative Cascades ===
# =============================================================
with tabB:
    st.subheader("Multiplicative Cascades (2x2 or 3x3)")

    # 动态 sidebar
    with sidebar_container:
        st.subheader("Multiplicative Cascade Controls")
        presetB = st.selectbox("Preset B", ["Quad balanced (2x2)","Quad concentrated (2x2)","Nonet balanced (3x3)","Custom"], key="presetB")
        branch = st.selectbox("Branch", [2,3], key="branch")
        levels = st.slider("Levels",4,9,7, key="levels")
        weights_text = st.text_input("Weights (comma-separated)", "0.4,0.3,0.2,0.1", key="weights")
        cascade_button = st.button("Generate Cascade", key="gen_cascade")

    if cascade_button:
        ws = [float(w) for w in weights_text.split(',') if w.strip()]
        grid = np.ones((1,1))
        np.random.seed(1)
        for _ in range(levels):
            h,w0 = grid.shape
            new = np.zeros((h*branch, w0*branch))
            for i in range(h):
                for j in range(w0):
                    ws_perm = np.random.permutation(ws)
                    idx=0
                    for bi in range(branch):
                        for bj in range(branch):
                            new[branch*i+bi, branch*j+bj] = grid[i,j]*ws_perm[idx]
                            idx+=1
            grid=new
        grid = grid/grid.sum()
        if grid.shape[0]>256:
            grid=grid[:256,:256]
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(grid, origin='lower', cmap='magma')
        ax.axis('off')
        ax.set_title(f"Cascade {branch}x{branch}, levels={levels}")
        st.pyplot(fig)

# =============================================================
# === C: Urban Scaling ===
# =============================================================
with tabC:
    st.subheader("Urban Scaling: multi-indicator fits")

    # 动态 sidebar
    with sidebar_container:
        st.subheader("Urban Scaling Controls")
        scenario_dd = st.selectbox("Scenario", ["Default (sub/≈lin/super)","All sublinear","All superlinear"], key="scenario")
        noise = st.slider("Noise σ", 0.0, 0.6, 0.2, step=0.05, key="noise")
        M = st.slider("Num cities", 60, 400, 120, step=10, key="M")
        seed = st.number_input("Seed", 0, 999, 0, key="seed")
        scaling_button = st.button("Generate Scaling Data", key="gen_scaling")

    sidebar_container = st.sidebar.container()

with sidebar_container:
    st.subheader("Urban Scaling Controls")
    scenario_dd = st.selectbox("Scenario", ["Default (sub/≈lin/super)","All sublinear","All superlinear"])
    noise = st.slider("Noise σ", 0.0, 0.6, 0.2, step=0.05)
    M = st.slider("Num cities", 60, 400, 120, step=10)
    seed = st.number_input("Seed", 0, 999, 0)
    scaling_button = st.button("Generate Scaling Data")

SCENARIOS = {
    'Default (sub/≈lin/super)': {'Infrastructure (sublinear)':0.85,'Employment (≈linear)':1.0,'Innovation (superlinear)':1.15},
    'All sublinear': {'Road length':0.85,'Electric network':0.88,'Water pipes':0.83},
    'All superlinear': {'Patents':1.15,'Creative jobs':1.20,'High-tech firms':1.12},
}

if scaling_button:
    betas_true = SCENARIOS[scenario_dd]
    rng = np.random.default_rng(seed)
    pop = rng.lognormal(mean=11.0, sigma=0.8, size=M)
    mat = {}
    Y0 = {name:0.5 for name in betas_true}
    for name,beta in betas_true.items():
        noise_vals = rng.lognormal(mean=0.0, sigma=noise, size=M)
        mat[name] = Y0[name]*(pop**beta)*noise_vals
    df = pd.DataFrame({'Population':pop,**mat})

    # Fit OLS
    x = np.log(df['Population'].values)
    X = sm.add_constant(x)
    results = {}
    for col in df.columns:
        if col=='Population': continue
        y = np.log(df[col].values)
        model = sm.OLS(y,X).fit()
        results[col]=model

    # Display table
    rows=[]
    for name,beta in betas_true.items():
        m = results[name]
        ci_low, ci_high = m.conf_int().iloc[1]
        rows.append([name,beta,m.params[1],ci_low,ci_high,m.rsquared])
    st.dataframe(pd.DataFrame(rows,columns=['Indicator','β true','β_hat','CI low','CI high','R²']))

    # Plot
    cols = [c for c in df.columns if c!='Population']
    fig, axes = plt.subplots(1,len(cols),figsize=(5*len(cols),4))
    if len(cols)==1: axes=[axes]
    for ax,col in zip(axes,cols):
        y = np.log(df[col].values)
        m = results[col]
        ax.scatter(x,y,alpha=0.6)
        xx = np.linspace(x.min(),x.max(),200)
        ax.plot(xx, m.params[0]+m.params[1]*xx,'--')
        ax.set_title(f"{col}\nβ_hat={m.params[1]:.3f} (95% CI {m.conf_int().iloc[1,0]:.3f}-{m.conf_int().iloc[1,1]:.3f})\nR²={m.rsquared:.2f}")
        ax.set_xlabel('log Population'); ax.set_ylabel(f'log {col}')
    st.pyplot(fig)
