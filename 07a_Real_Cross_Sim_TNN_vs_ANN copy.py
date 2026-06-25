'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 25 May 2026



07a_Real_Cross_Sim_TNN_vs_ANN.py



after modeling programmign error and drift in file 05 , here in file 06 we
got ideal ANN and TNN to only see that everything is work because
the digital and ideal must be same to prove cross sim is sync with pytorch
and even the Rmin Rmax and any devcie related doesn't matter becuase they
were Ideal.


In 07b we consider Rmin, max , programming error and ....
but here we donot want to consider programming error, we want  to only
comapre evrything TNN vs ANN


First we have ideal (from 06) for ANN and TNN.

then we compare the TNN and ANN in terms of configurations. and we decide
to go with TNN.
    



'''



# ============================================
'''                   Imports              '''
# ============================================
from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path

import torch.optim as optim

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from pathlib import Path

CROSS_SIM_DIR = Path("/Users/apm/Desktop/MASTER THESIS/Projects/04_VOL_Non_Vol/cross-sim")
if str(CROSS_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(CROSS_SIM_DIR))

from simulator import CrossSimParameters
from simulator.parameters.xbar_parameters import ADCRangeLimits
from simulator.algorithms.dnn.torch.convert import from_torch

from simulator.algorithms.dnn.torch.convert import convertible_modules



# ============================================
'''                Variables              '''
# ============================================
SEED = 42  # change this to try other runs; keep fixed for identical results

batch_size = 16
BATCH_SIZE = batch_size

learning_rate=1e-4 #for TNN
LEARNING_RATE = learning_rate

num_epochs = 100
NUM_EPOCHS = num_epochs

SAVE_FIGS = False
SAVE_CHECKPOINTS = False



MAIN_DIR = '/Users/apm/Desktop/MASTER THESIS/Projects/04_VOL_Non_Vol/24May/'
PTH_DIR= '/Users/apm/Desktop/MASTER THESIS/Projects/04_VOL_Non_Vol/24May/Pth_Models/'


FIGURE_DPI = 150
THRESHOLD = 0.05
SMOOTH_TW_WIDTH = 0.7
DEVICE = torch.device("cpu")





# Explicit deterministic checkpoint paths (no glob / no auto-pick)
ANN_CHECKPOINT =  f"{PTH_DIR}BASE_ANN_TNN_20may_Backup/BASE_ANN_mnist_lr0.0001_ep100_seed42_20260522.pth"
TNN_CHECKPOINT =f"{PTH_DIR}BASE_ANN_TNN_20may_Backup/TERNARY_ONLY_mnist_tw0.7_th0.05_seed42_20260522.pth"




#or
#ANN_CHECKPOINT =  f"{PTH_DIR}BASE_ANN_mnist_lr0.0001_ep100_seed42_20260525.pth"
#TNN_CHECKPOINT =f"{PTH_DIR}TERNARY_ONLY_mnist_tw0.7_th0.05_seed42_20260525.pth"



R_MIN=1.410360e+09  
R_MAX=4.429246e+09 


#=========================================================
'''                    TNN LOAD                '''
#=========================================================

from pathlib import Path
import sys

MAIN_DIR = Path("/Users/apm/Desktop/MASTER THESIS/Projects/04_VOL_Non_Vol/24May")
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))
import importlib
_load = importlib.import_module(f"04_Load_ANN_TNN")
load_ann = _load.load_ann
load_tnn_ternary = _load.load_tnn_ternary
print_model_summary = _load.print_model_summary


# ANN loaded silently — required by unchanged Hardware cost section below
net_ann, ann_path, ann_meta = load_ann(ANN_CHECKPOINT)

net_tnn, tnn_path, tnn_meta = load_tnn_ternary(TNN_CHECKPOINT)
print_model_summary(net_tnn, "TNN (ternary -1/0/+1)", tnn_path, tnn_meta)
'''
TNN (ternary -1/0/+1)
  checkpoint: TERNARY_ONLY_mnist_tw0.7_th0.05_seed42_20260522.pth
  parameters: 235,146
  saved test accuracy: 93.78%
  layers: fc1 (256, 784), fc2 (128, 256), fc3 (10, 128)
  
'''




#=========================================================
'''                  Load Data               '''
#=========================================================


#Load MNIST Test data (same preprocessing as Base_ANN_TNN / ANN_MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))   # flatten 28x28 to 784
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)



#ACcuracy fucntion
def evaluate(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            pred = output.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100 * correct / total






################################################################################
################################################################################
################################################################################
################################################################################
'''    REAL DEVICE (DEVICE-=LEVEL ONLY)  SONOS Vs APM programming error      '''
################################################################################
################################################################################
################################################################################
################################################################################





PROGRAMMING_ERROR_MODELS = ("SONOS", "APM_SINW_SBFET")
PROG_ERROR_SEED = 42  # fixed seed per model build for reproducible comparison



def build_cs_params_ann_real(programming_model: str) -> CrossSimParameters:
    """Bit-sliced ANN CrossSim params (kept for Hardware cost section below)."""
    p = CrossSimParameters()
    p.core.rows_max = 1024
    p.core.cols_max = 1024
    p.core.weight_bits = 8
    p.core.style = 2  # Bitsliced
    p.core.bit_sliced.style = 1  # Balanced per slice
    p.core.bit_sliced.num_slices = 8
    p.xbar.device.Rmin = R_MIN
    p.xbar.device.Rmax = R_MAX
    p.xbar.device.cell_bits = 1
    p.xbar.device.Vread = 1.0
    p.xbar.device.programming_error.enable = True
    p.xbar.device.programming_error.model = programming_model
    p.xbar.device.read_noise.enable = False
    p.xbar.device.drift_error.enable = False
    p.xbar.adc.mvm.bits = 0
    p.xbar.adc.vmm.bits = 0
    p.xbar.dac.mvm.bits = 0
    p.xbar.dac.vmm.bits = 0
    p.xbar.array.parasitics.enable = False
    p.xbar.array.parasitics.Rp_row = 0
    p.xbar.array.parasitics.Rp_col = 0
    p.xbar.array.parasitics.Rp_row_terminal = 0
    p.xbar.array.parasitics.Rp_col_terminal = 0
    p.xbar.array.parasitics.current_from_input = True
    p.xbar.array.parasitics.selected_rows = "top"
    return p




def build_cs_params_tnn_real(programming_model: str) -> CrossSimParameters:
    """Balanced-core TNN CrossSim params with device programming error enabled."""
    p = CrossSimParameters()
    p.core.rows_max = 1024
    p.core.cols_max = 1024
    p.core.style = 1  # Balanced
    p.core.balanced.style = 1
    p.core.balanced.interleaved_posneg = False
    p.core.balanced.subtract_current_in_xbar = True
    p.core.weight_bits = 0
    p.core.bit_sliced.num_slices = 1
    p.xbar.device.Rmin = R_MIN
    p.xbar.device.Rmax = R_MAX
    p.xbar.device.cell_bits = 1
    p.xbar.device.Vread = 1.0
    p.xbar.device.programming_error.enable = True
    p.xbar.device.programming_error.model = programming_model
    p.xbar.device.read_noise.enable = False
    p.xbar.device.drift_error.enable = False
    p.xbar.adc.mvm.bits = 0
    p.xbar.adc.vmm.bits = 0
    p.xbar.dac.mvm.bits = 0
    p.xbar.dac.vmm.bits = 0
    p.xbar.array.parasitics.enable = False
    p.xbar.array.parasitics.Rp_row = 0
    p.xbar.array.parasitics.Rp_col = 0
    p.xbar.array.parasitics.Rp_row_terminal = 0
    p.xbar.array.parasitics.Rp_col_terminal = 0
    p.xbar.array.parasitics.current_from_input = True
    p.xbar.array.parasitics.selected_rows = "top"
    return p


def _build_cs_params_ann_ideal() -> CrossSimParameters:
    """Ideal bit-sliced ANN params (kept for Hardware cost section below)."""
    p = CrossSimParameters()
    p.core.rows_max = 1024
    p.core.cols_max = 1024
    p.core.weight_bits = 8
    p.core.style = 2
    p.core.bit_sliced.style = 1
    p.core.bit_sliced.num_slices = 8
    p.xbar.device.Rmin = R_MIN
    p.xbar.device.Rmax = R_MAX
    p.xbar.device.cell_bits = 1
    p.xbar.device.Vread = 1.0
    p.xbar.device.programming_error.enable = False
    p.xbar.device.read_noise.enable = False
    p.xbar.device.drift_error.enable = False
    p.xbar.adc.mvm.bits = 0
    p.xbar.adc.vmm.bits = 0
    p.xbar.dac.mvm.bits = 0
    p.xbar.dac.vmm.bits = 0
    p.xbar.array.parasitics.enable = False
    p.xbar.array.parasitics.Rp_row = 0
    p.xbar.array.parasitics.Rp_col = 0
    p.xbar.array.parasitics.Rp_row_terminal = 0
    p.xbar.array.parasitics.Rp_col_terminal = 0
    p.xbar.array.parasitics.current_from_input = True
    p.xbar.array.parasitics.selected_rows = "top"
    return p


# ANN CrossSim baselines (not analyzed here; consumed by Hardware cost section)
convertible_modules(net_ann)
digital_real_ann_acc = evaluate(net_ann, test_loader, device="cpu")
_ann_prog_accuracy: dict[str, float] = {"Digital": digital_real_ann_acc}
for prog_model in PROGRAMMING_ERROR_MODELS:
    np.random.seed(PROG_ERROR_SEED)
    cs_ann = build_cs_params_ann_real(prog_model)
    analog_ann = from_torch(net_ann, cs_ann)
    analog_ann.eval()
    _ann_prog_accuracy[prog_model] = evaluate(analog_ann, test_loader, device="cpu")

cs_params_sliced = _build_cs_params_ann_ideal()
analog_ideal_ann_net = from_torch(net_ann, cs_params_sliced)
analog_ideal_ann_net.eval()


convertible_modules(net_tnn)
digital_real_tnn_acc = evaluate(net_tnn, test_loader, device="cpu")

prog_error_nets: dict[str, dict[str, object]] = {}
prog_error_accuracy: dict[str, dict[str, float]] = {
    "ANN": _ann_prog_accuracy,
    "TNN": {"Digital": digital_real_tnn_acc},
}

for prog_model in PROGRAMMING_ERROR_MODELS:
    np.random.seed(PROG_ERROR_SEED)

    cs_tnn = build_cs_params_tnn_real(prog_model)
    analog_tnn = from_torch(net_tnn, cs_tnn)
    analog_tnn.eval()
    tnn_acc = evaluate(analog_tnn, test_loader, device="cpu")

    prog_error_nets[prog_model] = {"tnn": analog_tnn}
    prog_error_accuracy["TNN"][prog_model] = tnn_acc

    print(f"\n=== Programming error model: {prog_model} ===")
    print(f"TNN  — Digital: {digital_real_tnn_acc:.2f}%  |  CrossSim: {tnn_acc:.2f}%")

'''
=== Programming error model: SONOS ===
TNN  — Digital: 93.78%  |  CrossSim: 91.05%

=== Programming error model: APM_SINW_SBFET ===
TNN  — Digital: 93.78%  |  CrossSim: 92.56%


'''

# Backward-compatible handles (last entry = APM_SINW_SBFET)
cs_params_real_ternary = build_cs_params_tnn_real(PROGRAMMING_ERROR_MODELS[-1])
analog_real_tnn_net = prog_error_nets[PROGRAMMING_ERROR_MODELS[-1]]["tnn"]
analog_real_tnn_acc = prog_error_accuracy["TNN"][PROGRAMMING_ERROR_MODELS[-1]]
digital_ideal_tnn_acc = digital_real_tnn_acc




# Programming-error model comparison: SONOS vs APM_SINW_SBFET (TNN only)


_COLOR_SONOS = "#2C5282"
_COLOR_APM = "#C05621"
_COLOR_DIGITAL_BAR = "#4A5568"
_PROG_MODEL_COLORS = {
    "SONOS": _COLOR_SONOS,
    "APM_SINW_SBFET": _COLOR_APM,
}
_PROG_MODEL_LABELS = {
    "SONOS": "SONOS",
    "APM_SINW_SBFET": "APM SiNW SBFET",
}
_LAYER_NAMES = ("fc1", "fc2", "fc3")



def collect_balanced_analog_layers(model):
    layers = []

    for name, module in model.named_modules():
        core_obj = None

        if hasattr(module, "core"):
            core_obj = module.core

        elif hasattr(module, "analog_core"):
            core_obj = module.analog_core

        if core_obj is None:
            continue

        try:
            wrapper_core = core_obj.cores[0][0]

            if hasattr(wrapper_core, "core_pos") and hasattr(wrapper_core, "core_neg"):
                layers.append({
                    "name": name,
                    "analog_core": core_obj,
                    "wrapper_core": wrapper_core
                })

        except Exception:
            pass

    return layers



def collect_analog_layers(model):
    layers = []

    for name, module in model.named_modules():
        analog_w = None

        if hasattr(module, "core"):
            try:
                analog_w = module.core.get_matrix()
            except Exception:
                pass

        if analog_w is None and hasattr(module, "analog_core"):
            try:
                analog_w = module.analog_core.get_matrix()
            except Exception:
                pass

        if analog_w is not None:
            analog_w = np.squeeze(np.array(analog_w))

            if analog_w.ndim >= 2:
                layers.append({
                    "name": name,
                    "weight": analog_w
                })

    return layers




def _save_prog_fig(fig, stem: str, show: bool = True) -> None:
    if show:
        plt.show()
    else:
        plt.close(fig)


def _align_flat(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    n = min(a.size, b.size)
    return a[:n], b[:n]


def _layer_prog_stats(real_layers, ref_layers) -> list[dict]:
    """Per-layer statistics of programmed weights vs a reference (ideal or other model)."""
    stats = []
    n = min(len(real_layers), len(ref_layers))
    for i in range(n):
        w_real, w_ref = _align_flat(real_layers[i]["weight"], ref_layers[i]["weight"])
        delta = w_real - w_ref
        denom = np.maximum(np.abs(w_ref), 1e-12)
        stats.append({
            "layer_idx": i + 1,
            "layer_name": _LAYER_NAMES[i] if i < len(_LAYER_NAMES) else real_layers[i]["name"],
            "delta": delta,
            "rel_abs": np.abs(delta) / denom,
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "corr": float(np.corrcoef(w_real, w_ref)[0, 1]),
        })
    return stats


def _collect_tnn_balanced_cores(model, normalize_by_Rmin: bool = True) -> list[dict]:
    """Per-layer positive / negative / effective conductance for balanced TNN."""
    rows = []
    for layer_idx, li in enumerate(collect_balanced_analog_layers(model)):
        wc = li["wrapper_core"]
        scale = wc.params.xbar.device.Rmin if normalize_by_Rmin else 1.0
        g_pos = np.array(wc.core_pos.matrix, dtype=np.float64) / scale
        g_neg = np.array(wc.core_neg.matrix, dtype=np.float64) / scale
        rows.append({
            "layer_idx": layer_idx + 1,
            "g_pos": g_pos,
            "g_neg": g_neg,
            "g_eff": g_pos - g_neg,
        })
    return rows



cs_params_ternary = CrossSimParameters()
cs_params_ternary.core.rows_max = 1024
cs_params_ternary.core.cols_max = 1024
cs_params_ternary.core.style = 1  # CoreStyle.BALANCED
cs_params_ternary.core.balanced.style = 1
cs_params_ternary.core.balanced.interleaved_posneg = False
cs_params_ternary.core.balanced.subtract_current_in_xbar = True
cs_params_ternary.core.weight_bits = 0
cs_params_ternary.core.bit_sliced.num_slices = 1
cs_params_ternary.xbar.device.Rmin = R_MIN  # LRS / ON
cs_params_ternary.xbar.device.Rmax = R_MAX  # HRS / OFF
cs_params_ternary.xbar.device.cell_bits = 1
cs_params_ternary.xbar.device.Vread = 1.0
cs_params_ternary.xbar.device.programming_error.enable = False
cs_params_ternary.xbar.device.read_noise.enable = False
cs_params_ternary.xbar.device.drift_error.enable = False
cs_params_ternary.xbar.adc.mvm.bits = 0
cs_params_ternary.xbar.adc.vmm.bits = 0
cs_params_ternary.xbar.dac.mvm.bits = 0
cs_params_ternary.xbar.dac.vmm.bits = 0
cs_params_ternary.xbar.array.parasitics.enable = False
cs_params_ternary.xbar.array.parasitics.Rp_row = 0
cs_params_ternary.xbar.array.parasitics.Rp_col = 0
cs_params_ternary.xbar.array.parasitics.Rp_row_terminal = 0
cs_params_ternary.xbar.array.parasitics.Rp_col_terminal = 0
cs_params_ternary.xbar.array.parasitics.current_from_input = True
cs_params_ternary.xbar.array.parasitics.selected_rows = "top"

#convert pytorch to crossim analog model
analog_ideal_tnn_net = from_torch(net_tnn, cs_params_ternary)
analog_ideal_tnn_net.eval()














# --- Reference layers (ideal CrossSim, no programming error) ---
ideal_tnn_ref_layers = collect_analog_layers(analog_ideal_tnn_net)
ideal_tnn_balanced = _collect_tnn_balanced_cores(analog_ideal_tnn_net)

prog_layers = {}
prog_vs_ideal = {}
prog_tnn_balanced = {}

for model_name in PROGRAMMING_ERROR_MODELS:
    tnn_net = prog_error_nets[model_name]["tnn"]
    prog_layers[model_name] = {
        "tnn": collect_analog_layers(tnn_net),
    }
    prog_vs_ideal[model_name] = {
        "tnn": _layer_prog_stats(prog_layers[model_name]["tnn"], ideal_tnn_ref_layers),
    }
    prog_tnn_balanced[model_name] = _collect_tnn_balanced_cores(tnn_net)

sonos_key, apm_key = PROGRAMMING_ERROR_MODELS

# Cross-model layer stats (SONOS programmed weights vs APM programmed weights)
cross_model_tnn = _layer_prog_stats(
    prog_layers[sonos_key]["tnn"], prog_layers[apm_key]["tnn"]
)

print("\n" + "=" * 72)
print("Programming-error comparison summary (TNN vs ideal CrossSim mapping)")
print("=" * 72)
print("\nTNN:")
for model_name in PROGRAMMING_ERROR_MODELS:
    acc = prog_error_accuracy["TNN"][model_name]
    drop = prog_error_accuracy["TNN"]["Digital"] - acc
    print(f"  {_PROG_MODEL_LABELS[model_name]:16s}  acc={acc:5.2f}%  Δdigital={drop:+.2f} pp")
    for st in prog_vs_ideal[model_name]["tnn"]:
        print(
            f"    {st['layer_name']:4s}  MAE={st['mae']:.4e}  "
            f"RMSE={st['rmse']:.4e}  r(ideal)={st['corr']:.4f}"
        )

print("\nCross-model effective-weight correlation (SONOS vs APM):")
for st in cross_model_tnn:
    print(f"  TNN {st['layer_name']}: r={st['corr']:.4f}  MAE={st['mae']:.4e}")
        
        


'''
========================================================================
Programming-error comparison summary (TNN vs ideal CrossSim mapping)
========================================================================

TNN:
  SONOS             acc=91.05%  Δdigital=+2.73 pp
    fc1   MAE=7.0239e-02  RMSE=9.4222e-02  r(ideal)=0.9687
    fc2   MAE=7.8196e-02  RMSE=1.0671e-01  r(ideal)=0.9760
    fc3   MAE=1.0574e-01  RMSE=1.4184e-01  r(ideal)=0.9756
  APM SiNW SBFET    acc=92.56%  Δdigital=+1.22 pp
    fc1   MAE=3.0888e-02  RMSE=3.8551e-02  r(ideal)=0.9959
    fc2   MAE=3.2878e-02  RMSE=4.0907e-02  r(ideal)=0.9975
    fc3   MAE=4.0606e-02  RMSE=4.9058e-02  r(ideal)=0.9987

Cross-model effective-weight correlation (SONOS vs APM):
  TNN fc1: r=0.9646  MAE=7.6762e-02
  TNN fc2: r=0.9737  MAE=8.4361e-02
  TNN fc3: r=0.9744  MAE=1.1451e-01
  
'''





_ACADEMIC_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "normal",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

# --- Figure 1: MNIST test accuracy (TNN) ---
with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    x = np.arange(len(PROGRAMMING_ERROR_MODELS))

    digital_acc = prog_error_accuracy["TNN"]["Digital"]
    accs = [prog_error_accuracy["TNN"][m] for m in PROGRAMMING_ERROR_MODELS]
    colors = [_PROG_MODEL_COLORS[m] for m in PROGRAMMING_ERROR_MODELS]

    ax.axhline(
        digital_acc, color=_COLOR_DIGITAL_BAR, linestyle="--", linewidth=1.2,
        label=f"Digital ({digital_acc:.2f}%)", zorder=1,
    )
    bars = ax.bar(
        x, accs, width=0.55, color=colors,
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    ax.set_xticks(x, [_PROG_MODEL_LABELS[m] for m in PROGRAMMING_ERROR_MODELS])
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("TNN: Programming-Error Model Comparison")
    y0 = min(digital_acc, min(accs)) - 1.5
    y1 = max(digital_acc, max(accs)) + 0.8
    ax.set_ylim(y0, y1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, acc + 0.05,
            f"{acc:.2f}%", ha="center", va="bottom", fontsize=9,
        )

    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "SONOS vs APM SiNW SBFET — TNN MNIST Inference Accuracy\n"
        "(programming error only; read noise and drift disabled)",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    _save_prog_fig(fig, "fig01_tnn_accuracy_sonos_vs_apm")


# --- Figure 2: Accuracy drop from digital baseline (TNN) ---
with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    x = np.arange(1)
    width = 0.35

    sonos_drop = prog_error_accuracy["TNN"]["Digital"] - prog_error_accuracy["TNN"][sonos_key]
    apm_drop = prog_error_accuracy["TNN"]["Digital"] - prog_error_accuracy["TNN"][apm_key]

    ax.bar(x - width / 2, [sonos_drop], width, label=_PROG_MODEL_LABELS[sonos_key],
           color=_COLOR_SONOS, edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, [apm_drop], width, label=_PROG_MODEL_LABELS[apm_key],
           color=_COLOR_APM, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, ["TNN"])
    ax.set_ylabel("Accuracy Drop from Digital (pp)")
    ax.set_title("TNN: Impact of Programming Error on MNIST Accuracy")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    ax.text(x[0] - width / 2, sonos_drop + 0.02, f"{sonos_drop:.2f}", ha="center", fontsize=9)
    ax.text(x[0] + width / 2, apm_drop + 0.02, f"{apm_drop:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    _save_prog_fig(fig, "fig02_tnn_accuracy_drop_pp")


# --- Figure 3: SONOS vs APM effective-weight scatter (TNN, per layer) ---
def _plot_crossmodel_scatter(
    stats_list, net_label: str, xlabel_model: str, ylabel_model: str, stem: str,
) -> None:
    n = len(stats_list)
    with plt.rc_context(_ACADEMIC_RC):
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
        axes = axes[0]
        for ax, st in zip(axes, stats_list):
            w_x, w_y = _align_flat(
                prog_layers[sonos_key][net_label.lower()][st["layer_idx"] - 1]["weight"],
                prog_layers[apm_key][net_label.lower()][st["layer_idx"] - 1]["weight"],
            )
            ax.scatter(
                w_x, w_y, s=3, alpha=0.25, color=_COLOR_SONOS,
                edgecolors="none", rasterized=True,
            )
            lims = [
                min(w_x.min(), w_y.min()), max(w_x.max(), w_y.max()),
            ]
            ax.plot(lims, lims, "k--", linewidth=0.9, alpha=0.7)
            ax.set_xlabel(f"{xlabel_model} weight")
            ax.set_ylabel(f"{ylabel_model} weight")
            ax.set_title(f"{st['layer_name']}\n$r$={st['corr']:.4f}, MAE={st['mae']:.2e}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.4, linestyle="--", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle(
            f"{net_label}: Programmed Effective Weights — "
            f"{_PROG_MODEL_LABELS[sonos_key]} vs {_PROG_MODEL_LABELS[apm_key]}",
            y=1.02,
        )
        fig.tight_layout()
        _save_prog_fig(fig, stem)


_plot_crossmodel_scatter(
    cross_model_tnn, "TNN", _PROG_MODEL_LABELS[sonos_key],
    _PROG_MODEL_LABELS[apm_key], "fig03_tnn_crossmodel_weight_scatter",
)


# --- Figure 4: Programming displacement vs ideal mapping (TNN boxplots) ---
def _plot_prog_displacement_boxplots(vs_ideal_dict, net_label: str, stem: str) -> None:
    n_layers = len(vs_ideal_dict[sonos_key])
    with plt.rc_context(_ACADEMIC_RC):
        fig, axes = plt.subplots(1, n_layers, figsize=(4.0 * n_layers, 4.5), sharey=True)
        if n_layers == 1:
            axes = [axes]
        for ax, layer_i in zip(axes, range(n_layers)):
            data, labels, colors = [], [], []
            for model_name in PROGRAMMING_ERROR_MODELS:
                rel = vs_ideal_dict[model_name][layer_i]["rel_abs"]
                data.append(rel[np.isfinite(rel)])
                labels.append(_PROG_MODEL_LABELS[model_name])
                colors.append(_PROG_MODEL_COLORS[model_name])
            bp = ax.boxplot(
                data, tick_labels=labels, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.2),
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
            ax.set_yscale("log")
            ax.set_title(_LAYER_NAMES[layer_i])
            ax.set_xlabel("Programming-error model")
            ax.grid(axis="y", alpha=0.4, linestyle="--", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(r"$|G_\mathrm{prog} - G_\mathrm{ideal}| / |G_\mathrm{ideal}|$")
        fig.suptitle(
            f"{net_label}: Relative Programming Displacement vs Ideal CrossSim",
            y=1.02,
        )
        fig.tight_layout()
        _save_prog_fig(fig, stem)


_plot_prog_displacement_boxplots(
    {m: prog_vs_ideal[m]["tnn"] for m in PROGRAMMING_ERROR_MODELS},
    "TNN",
    "fig04_tnn_prog_displacement_vs_ideal",
)


# --- Figure 5: TNN positive / negative core MAE vs ideal ---
def _tnn_core_mae_bars() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_labels, mae_pos_s, mae_neg_s, mae_pos_a, mae_neg_a = [], [], [], [], []
    for i, (row_s, row_a, row_i) in enumerate(
        zip(prog_tnn_balanced[sonos_key], prog_tnn_balanced[apm_key], ideal_tnn_balanced)
    ):
        x_labels.append(_LAYER_NAMES[i])
        gp_s, gp_i = _align_flat(row_s["g_pos"], row_i["g_pos"])
        gn_s, gn_i = _align_flat(row_s["g_neg"], row_i["g_neg"])
        gp_a, _ = _align_flat(row_a["g_pos"], row_i["g_pos"])
        gn_a, _ = _align_flat(row_a["g_neg"], row_i["g_neg"])
        mae_pos_s.append(np.mean(np.abs(gp_s - gp_i)))
        mae_neg_s.append(np.mean(np.abs(gn_s - gn_i)))
        mae_pos_a.append(np.mean(np.abs(gp_a - gp_i)))
        mae_neg_a.append(np.mean(np.abs(gn_a - gn_i)))
    return (
        x_labels,
        np.array(mae_pos_s), np.array(mae_neg_s),
        np.array(mae_pos_a), np.array(mae_neg_a),
    )


x_labels, mae_pos_s, mae_neg_s, mae_pos_a, mae_neg_a = _tnn_core_mae_bars()
x = np.arange(len(x_labels))
width = 0.18

with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 1.5 * width, mae_pos_s, width, label=f"{_PROG_MODEL_LABELS[sonos_key]} (+)",
           color=_COLOR_SONOS, alpha=0.9, edgecolor="white")
    ax.bar(x - 0.5 * width, mae_neg_s, width, label=f"{_PROG_MODEL_LABELS[sonos_key]} (−)",
           color=_COLOR_SONOS, alpha=0.45, edgecolor="white", hatch="//")
    ax.bar(x + 0.5 * width, mae_pos_a, width, label=f"{_PROG_MODEL_LABELS[apm_key]} (+)",
           color=_COLOR_APM, alpha=0.9, edgecolor="white")
    ax.bar(x + 1.5 * width, mae_neg_a, width, label=f"{_PROG_MODEL_LABELS[apm_key]} (−)",
           color=_COLOR_APM, alpha=0.45, edgecolor="white", hatch="//")
    ax.set_xticks(x, x_labels)
    ax.set_ylabel(r"MAE vs ideal ($G/R_\mathrm{min}$)")
    ax.set_xlabel("Layer")
    ax.set_title("TNN Balanced Core: Programming Error on Positive and Negative Arrays")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_prog_fig(fig, "fig05_tnn_posneg_mae_vs_ideal")


# --- Figure 6: Cross-model correlation per layer (TNN) ---
with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    layers = [st["layer_name"] for st in cross_model_tnn]
    corrs = [st["corr"] for st in cross_model_tnn]
    x = np.arange(len(layers))
    ax.bar(x, corrs, color=_COLOR_SONOS, edgecolor="white", linewidth=0.8, width=0.55)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x, layers)
    ax.set_ylim(0.70, 1.01)
    ax.set_title("TNN")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Pearson $r$ (SONOS vs APM weights)")
    for i, c in enumerate(corrs):
        ax.text(i, c + 0.00015, f"{c:.4f}", ha="center", fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("TNN Cross-Model Weight Correlation by Layer", y=1.02)
    fig.tight_layout()
    _save_prog_fig(fig, "fig06_tnn_crossmodel_correlation_by_layer")


# --- Figure 7: Pooled cross-model programming-error distribution (TNN) ---
pooled_delta_tnn = np.concatenate([st["delta"] for st in cross_model_tnn])

with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(
        pooled_delta_tnn, bins=80, density=True, alpha=0.75,
        color=_COLOR_SONOS, edgecolor="white", linewidth=0.4,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\Delta w = w_\mathrm{APM} - w_\mathrm{SONOS}$")
    ax.set_ylabel("Density")
    ax.set_title("TNN")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.4, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.97, 0.95,
        f"MAE={np.mean(np.abs(pooled_delta_tnn)):.2e}\nstd={pooled_delta_tnn.std():.2e}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#CCC"),
    )
    fig.suptitle(
        "TNN: Distribution of Programmed Weight Differences (APM − SONOS)",
        y=1.02,
    )
    fig.tight_layout()
    _save_prog_fig(fig, "fig07_tnn_crossmodel_delta_histogram")























################################################################################
################################################################################
################################################################################
################################################################################
'''    Hardware cost (ANN vs TNN) strategy comaprison     '''
################################################################################
################################################################################
################################################################################
################################################################################




#-------------------------------------------------------------------------------
# Hardware cost: ANN (bit-sliced) vs TNN (ternary balanced)
# Memristor / crossbar counts from actual layer weight shapes

ANN_NUM_SLICES=8

HW_ANN_SLICES = ANN_NUM_SLICES  # 8-bit bit-sliced → 8 slices
HW_BALANCED_SIDES = 2           # G_pos and G_neg per balanced slice/array
HW_LAYER_NAMES = ("fc1", "fc2", "fc3")


def _save_hw_fig(fig, stem: str) -> None:
    #fig.savefig(FIGURES_HW_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    #fig.savefig(FIGURES_HW_DIR / f"{stem}.png", bbox_inches="tight", dpi=FIGURE_DPI)
    plt.show()


def _linear_weight_shape(model, layer_name: str) -> tuple[int, int]:
    w = getattr(model, layer_name).weight
    return int(w.shape[0]), int(w.shape[1])  # (out_features, in_features)


def _count_ann_hardware(out: int, inp: int) -> dict:
    """Bit-sliced balanced: N_slices × (G_pos + G_neg), one device per matrix entry each."""
    synapses = out * inp
    crossbars = HW_ANN_SLICES * HW_BALANCED_SIDES
    memristors = synapses * crossbars
    adcs_per_layer = HW_ANN_SLICES
    return {
        "out": out,
        "in": inp,
        "synapses": synapses,
        "crossbars": crossbars,
        "memristors": memristors,
        "adcs": adcs_per_layer,
    }


def _count_tnn_hardware(out: int, inp: int, weight: np.ndarray) -> dict:
    """Balanced ternary: G_pos + G_neg; report non-zero ternary occupancy."""
    synapses = out * inp
    crossbars = HW_BALANCED_SIDES
    memristors = synapses * crossbars
    w_flat = weight.ravel()
    n_total = w_flat.size
    n_pos = int((w_flat > 0).sum())
    n_neg = int((w_flat < 0).sum())
    n_zero = int((w_flat == 0).sum())
    return {
        "out": out,
        "in": inp,
        "synapses": synapses,
        "crossbars": crossbars,
        "memristors": memristors,
        "adcs": 1,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "n_total": n_total,
        "pct_nonzero": 100.0 * (n_pos + n_neg) / n_total if n_total else 0.0,
    }


def collect_layer_info(model) -> list[dict]:
    """Extract weight/bias arrays per Linear layer."""
    layer_info = []
    for name in ("fc1", "fc2", "fc3"):
        layer = getattr(model, name)
        w = layer.weight.detach().cpu().numpy()
        b = layer.bias.detach().cpu().numpy()
        layer_info.append({
            "name": name,
            "weight": w,
            "bias": b,
            "weight_shape": w.shape,
        })
    return layer_info


ann_hw_layers = []
tnn_hw_layers = []
ann_layer_info = collect_layer_info(net_ann)
tnn_layer_info = collect_layer_info(net_tnn)

for name, ann_li, tnn_li in zip(HW_LAYER_NAMES, ann_layer_info, tnn_layer_info):
    out_a, in_a = _linear_weight_shape(net_ann, name)
    out_t, in_t = _linear_weight_shape(net_tnn, name)
    ann_hw_layers.append({"name": name, **_count_ann_hardware(out_a, in_a)})
    tnn_hw_layers.append({
        "name": name,
        **_count_tnn_hardware(out_t, in_t, tnn_li["weight"]),
    })

ann_totals = {k: sum(L[k] for L in ann_hw_layers)
              for k in ("synapses", "crossbars", "memristors", "adcs")}
tnn_totals = {k: sum(L[k] for L in tnn_hw_layers)
              for k in ("synapses", "crossbars", "memristors", "adcs")}

print("\n" + "=" * 78)
print("Hardware cost — ANN (8-bit bit-sliced balanced) vs TNN (ternary balanced)")
print("=" * 78)
print(
    f"{'Layer':<6} {'Shape':>12}  "
    f"{'ANN memristors':>16}  {'TNN memristors':>16}  {'Ratio ANN/TNN':>14}"
)
print("-" * 78)
for a, t in zip(ann_hw_layers, tnn_hw_layers):
    shape = f"{a['out']}×{a['in']}"
    ratio = a["memristors"] / t["memristors"] if t["memristors"] else float("inf")
    print(
        f"{a['name']:<6} {shape:>12}  "
        f"{a['memristors']:>16,}  {t['memristors']:>16,}  {ratio:>14.1f}×"
    )
print("-" * 78)
total_ratio = ann_totals["memristors"] / tnn_totals["memristors"]
print(
    f"{'TOTAL':<6} {'':<12}  "
    f"{ann_totals['memristors']:>16,}  {tnn_totals['memristors']:>16,}  {total_ratio:>14.1f}×"
)
print(
    f"\nCrossbar sub-arrays (pos/neg planes): ANN {ann_totals['crossbars']}  |  "
    f"TNN {tnn_totals['crossbars']}  (ratio {ann_totals['crossbars'] / tnn_totals['crossbars']:.1f}×)"
)
print(
    f"ADC blocks (per layer, 8b RampADC model): ANN {ann_totals['adcs']}  |  "
    f"TNN {tnn_totals['adcs']}"
)
print("\nTNN ternary weight occupancy (from checkpoint weights):")
for t in tnn_hw_layers:
    print(
        f"  {t['name']} ({t['out']}×{t['in']}):  "
        f"+1: {t['n_pos']:,}  -1: {t['n_neg']:,}  0: {t['n_zero']:,}  "
        f"({t['pct_nonzero']:.1f}% non-zero)"
    )
print(
    f"\nEncoding: ANN uses {HW_ANN_SLICES} bit slices × {HW_BALANCED_SIDES} "
    f"(G+/G−) = {HW_ANN_SLICES * HW_BALANCED_SIDES}× devices per synapse; "
    f"TNN uses {HW_BALANCED_SIDES}× (G+/G−) per synapse."
)


'''
==============================================================================
Hardware cost — ANN (8-bit bit-sliced balanced) vs TNN (ternary balanced)
==============================================================================
Layer         Shape    ANN memristors    TNN memristors   Ratio ANN/TNN
------------------------------------------------------------------------------
fc1         256×784         3,211,264           401,408             8.0×
fc2         128×256           524,288            65,536             8.0×
fc3          10×128            20,480             2,560             8.0×
------------------------------------------------------------------------------
TOTAL                       3,756,032           469,504             8.0×

Crossbar sub-arrays (pos/neg planes): ANN 48  |  TNN 6  (ratio 8.0×)
ADC blocks (per layer, 8b RampADC model): ANN 24  |  TNN 3

TNN ternary weight occupancy (from checkpoint weights):
  fc1 (256×784):  +1: 16,621  -1: 10,772  0: 173,311  (13.6% non-zero)
  fc2 (128×256):  +1: 2,350  -1: 5,365  0: 25,053  (23.5% non-zero)
  fc3 (10×128):  +1: 102  -1: 700  0: 478  (62.7% non-zero)

Encoding: ANN uses 8 bit slices × 2 (G+/G−) = 16× devices per synapse; TNN uses 2× (G+/G−) per synapse.

'''







layer_labels = [L["name"] for L in ann_hw_layers]
x = np.arange(len(layer_labels))
ann_mem = [L["memristors"] for L in ann_hw_layers]
tnn_mem = [L["memristors"] for L in tnn_hw_layers]
ann_xbars = [L["crossbars"] for L in ann_hw_layers]
tnn_xbars = [L["crossbars"] for L in tnn_hw_layers]


_COLOR_ANN_STRAT = "#2C5282"
_COLOR_TNN_STRAT = "#C05621"





# --- Figure 0: Mapping strategy comparison (Digital vs Ideal CrossSim) ---
def plot_mapping_strategy_comparison():
    
    ann_digital = prog_error_accuracy["ANN"]["Digital"]
    tnn_digital = prog_error_accuracy["TNN"]["Digital"]

    ann_ideal = evaluate(analog_ideal_ann_net, test_loader, device="cpu")
    tnn_ideal = evaluate(analog_ideal_tnn_net, test_loader, device="cpu")

    groups = [
        "ANN\n(8-bit bit-sliced)",
        "TNN\n(ternary balanced)"
    ]

    digital_vals = [ann_digital, tnn_digital]
    ideal_vals   = [ann_ideal, tnn_ideal]

    x = np.arange(len(groups))
    width = 0.32

    COLOR_ANN_DIGITAL = _COLOR_ANN_STRAT          # dark blue
    COLOR_ANN_IDEAL   = "#3c60a6"                 # light blue
    COLOR_TNN_DIGITAL = _COLOR_TNN_STRAT          # dark orange
    COLOR_TNN_IDEAL   = "#cc6d31"                 # light orange

    bar_colors_digital = [COLOR_ANN_DIGITAL, COLOR_TNN_DIGITAL]
    bar_colors_ideal = [COLOR_ANN_IDEAL, COLOR_TNN_IDEAL]

    with plt.rc_context(_ACADEMIC_RC):

        fig, ax = plt.subplots(figsize=(10, 5.6))

        bars1 = ax.bar(
            x - width / 2,
            digital_vals,
            width,
            color=bar_colors_digital,
            edgecolor="white",
            linewidth=0.8,
            label="Digital PyTorch",
        )

        bars2 = ax.bar(
            x + width / 2,
            ideal_vals,
            width,
            color=bar_colors_ideal,
            edgecolor="white",
            linewidth=0.8,
            label="Ideal CrossSim",
        )

        ax.set_xticks(x, groups)

        ax.set_ylabel("MNIST Test Accuracy (%)")

        ax.set_title(
            "Mapping Strategy Comparison\n"
            "ANN (bit-sliced) vs TNN (ternary balanced)"
        )

        y_min = min(digital_vals + ideal_vals) - 4
        y_max = max(digital_vals + ideal_vals) + 1

        ax.set_ylim(y_min, y_max)

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.6,
            alpha=0.4,
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        def _annotate(bars):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    h + 0.12,
                    f"{h:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        _annotate(bars1)
        _annotate(bars2)

        from matplotlib.patches import Patch
        legend = ax.legend(
            handles=[
                Patch(facecolor=COLOR_ANN_DIGITAL, edgecolor="white", label="ANN — Digital"),
                Patch(facecolor=COLOR_ANN_IDEAL, edgecolor="white", label="ANN — Ideal CrossSim"),
                Patch(facecolor=COLOR_TNN_DIGITAL, edgecolor="white", label="TNN — Digital"),
                Patch(facecolor=COLOR_TNN_IDEAL, edgecolor="white", label="TNN — Ideal CrossSim"),
            ],
            loc="upper right",
            fontsize=9,
        )
        legend.get_frame().set_alpha(0.95)

        fig.tight_layout()

        _save_prog_fig(fig, "fig00_mapping_strategy_comparison")


plot_mapping_strategy_comparison()

width = 0.35




# --- Figure HW-2: Crossbar sub-arrays per layer ---
with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, ann_xbars, width, label="ANN", color=_COLOR_ANN_STRAT,
           edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, tnn_xbars, width, label="TNN", color=_COLOR_TNN_STRAT,
           edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, layer_labels)
    ax.set_ylabel("Crossbar sub-arrays (G$_+$ or G$_-$ planes)")
    ax.set_xlabel("Layer")
    ax.set_title("Number of Crossbar Arrays to Fabricate per Layer")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_hw_fig(fig, "fig02_crossbars_per_layer")


# --- Figure HW-3: Memristors per layer ---
with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.35
    ax.bar(x - width / 2, ann_mem, width, label="ANN (8-bit bit-sliced)",
           color=_COLOR_ANN_STRAT, edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, tnn_mem, width, label="TNN (ternary balanced)",
           color=_COLOR_TNN_STRAT, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, layer_labels)
    ax.set_ylabel("Memristor devices (count)")
    ax.set_xlabel("Layer")
    ax.set_title(
        "Physical Memristor Count per Layer\n"
        f"(ANN: {HW_ANN_SLICES} slices × G$_+$/G$_-$; TNN: G$_+$/G$_-$ only)",
    )
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")
    )
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    for i, (am, tm) in enumerate(zip(ann_mem, tnn_mem)):
        ax.text(i - width / 2, am, f"{am/1e6:.2f}M", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, tm, f"{tm/1e3:.0f}k", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save_hw_fig(fig, "fig01_memristors_per_layer")


# --- Figure HW-4: System totals (memristors, crossbars, ADCs) ---
categories = ["Memristors", "Crossbar\nsub-arrays", "ADC\nblocks"]
ann_tot_vals = [ann_totals["memristors"], ann_totals["crossbars"], ann_totals["adcs"]]
tnn_tot_vals = [tnn_totals["memristors"], tnn_totals["crossbars"], tnn_totals["adcs"]]
x_cat = np.arange(len(categories))

with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x_cat - width / 2, ann_tot_vals, width, label="ANN", color=_COLOR_ANN_STRAT,
           edgecolor="white", linewidth=0.8)
    ax.bar(x_cat + width / 2, tnn_tot_vals, width, label="TNN", color=_COLOR_TNN_STRAT,
           edgecolor="white", linewidth=0.8)
    ax.set_xticks(x_cat, categories)
    ax.set_ylabel("Count (entire network)")
    ax.set_title(
        f"Total Hardware Cost — MNIST MLP\n",
    )
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (av, tv) in enumerate(zip(ann_tot_vals, tnn_tot_vals)):
        ax.text(i - width / 2, av * 1.08, f"{av:,}", ha="center", va="bottom", fontsize=8, rotation=15)
        ax.text(i + width / 2, tv * 1.08, f"{tv:,}", ha="center", va="bottom", fontsize=8, rotation=15)
    fig.tight_layout()
    _save_hw_fig(fig, "fig03_total_hardware_log_scale")


# --- Figure HW-5: TNN ternary weight distribution per layer ---
with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    n_pos = [L["n_pos"] for L in tnn_hw_layers]
    n_neg = [L["n_neg"] for L in tnn_hw_layers]
    n_zero = [L["n_zero"] for L in tnn_hw_layers]
    ax.bar(x, n_pos, width=0.55, label="$w = +1$", color="#2C5282", edgecolor="white")
    ax.bar(x, n_neg, width=0.55, bottom=n_pos, label="$w = -1$", color="#C05621", edgecolor="white")
    ax.bar(x, n_zero, width=0.55, bottom=np.array(n_pos) + np.array(n_neg),
           label="$w = 0$", color="#CBD5E0", edgecolor="white")
    ax.set_xticks(x, layer_labels)
    ax.set_ylabel("Weight count")
    ax.set_xlabel("Layer")
    ax.set_title("TNN Checkpoint: Ternary Weight Distribution per Layer")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_hw_fig(fig, "fig04_tnn_ternary_weight_stack")


# --- Figure HW-6: Synapse-level comparison table as figure ---
table_rows = []
for a, t in zip(ann_hw_layers, tnn_hw_layers):
    table_rows.append([
        a["name"],
        f"{a['out']}×{a['in']}",
        f"{a['synapses']:,}",
        f"{a['memristors']:,}",
        f"{t['memristors']:,}",
        f"{a['memristors'] / t['memristors']:.0f}×",
    ])
table_rows.append([
    "Total", "—",
    f"{ann_totals['synapses']:,}",
    f"{ann_totals['memristors']:,}",
    f"{tnn_totals['memristors']:,}",
    f"{total_ratio:.0f}×",
])

with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    col_labels = [
        "Layer", "Shape\n(out×in)", "Synapses",
        "ANN memristors", "TNN memristors", "Ratio",
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    ax.set_title(
        "Hardware Mapping Summary (based on loaded fc1–fc3 weights)",
        pad=20, fontsize=12,
    )
    fig.tight_layout()
    _save_hw_fig(fig, "fig05_hardware_summary_table")







