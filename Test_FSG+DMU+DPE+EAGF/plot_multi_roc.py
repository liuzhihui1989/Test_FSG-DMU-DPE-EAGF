import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def plot_multi_roc(data_dir="roc_data"):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 找所有实验的 fpr/tpr 数据
    fpr_files = sorted(glob.glob(os.path.join(data_dir, "*_fpr.npy")))

    for fpr_path in fpr_files:
        model_name = os.path.basename(fpr_path).replace("_fpr.npy", "")
        tpr_path = f"roc_data/{model_name}_tpr.npy"
        auc_path = f"roc_data/{model_name}_auc.npy"

        fpr = np.load(fpr_path)
        tpr = np.load(tpr_path)
        auc_value = np.load(auc_path)[0]

        ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={auc_value:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.set_title("ROC Curves of Ablation Experiments",fontsize=15)
    ax.legend(loc="lower right")

    # 放大区域主坐标轴数值字体调整
    ax.tick_params(axis='both', labelsize=15)  # 调整主坐标轴的刻度字体大小

    # ===============================
    # 左上角局部放大（向下偏移，不遮挡主图）
    # ===============================
    axins = inset_axes(
        ax,
        width=2.8,  # 英寸，稳定
        height=2.2,  # 英寸
        loc="upper right",
        bbox_to_anchor=(0.95, 0.88),  # 向下移动（关键）
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    # inset 中重复画所有 ROC 曲线
    for fpr_path in fpr_files:
        model_name = os.path.basename(fpr_path).replace("_fpr.npy", "")
        tpr_path = f"roc_data/{model_name}_tpr.npy"

        fpr = np.load(fpr_path)
        tpr = np.load(tpr_path)

        axins.plot(fpr, tpr, lw=2)

    # 放大区域范围（根据你给的示例图）
    axins.set_xlim(0.0, 0.2)   # FPR 小范围
    axins.set_ylim(0.8, 1.0)   # TPR 高范围


    # 移除插入区域的坐标刻度（去掉坐标值）
    axins.set_xticks([])  # 不显示 x 坐标
    axins.set_yticks([])  # 不显示 y 坐标

    axins.tick_params(labelsize=15) #放大区域坐标轴大小

    # 在主图和 inset 之间画虚线框
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="gray", linestyle="--", linewidth=2)

    plt.tight_layout()
    plt.savefig("multi_roc_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_multi_roc()