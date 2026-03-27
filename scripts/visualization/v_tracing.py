import os
import sys
from argparse import ArgumentParser

import torch
import tomlkit
import matplotlib.pyplot as plt

from DTransformer.data import KTData
from DTransformer.model import DTransformer
from DTransformer.visualize import heat_map

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
# 将项目根目录添加到 Python 路径中
sys.path.insert(0, project_root)

DATA_DIR = "data"

# configure the main parser
parser = ArgumentParser()

# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument(
    "-s", "--seq_id", help="select a sequence index", default=0, type=int
)

# data setup
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    required=True,
)
parser.add_argument(
    "-p", "--with_pid", help="provide DTransformer with pid", action="store_true"
)

# DTransformer setup
parser.add_argument("--d_model", help="DTransformer hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default=3)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument(
    "--n_know", help="dimension of knowledge parameter", type=int, default=16

)

# plot setup
parser.add_argument("-f", "--from_file", help="test existing DTransformer file", required=True)

def main(args):
    # prepare datasets
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    test_data = KTData(
        os.path.join(DATA_DIR, dataset["test"]),
        dataset["inputs"],
        seq_len=seq_len,
    )

    # prepare DTransformer
    model = DTransformer(
        dataset["n_questions"],
        dataset["n_pid"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers,
    )

    model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s))
    model.to(args.device)
    model.eval()

    # get one sequence
    data = test_data[args.seq_id]
    q, s, pid = data.get("q", "s", "pid")
    q = q.unsqueeze(0)
    s = s.unsqueeze(0)
    if pid is not None:
        pid = pid.unsqueeze(0)
    with torch.no_grad():
        _, _, _, _, (q_score, k_score) = model.predict(q, s, pid)

    # 打印 k_score 尺寸
    print("k_score size:", k_score.size())

    # question attention heatmap
    heads = [0, 1, 2, 3]
    seq_len = 8  # 根据实际数据调整 seq_len
    fig1, ax = plt.subplots(1, 4, figsize=(12,3), constrained_layout=True)  # 调整子图的尺寸
    # 设置颜色范围映射：使用 vmin 和 vmax 调整颜色范围
    vmin = 0.1  # 调整为你想要的最低值
    vmax = 0.4  # 最高值一般是 1.0
    for i in range(len(heads)):
        im = heat_map(ax[i], q_score[0, heads[i], :seq_len, :seq_len], cmap="hot")

        # 手动设置颜色范围
        im.set_clim(vmin, vmax)

    cbar = plt.colorbar(im, ax=ax.ravel().tolist(), shrink=1)  # 调整颜色条
    cbar.set_label('Attention Score')  # 可选：为颜色条添加标签

    # knowledge attention heatmap on one head
    max_step = k_score.size(2)  # 获取 k_score 的最大步长
    steps = [int(max_step * i / 4) for i in range(4)]  # 动态生成步骤索引
    steps = [min(step, max_step - 1) for step in steps]  # 确保步骤索引不超出范围
    head = heads[0]
    fig2, ax = plt.subplots(4, 1, figsize=(8, 16), constrained_layout=True)  # 增加图像尺寸
    for i in range(len(steps)):
        xticks = None if i == len(steps) - 1 else []
        im = heat_map(ax[i], k_score[0, head, steps[i], :, :max_step], xticks=xticks)
    plt.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6)  # 调整颜色条

    # plt.tight_layout()
    plt.show()
    # fig.savefig('qt.pdf', bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
