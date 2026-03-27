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

    # 对q_score和k_score进行归一化处理
    q_score = (q_score - q_score.min()) / (q_score.max() - q_score.min())
    k_score = (k_score - k_score.min()) / (k_score.max() - k_score.min())

    # question attention heatmap
    heads = [0, 1, 2, 3]
    seq_len = 18
    fig1, ax = plt.subplots(1, 4, figsize=(18, 3))
    for i in range(len(heads)):
        im = heat_map(ax[i], q_score[0, heads[i], :seq_len, :seq_len])
    plt.colorbar(im, ax=ax, location="right")

    # knowledge attention heatmap on one head
    steps = [10, 20, 30, 40]
    head = heads[0]
    fig2, ax = plt.subplots(4, 1, figsize=(6, 6))
    for i in range(len(heads)):
        xticks = None if i == len(heads) - 1 else []
        im = heat_map(ax[i], k_score[0, head, steps[i], :, : steps[-1]], xticks=xticks)
    plt.colorbar(im, ax=ax, location="right")

    plt.show()
    # fig.savefig('qt.pdf', bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

#
# import os
# import sys
# from argparse import ArgumentParser
#
# import torch
# import tomlkit
# import matplotlib.pyplot as plt
#
# from DTransformer.data import KTData
# from DTransformer.model import DTransformer
# from DTransformer.visualize import heat_map
#
# # 获取当前文件的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取项目根目录
# project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
# # 将项目根目录添加到 Python 路径中
# sys.path.insert(0, project_root)
#
# DATA_DIR = "data"
#
# # configure the main parser
# parser = ArgumentParser()
#
# # general options
# parser.add_argument("--device", help="device to run network on", default="cpu")
# parser.add_argument(
#     "-s", "--seq_id", help="select a sequence index", default=0, type=int
# )
#
# # data setup
# datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
# parser.add_argument(
#     "-d",
#     "--dataset",
#     help="choose from a dataset",
#     choices=datasets.keys(),
#     required=True,
# )
# parser.add_argument(
#     "-p", "--with_pid", help="provide DTransformer with pid", action="store_true"
# )
#
# # DTransformer setup
# parser.add_argument("--d_model", help="DTransformer hidden size", type=int, default=128)
# parser.add_argument("--n_layers", help="number of layers", type=int, default=3)
# parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
# parser.add_argument(
#     "--n_know", help="dimension of knowledge parameter", type=int, default=16
# )
#
# # plot setup
# parser.add_argument("-f", "--from_file", help="test existing DTransformer file", required=True)
#
#
# def main(args):
#     # prepare datasets
#     dataset = datasets[args.dataset]
#     seq_len = dataset["seq_len"] if "seq_len" in dataset else None
#     test_data = KTData(
#         os.path.join(DATA_DIR, dataset["test"]),
#         dataset["inputs"],
#         seq_len=seq_len,
#     )
#
#     # prepare DTransformer
#     model = DTransformer(
#         dataset["n_questions"],
#         dataset["n_pid"],
#         d_model=args.d_model,
#         n_heads=args.n_heads,
#         n_know=args.n_know,
#         n_layers=args.n_layers,
#     )
#
#     model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s))
#     model.to(args.device)
#     model.eval()
#
#     # get one sequence
#     data = test_data[args.seq_id]
#     q, s, pid = data.get("q", "s", "pid")
#     q = q.unsqueeze(0)
#     s = s.unsqueeze(0)
#     if pid is not None:
#         pid = pid.unsqueeze(0)
#     with torch.no_grad():
#         _, _, _, _, (q_score, k_score) = model.predict(q, s, pid)
#
#     # 对q_score和k_score进行归一化处理
#     q_score = (q_score - q_score.min()) / (q_score.max() - q_score.min())
#     k_score = (k_score - k_score.min()) / (k_score.max() - k_score.min())
#
#     # question attention heatmap
#     heads = [0, 1, 2, 3]
#     seq_len = 33  # 修改 seq_len 以显示序列后部分
#     start_idx = max(0, seq_len - 18)
#     fig1, ax = plt.subplots(2, 2, figsize=(12, 12))
#     ax = ax.flatten()
#     for i in range(len(heads)):
#         im = heat_map(ax[i], q_score[0, heads[i], start_idx:seq_len, start_idx:seq_len])
#         plt.colorbar(im, ax=ax[i], location="right")
#
#     # knowledge attention heatmap on one head
#     steps = [10, 20, 30, 40]
#     head = heads[0]
#     fig2, ax = plt.subplots(4, 1, figsize=(6, 6))
#     for i in range(len(heads)):
#         xticks = None if i == len(heads) - 1 else []
#         im = heat_map(ax[i], k_score[0, head, steps[i], :, : steps[-1]], xticks=xticks)
#         plt.colorbar(im, ax=ax[i], location="right")
#
#     plt.show()
#     # fig.savefig('qt.pdf', bbox_inches='tight')
#
#
# if __name__ == "__main__":
#     args = parser.parse_args()
#     main(args)
#
#     plt.show()
#     # fig.savefig('qt.pdf', bbox_inches='tight')
#
#
# if __name__ == "__main__":
#     args = parser.parse_args()
#     main(args)
