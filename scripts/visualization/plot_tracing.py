import os
import sys
from argparse import ArgumentParser

import torch
import tomlkit
import matplotlib.pyplot as plt

from DTransformer.data import KTData
from DTransformer.model import DTransformer
from DTransformer.visualize import trace_map

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
parser.add_argument("--n_layers", help="number of layers", type=int, default=1)
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
    y = model.tracing(q, s, pid)

    # knowledge tracing on a specific knowledge set
    ind_k = [0, 1, 3, 5, 6]
    span = range(0, 25)
    fig = trace_map(y[ind_k, :], q, s, span, text_label=True)

    plt.show()
    # fig.savefig('tracing.pdf', bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


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
# from DTransformer.visualize import trace_map
#
# # 获取当前文件的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取项目根目录
# project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
# # 将项目根目录添加到 Python 路径中
# sys.path.insert(0, project_root)
#
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
#     "--n_know", help="dimension of knowledge parameter", type=int, default=32
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
#     y = model.tracing(q, s, pid)
#
#     # knowledge tracing on a specific knowledge set
#     ind_k = [0, 1, 3, 5, 6]
#     # 确保 span 不会超出 y 和 q 的实际大小
#     max_span = min(25, y.shape[1], q.shape[0])
#     span = range(0, max_span)
#     fig = trace_map(y[ind_k, :], q, s, span, text_label=True)
#
#     plt.tight_layout()  # 调整图形布局
#     plt.show()
#     # fig.savefig('tracing.pdf', bbox_inches='tight')
#
#
# if __name__ == "__main__":
#     args = parser.parse_args()
#     main(args)

