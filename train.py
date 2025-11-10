# python -u train.py --from_begin
import argparse
import os

import torch

import p3graph

log = p3graph.utils.get_logger()


def main(args):
    p3graph.utils.set_seed(args.seed)

    # load data
    log.debug("Loading data from '%s'." % args.data)
    data = p3graph.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = p3graph.Dataset(data["train"], args.batch_size)
    # devset = dgcn.Dataset(data["dev"], args.batch_size)
    # testset = dgcn.Dataset(data["test"], args.batch_size)
    # 合并devset和testset
    devset = p3graph.Dataset(data["dev"] + data["test"], args.batch_size)

    log.debug("Building model...")
    model_file = "./save_once/model"
    if not os.path.exists(os.path.dirname(model_file)):
        os.makedirs(os.path.dirname(model_file))
    model = p3graph.P3G(args).to(args.device)
    opt = p3graph.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)

    coach = p3graph.Coach(trainset, devset, model, opt, args)
    # Train.
    log.info("Start training...")
    ret = coach.train()

    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file + '_' + str(ret[0]) + '.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data", type=str, default="data/mpdd/ckpt/data.pkl",
                        help="Path to data")

    # Training parameters
    parser.add_argument("--once", action="store_true", help = "Train once.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computing device.")
    parser.add_argument("--epochs", default=300, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", default=20, type=int,
                        help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["sgd", "rmsprop", "adam"],
                        help="Name of optimizer.")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-8,
                        help="Weight decay.")
    parser.add_argument("--max_grad_value", default=-1, type=float,
                        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""")
    parser.add_argument("--drop_rate", type=float, default=0.4,
                        help="Dropout rate.")

    # Model parameters
    parser.add_argument("--wp", type=int, default=5,
                        help="Past context window size. Set wp to -1 to use all the past context.")
    parser.add_argument("--wf", type=int, default=5,
                        help="Future context window size. Set wp to -1 to use all the future context.")
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Hidden size of two layer GCN.")
    parser.add_argument("--rnn", type=str, default="lstm",
                        choices=["lstm", "gru"], help="Type of RNN cell.")
    parser.add_argument("--class_weight", action="store_true",
                        help="Use class weights in nll loss.")

    # others
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    args = parser.parse_args()
    log.debug(args)

    try:
        main(args)
    finally:
        # 清理CUDA资源
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

