"""
Module with command line interface arguments for Argparser
"""
import argparse

def parse_args():
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Multi Channel Bayesian Personalized Ranking")
    parser.add_argument(
        '--version',
        action='version')
        # version='multi_channel_bpr {ver}'.format(ver=__version__))
    parser.add_argument(
        '-d',
        dest="d",
        help="latent feature dimension",
        type=int,
        metavar="INT",
        default=64)
    parser.add_argument(
        '-beta',
        nargs='+',
        dest="beta_list",
        help="share of unobserved within negative feedback",
        type=float,
        default=[1.],
        metavar="FLOAT")
    parser.add_argument(
        '-lr',
        nargs='+',
        dest="lr_list",
        help="learning rate to test",
        type=float,
        default=[0.05],
        metavar="FLOAT")
    parser.add_argument(
        '-optimizer',
        dest="optim_list",
        nargs='+',
        help="learning rate",
        type=str,
        default=['adam'],
        metavar="STR")
    parser.add_argument(
        '-reg',
        nargs=3,
        dest="reg_param_list",
        help="regularization parameters for user, positive and negative item",
        type=float,
        default=[0.002]*3,
        metavar="FLOAT")
    parser.add_argument(
        '-k',
        nargs='+',
        dest="k",
        help="no. of items with highest predicted rating",
        type=int,
        metavar="INT",
        default=[10,50,100])
    parser.add_argument(
        '-seed',
        dest="rd_seed",
        help="seed for random number generators",
        type=int,
        default=42,
        metavar="INT")
    parser.add_argument(
        '-epochs',
        dest="n_epochs",
        help="no. of training epochs",
        type=int,
        default=10,
        metavar="INT")
    parser.add_argument(
        '-batch_size',
        dest="batch_size",
        help="size of the batch",
        type=int,
        default=256,
        metavar="INT")
    parser.add_argument(
        '-sampling',
        nargs='+',
        dest="neg_sampling_modes",
        help="list of negative item sampling modes",
        type=str,
        default=['uniform'],
        metavar="STR")
    parser.add_argument(
        '-data',
        dest="data_path",
        help="path to read training data from",
        type=str,
        default='../Data/Beibei/train_tb.csv',
        metavar="STR")

    parser.add_argument(
        '-test_data',
        dest="test_data_path",
        help="path to read test data from",
        type=str,
        default='../Data/Beibei/test_tb.csv',
        metavar="STR")
    parser.add_argument(
        '-results',
        dest="results_path",
        help="path to store trained embeddings",
        type=str,
        default='./embeddings/',
        metavar="STR")
    parser.add_argument(
        '-eval_results',
        dest="eval_results_path",
        help="path to write evaluation results into",
        type=str,
        default='./eval_results/',
        metavar="STR")

    return parser.parse_args()

args = parse_args()
