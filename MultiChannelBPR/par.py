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
        '-factor_num',
        dest="factor_num",
        help="latent feature dimension",
        type=int,
        default =64,
        metavar="INT")
    parser.add_argument(
        '-beta',
        help="share of unobserved within negative feedback",
        type=float,
        default=1,
        metavar="FLOAT")
    parser.add_argument(
        '-lr',
        dest="lr",
        help="learning rate",
        type=float,
        default=0.001,
        metavar="FLOAT")
    parser.add_argument(
        '-lamda',
        help="regularization parameters",
        type=float,
        default=0.0001,
        metavar="FLOAT")
    parser.add_argument(
        '-rel_order',
        nargs=4,
        help="the order of importance of interactions (buy(0), pv(1), cart(2))",
        type=int,
        default = [0, 2, 1],
        metavar="INT")
    parser.add_argument(
        '-rel_weight',
        nargs= 3,
        help="the sample weight of buy, pv, cart",
        type=float,
        default = [1, 0, 5],
        metavar="float")
    parser.add_argument("--neg_num", 
		type=int,
		default=4, 
		help="sample negative items for training")
    parser.add_argument(
        '-seed',
        dest="rd_seed",
        help="seed for random number generators",
        type=int,
        default=42,
        metavar="INT")
    parser.add_argument(
        "-batch_size", 
		type=int, 
		default=256, 
		help="batch size for training")
    parser.add_argument(
        '-epoch',
        help="no. of training epochs",
        type=int,
        default=1000,
        metavar="INT")
    parser.add_argument(
        '-sample_mode',
        help="negative item sampling modes (uniform or multi_level)",
        type=str,
        default='uniform',
        metavar="STR")
    parser.add_argument(
        "-Ks", 
		nargs='?', 
		default = '[1,5,10,20,50,80,100]',
		help="compute metrics@top_k")
    parser.add_argument(
        '-data_path',
        dest="data_path",
        help="path to read the dataset",
        type=str,
        default='../data/',
        metavar="STR")
    parser.add_argument("-dataset", 
		type=str, 
		default='Beibei', 
		help="dataset (Beibei or Taobao)",
        metavar="STR")
    parser.add_argument(
        '-results',
        dest="results_path",
        help="path to write results into",
        type=str,
        default='../results/',
        metavar="STR")
    return parser.parse_args()

args = parse_args()
