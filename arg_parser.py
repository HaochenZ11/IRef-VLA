'''
Parts adapted from: butd_detr, referit3d, MVT-3DVG repositories
https://github.com/nickgkan/butd_detr
https://github.com/referit3d/referit3d
https://github.com/sega-hsj/MVT-3DVG
'''

import argparse
from utils import str2bool


def parse_args(parser):
    '''
    Takes in parser object from model-specific scripts and adds general args
    :return: parsed args
    '''
    # dataloading
    parser.add_argument('--dataset', type=str, choices=['r3d', 'vla', 'both'], help='Datasets to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size during training')
    parser.add_argument('--data_path', default='data/', help="Path to dataset root folder")
    parser.add_argument('--train_split', default='train', help="Name of train split file")
    parser.add_argument('--test_split', default='test', help="Name of test split file")
    parser.add_argument('--use_multiview', action='store_true', help="Flag for MVT model to use multiple views")
    parser.add_argument('--use_r3d', action='store_true', help="Whether to load r3d data")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--points_per_obj', type=int, default=1024,
                        help='points sampled to make a point-cloud per object of a scan.')
    # data augmentation
    parser.add_argument('--unit_sphere_norm', type=str2bool, default=False,
                        help="Normalize each point-cloud to be in a unit sphere")
    parser.add_argument('--use_height', action='store_true', help='Use height information in input')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input')
    parser.add_argument('--prune_sparse', action='store_true', help='Prune out objects that are too sparse')
    parser.add_argument('--sparsity_thresh', type=int, default=100, help="Number of points to maintain for object pruning")
    parser.add_argument('--use_context', action='store_true', help='Use subset of objects as context in each region')
    parser.add_argument('--context_size', type=int, default=100, help='Max number of objects to use in a region if use_context')
    parser.add_argument('--include_raw_labels', action='store_true', help='Include raw object labels in the dataset.')
    parser.add_argument('--load_false_statements', action='store_true', help='Load false statements in the dataset')
    parser.add_argument('--balance_false', action='store_true', help='Sample at most 1 false class per true class in the dataset')

    # training
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument('--lr_decay_epochs', type=int, default=[40, 50, 60, 70, 80, 90],
                        nargs='+', help='Epoch number at which to decay lr; can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.65, help='Decay rate for lr if using step scheduler')
    parser.add_argument('--clip_norm', default=0.1, type=float, help='Gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--reduce_lr', action='store_true')
    parser.add_argument('--random_seed', type=int, default=2020,help='Control pseudo-randomness for reproducibility')
    parser.add_argument('--include_binary_classifier', action='store_true')

    # logging
    parser.add_argument('--proj_name', required=True, type=str, help='Project name for wandb')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for wandb')
    parser.add_argument('--resume_path', default=None, help='Path to existing model checkpoint')
    parser.add_argument('--log_dir', default='log/', help='Dump dir to save model checkpoint')
    parser.add_argument('--log_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--val_freq', type=int, default=5)  # batch-wise
    parser.add_argument('--test_freq', type=int, default=5)  # epoch-wise
    parser.add_argument('--save_freq', type=int, default=5)  # epoch-wise

    # others
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--loss_type', default='refer')

    args, _ = parser.parse_known_args()

    # argument error handling
    if args.include_binary_classifier and not args.load_false_statements:
        print("Warning: include_binary_classifier is set to True, but load_false_statements is set to False.")
    if args.include_raw_labels and args.batch_size > 1:
        raise ValueError("Batch size must be 1 if include_raw_labels is set to True.")

    return args
