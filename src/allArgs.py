import argparse

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

def getArgs():
    parser = argparse.ArgumentParser(
        description='Training script. Default values of all arguments are recommended for reproducibility',
        fromfile_prefix_chars='@',
        conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
    parser.add_argument('--n-bins', '--n_bins', default=80, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")

    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument('--validate-every', '--validate_every', default=100, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")

    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--dp_crop_func", default='random', type=str,
                        help="The way of depth prior crop. (random, max_count, )")
    parser.add_argument('--dpc_rate', type=float, help='Posibility of depth prior crop', default=0.5)

    parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--do_random_rotate', default=False,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')

    parser.add_argument('--data_path_eval',
                        default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    return parser