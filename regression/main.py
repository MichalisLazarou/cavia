import arguments
import cavia
import maml
import trans_maml
import torch


if __name__ == '__main__':

    args = arguments.parse_args()
    print(args)
    if args.method == 'maml' :
        logger = maml.run(args, log_interval=100, rerun=True)
    elif args.method == 'transformer':
        #print("transformer")
        with torch.autograd.detect_anomaly():
            logger = trans_maml.run(args, log_interval=100, rerun=True)
    else:
        logger = cavia.run(args, log_interval=100, rerun=True)
