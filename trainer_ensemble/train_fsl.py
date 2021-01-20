import numpy as np
import torch
from trainer_ensemble.fsl_trainer import FSLTrainer
from trainer_ensemble.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
# from ipdb import launch_ipdb_on_exception

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    trainer.train()



