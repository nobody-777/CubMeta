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

    args.save_path =  args.test_model

    print("\nTest with Max Prob Acc: ")
    trainer.evaluate_test('max_pa.pth')

    print("Test with Min Loss: ")
    trainer.evaluate_test('min_loss.pth')

    print("\nTest with last epoch: ")
    trainer.evaluate_test('last.pth')




