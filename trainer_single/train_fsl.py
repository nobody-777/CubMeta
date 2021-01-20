from trainer_single.fsl_trainer import FSLTrainer
from trainer_single.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # with launch_ipdb_on_exception():
    pprint(vars(args))
    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    trainer.train()
    trainer.evaluate_test()
    print(args.save_path)



