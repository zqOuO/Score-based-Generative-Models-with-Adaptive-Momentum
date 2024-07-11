import yaml
from easydict import EasyDict as edict


def get_config(args, seed):
    config_dir = f'./config/{args.config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed

    if args.steps != 0:
        config.sample.steps = args.steps

    if args.corrector:
        config.sampler.corrector = args.corrector

    if args.predictor:
        config.sampler.predictor = args.predictor

    if args.snr != 0:
        config.sampler.snr = args.snr

    if args.nsigma !=0:
        config.sampler.n_steps = args.nsigma

    if args.eps != -1:
        config.sampler.scale_eps = args.eps
        
    return config