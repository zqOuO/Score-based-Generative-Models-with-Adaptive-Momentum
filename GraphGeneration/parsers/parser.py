import argparse

class Parser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='GDSS')
        self.parser.add_argument('--type', type=str, required=True)

        self.set_arguments()

    def set_arguments(self):
        

        self.parser.add_argument('--config', type=str,
                                    required=True, help="Path of config file")
        self.parser.add_argument('--comment', type=str, default="", 
                                    help="A single line comment for the experiment")
        self.parser.add_argument('--seed', type=int, default=42)
        
        self.parser.add_argument('--steps', type=int, default=0)
        self.parser.add_argument('--corrector', type=str, help='The sampling corrector')
        self.parser.add_argument('--predictor', type=str, help='The sampling predictor')
        self.parser.add_argument('--snr', type=float, default=0, help='Sampling step snr')
        self.parser.add_argument('--nsigma', type=int, default=0, help='Number of steps per sigmas')        
        self.parser.add_argument('--eps', type=float, default=-1, help='The noise parameter')     

    def parse(self):

        args, unparsed  = self.parser.parse_known_args()
        
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        
        return args