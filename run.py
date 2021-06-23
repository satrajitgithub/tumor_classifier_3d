import argparse
from routines import predict_val_test

parser = argparse.ArgumentParser(description='Argument parser, provide fold and experiment')
parser.add_argument('-v','--val', nargs='?', help='Choose validation fold', default="1")
parser.add_argument('-e','--exp', nargs='?', help='Give experiment name', default="001")

args = vars(parser.parse_args())

predict_val_test.main(fold=args['val'], exp=args['exp'])


	
