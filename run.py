
import argparse
import os, errno
import sys
import yaml

from program import Program
from models.rec.model import Model
from data.data_loader import Load_Loader

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="text_recognition model training session."
    )
    
    parser.add_argument(
        "-pt",
        "--phoneme_type",
        type=bool,
        help="Set target source as {True:phoneme / False:character}",
        default=False
    )
    
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Set mode to Train/Test",
        default='Train'
    )
    
    parser.add_argument(
        "-d",
        "--delete",
        type=bool,
        help="Delete current saving folder",
        default=False
    )
    
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="Folder path where target data stored (Only available when Test mode triggerd)",
        default='./test_data/'
    )
    
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Name for save current training session (under ./result folder)"
    )
    
    return parser.parse_args()

def main():
    
    args = parse_arguments()
    
    if args.phoneme_type:
        with open('./conf/phoneme.yml') as f:
            #https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
            conf = yaml.load(f,Loader=yaml.FullLoader)
    else:
        with open('./conf/character.yml') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
            
    dataloader, num_target = Load_Loader(conf)
    program = Program(conf)
    model = Model(conf,num_target+2)
    
    if args.mode=='Train':
        program.train(model, dataloader, args.name,args.delete)
    else:
        if not os.path.isdir(args.folder):
            raise FileNotFoundError(f'No such Test data set {args.folder}')
        program.test(model, args.folder, dataloader)

if __name__ == "__main__":
    main()
