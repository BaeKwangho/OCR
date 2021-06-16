import torch
import argparse
import os, errno
import sys
import yaml

from program import Program
from data.data_loader import Load_Loader, split_loader

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
        default='Test'
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
        help="Name for save current training session or load trained model to test (under ./result folder)"
    )
    parser.add_argument(
        "-cm",
        "--choose_model",
        type=str,
        required=True,
        help="Select model to extract feature(default : CRNN)"
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
    program = Program(conf, args)

    if(args.choose_model=="CRNN"):
        from models.crnn.model import Model
        model = Model(conf, num_target+2)
    elif(args.choose_model=="ASTER"):
        from models.aster.model import Model
        model = Model(conf, num_target+2, sDim=512, attDim=512, max_len_labels=25, STN_ON=True)
    else:
        raise("You write wrong model name!")        
    
    if args.mode=='Train':
        test_len = 20000 // conf["Program"]["batch_size"]
        train_loader, valid_loader = split_loader(dataloader, conf["Program"]["batch_size"])
        program.train(model, dataloader, train_loader, valid_loader)
    else:
        if not os.path.isdir(args.folder):
            raise FileNotFoundError(f'No such Test data set {args.folder}')
        program.test(model, args.folder, dataloader)

if __name__ == "__main__":
    main()