'''
# Train
python ./main.py --mode train --in_file_trn_dialog ./data/ddata.wst.txt --in_file_fashion ./data/mdata.wst.txt --in_file_img_feats ./data/extracted_feat.json --subWordEmb_path ./sstm_v0p5_deploy/sstm_v4p49_np_final_n36134_d128_r_eng_upper.dat --model_path ./gAIa_model --mem_size 16 --key_size 300 --hops 3 --eval_node [6000,6000,6000,200][2000,2000] --epochs 10 --save_freq 1 --batch_size 100 --learning_rate 0.005 --max_grad_norm 20.0 --use_multimodal True --use_dropout True --zero_prob 0.5 --permutation_iteration 3 --num_augmentation 5 --corr_thres 0.7

# Test
python ./main.py --mode test --in_file_tst_dialog ./data/ac_eval_t1.wst.dev --in_file_fashion ./data/mdata.wst.txt --in_file_img_feats ./data/extracted_feat.json --subWordEmb_path ./sstm_v0p5_deploy/sstm_v4p49_np_final_n36134_d128_r_eng_upper.dat --model_path ./gAIa_model --model_file gAIa-10.pt --mem_size 16 --key_size 300 --hops 3 --eval_node [6000,6000,6000,200][2000,2000] --batch_size 100 --use_multimodal True

# Predict
python /home/work/model/main.py --mode pred --in_file_tst_dialog /home/work/data/ac_eval_t1.wst.dev --in_file_fashion /home/work/data/mdata.wst.txt --in_file_img_feats /home/work/data/extracted_feat.json --subWordEmb_path /home/work/model/sstm_v0p5_deploy/sstm_v4p49_np_final_n36134_d128_r_eng_upper.dat --model_path /home/work/model/gAIa_model --model_file gAIa-10.pt --mem_size 16 --key_size 300 --hops 3 --eval_node [6000,6000,6000,200][2000,2000] --batch_size 100 --use_multimodal True
'''

'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2021, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2021.05.03.
'''


import argparse
import torch
from gaia import *
import os
cores = os.cpu_count()
torch.set_num_threads(cores)


def get_udevice():
    """
    function: get usable devices(CPU and GPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpu = torch.cuda.device_count()
    else:    
        device = torch.device('cpu')
    print('Using device: {}'.format(device))
    if torch.cuda.is_available():
        print('# of GPU: {}'.format(num_gpu))
    device = torch.device('cpu')
    return device


def str2bool(v):
    """
    function: convert into bool type(True or False)
    """
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')


# input options
parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')

parser.add_argument('--mode', type=str, 
                    default='train', 
                    help='training or eval or test mode')
parser.add_argument('--in_file_trn_dialog', type=str, 
                    default='./data/ddata.wst.txt', 
                    help='training dialog DB')
parser.add_argument('--in_file_tst_dialog', type=str, 
                    default='./data/ac_eval_t1.wst.dev', 
                    help='test dialog DB')
parser.add_argument('--in_file_fashion', type=str, 
                    default='./data/mdata.wst.txt', 
                    help='fashion item metadata')
parser.add_argument('--in_file_img_feats', type=str, 
                    default='./data/extracted_feat.json', 
                    help='fashion item image features')
parser.add_argument('--model_path', type=str, 
                    default='./gAIa_model', 
                    help='path to save/read model')
parser.add_argument('--model_file', type=str, 
                    default=None, 
                    help='model file name')
parser.add_argument('--eval_node', type=str, 
                    default='[6000,6000,6000,200][2000,2000]', 
                    help='nodes of evaluation network')
parser.add_argument('--subWordEmb_path', type=str, 
                    default='./sstm_v0p5_deploy/sstm_v4p49_np_final_n36134_d128_r_eng_upper.dat', 
                    help='path of subword embedding')
parser.add_argument('--learning_rate', type=float,
                    default=0.0001, 
                    help='learning rate')
parser.add_argument('--max_grad_norm', type=float,
                    default=40.0, 
                    help='clip gradients to this norm')
parser.add_argument('--zero_prob', type=float,
                    default=0.0, 
                    help='dropout prob.')
parser.add_argument('--corr_thres', type=float,
                    default=0.7, 
                    help='correlation threshold')
parser.add_argument('--batch_size', type=int,
                    default=100,   
                    help='batch size for training')
parser.add_argument('--epochs', type=int,
                    default=10,   
                    help='epochs to training')
parser.add_argument('--save_freq', type=int,
                    default=2,   
                    help='evaluate and save results per # epochs')
parser.add_argument('--hops', type=int,
                    default=3,   
                    help='number of hops in the MemN2N')
parser.add_argument('--mem_size', type=int,
                    default=16,   
                    help='memory size for the MemN2N')
parser.add_argument('--key_size', type=int,
                    default=300,   
                    help='memory size for the MemN2N')
parser.add_argument('--permutation_iteration', type=int,
                    default=3,   
                    help='# of permutation iteration')
parser.add_argument('--evaluation_iteration', type=int,
                    default=10,   
                    help='# of test iteration')
parser.add_argument('--num_augmentation', type=int,
                    default=3,   
                    help='# of data augmentation')
parser.add_argument('--use_batch_norm', type=str2bool, 
                    default=False, 
                    help='use batch normalization')
parser.add_argument('--use_dropout', type=str2bool, 
                    default=False, 
                    help='use dropout')
parser.add_argument('--use_multimodal', type=str2bool,
                    default=True, 
                    help='use multimodal input')

args = parser.parse_args()


if __name__ == '__main__':
    
    print('\n')
    print('-'*60)
    print('\t\tAI Fashion Coordinator')
    print('-'*60)
    print('\n')

    mode = args.mode    
    if mode not in ['train', 'test', 'pred'] :
        raise ValueError('Unknown mode {}'.format(mode))

    print('<Parsed arguments>')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('')
    
    gaia = gAIa(args, get_udevice())
    if mode == 'train':
        # training
        gaia.train()
    elif mode == 'test':
        # test
        gaia.test()
    elif mode == 'pred':
        # pred
        gaia.pred()

