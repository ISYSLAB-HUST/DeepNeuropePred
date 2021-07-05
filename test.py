import tempfile
import os
import argparse
import json

from Bio import SeqIO

from model.utils import feature_generate
from model.data import NeuroPepDataset, collate_fn
from model.net import AttentiveNet
from torch.utils.data import DataLoader
import torch


def test_data_iter(data_path):
    data = NeuroPepDataset(data_path)
    test = DataLoader(data, len(data), collate_fn=collate_fn)
    return test



def pred_cl(sequence, model_path):

    tmp_dir = tempfile.mkdtemp(suffix='-neuropep')
    with open(os.path.join(tmp_dir, 'seq.fasta'), mode='w') as fw:
        fw.write(sequence)

    signalp_pos = feature_generate(os.path.join(
        tmp_dir, 'seq.fasta'), tmp_dir, device='cpu')
    test_loader = test_data_iter(tmp_dir)
    pos, pred_prob = neuropepCpred(model_path, test_loader, 'cpu')

    pos = [item+signalp_pos+1 for item in pos]
    data = sorted([(i, j) for i, j in zip(pos, pred_prob)], key=lambda k: k[0])
    return {'signal_pos': signalp_pos, 'predict':data, 'sequence':sequence}


def neuropepCpred(model_path, test_loader, device):
    model = AttentiveNet(768, 16)
    model_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_dict['state_dict'])
    model.eval()
    with torch.no_grad():
        for _, (tokens, pos) in enumerate(test_loader):
            tokens = tokens.to(device)
            predict = model(tokens)
        return pos, predict.cpu().tolist()

def get_params():
    parser = argparse.ArgumentParser('DeepNeuropePred model')


    parser.add_argument('--model-file', type=str, help='model file')
    parser.add_argument('--input-fasta', type=str, help='fasta file')
    parser.add_argument('--ouput-json', type=str, help='json file')

    args, _ = parser.parse_known_args()
    return args


if __name__ == "_main__":
    args = vars(get_params())
    model_path = args['model_file']
    out = {}

    for record in SeqIO.parse(args['input_fasta']):
        out[record.id] = pred_cl(str(record.seq), model_path)

    json.dump(out, open(args['output_json'], 'w'))
