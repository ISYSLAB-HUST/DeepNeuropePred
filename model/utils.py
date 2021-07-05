'''
utils
'''

from Bio import SeqIO
import torch
import numpy as np
import os
import re
from subprocess import call
import shlex
c_s = ['K', 'R']


def esm_feature(data_list, device):
    import esm
    model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data_list)

    # Extract per-residue representations (on CPU)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)
    token_representations = results["representations"][6]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, (_, seq) in enumerate(data_list):
        sequence_representations.append(
            token_representations[i, 1: len(seq) + 1])

    return sequence_representations, batch_labels

def signalP(seq, output_dir):
    output_path = os.path.join(output_dir, 'neuropep_signalp_summary.txt')
    cmd_str = "signalp -fasta {0} -org euk -format short -prefix neuropep_signalp -stdout".format(seq)
    p = call(shlex.split(cmd_str), cwd=output_dir, stdout=open(output_path, 'w'), shell=False)
    cs_postiton_str = open(output_path, 'r').readlines()[-1].split('\t')[-1]
    if re.search(r'[1-9]\d-[1-9]\d', cs_postiton_str) == None:
        return 0
    cs_postiton =re.search(r'[1-9]\d-[1-9]\d', cs_postiton_str).group().split('-')[0]
    return int(cs_postiton)

def feature_generate(file_name, output_dir, device):
    cs_postiton = signalP(file_name, output_dir)
    for record in SeqIO.parse(file_name, 'fasta'):
        protein_str = record.seq
        sequence_feature, batch_name = esm_feature([(record.id, protein_str)], device)
        new_feature = sequence_feature[0].cpu().numpy()
        if cs_postiton == 'None':
            pos = 0
        else:
            pos = int(cs_postiton)
        for i, aa in enumerate(protein_str[pos:]):
            if aa in c_s:
                left_cut = max(0, i-8)
                right_cut = min(len(protein_str[pos:]), i+10)
                if i == len(protein_str[pos:]) - 1:
                    continue
                if len(protein_str[pos:][i:right_cut]) >= 5 and len(protein_str[pos:][left_cut:i]) >= 5:
                    if (protein_str[pos:][i+1] not in c_s) and (protein_str[pos:][i+4] not in c_s):
                        _temp = new_feature[pos:][left_cut:right_cut]

                        np.savez(os.path.join(output_dir, record.id + f'_{i}' +'.npz'),
                                feature=_temp, pos = np.array(i))
        return cs_postiton


if __name__ == '__main__':
    import sys
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    feature_generate(sys.argv[1], sys.argv[2], device)
