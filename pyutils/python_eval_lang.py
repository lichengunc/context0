"""
This code is used to evalute the validation results.
The validation json file is of data [{'ref_id', 'sent'}]
We call REFER and RefEvaluation to evalute different types of scores.

Things from RefEvaluation of interests:
evalRefs  - list of ['ref_id', 'CIDEr', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR']
eval      - dict of {metric: score}
refToEval - dict of {ref_id: ['ref_id', 'CIDEr', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR']}

We name output file as 'model_idx_split_out.json'
"""
import os
import os.path as osp
import sys
import json
import argparse

# val_json is dataset_splitBy_modelidX_val|test.json
# get dataset, splitBy, and model json
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_splitBy', default='refcoco_licheng', help='name of dataset+splitBy')
parser.add_argument('--model_id', default='0', help='model_id to be loaded')
parser.add_argument('--split', default='test', help='split name, val|test|train')
args = parser.parse_args()
params = vars(args)

dataset_splitBy = params['dataset_splitBy']   # in Lua, we simply use dataset denoting detaset_splitBy
i, j = dataset_splitBy.find('_'), dataset_splitBy.rfind('_')
dataset = dataset_splitBy[:i]
splitBy = dataset_splitBy[i+1:] if i == j else dataset_splitBy[i+1:j]
model_id = params['model_id']
split = params['split']

# load refer and refToEvaluation
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(ROOT_DIR, 'pyutils', 'datasets'))
sys.path.insert(0, osp.join(ROOT_DIR, 'pyutils', 'evaluation'))
from refer import REFER
from refEvaluation import RefEvaluation
refer = REFER(dataset, splitBy = splitBy)

# load predictions
result_path = osp.join('cache/lang', dataset_splitBy, 'model_id' + model_id + '_' + split)
Res = json.load(open(result_path + '.json', 'r'))['predictions']  # [{'ref_id', 'sent'}]

# evaluate
refEval = RefEvaluation(refer, Res)
refEval.evaluate()
overall = {}
for metric, score in refEval.eval.items():
	overall[metric] = score
print overall

refToEval = refEval.refToEval
for res in Res:
	ref_id, sent = res['ref_id'], res['sent']
	refToEval[ref_id]['sent'] = sent
with open(result_path + '_out.json', 'w') as outfile:
	json.dump({'overall': overall, 'refToEval': refToEval}, outfile)




