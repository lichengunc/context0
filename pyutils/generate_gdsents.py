"""
This code is to provide a list of ground-truth sentence for each referred object.
The output would be saved as 'sents_gd.json' (others are sents_id0.json)
[{'sent', 'tokens', 'ref_id', 'image_id', 'box', 'split'}]

Note, we may have multiple sentences for each referred object. 
We add all of them to the list, and we don't index sent_id here!
"""
import os.path as osp
import os
import sys
import json
import argparse

# input
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_splitBy', default='refcoco_unc', help='name of dataset+splitBy')
# parser.add_argument('--split', default='testA', help='split name, val|test|train')
args = parser.parse_args()
params = vars(args)

dataset_splitBy = params['dataset_splitBy']  # in Lua, we simply use dataset denoting dataset_splitBy
i = dataset_splitBy.find('_')
dataset, splitBy = dataset_splitBy[:i], dataset_splitBy[i+1:]


# load refer
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(ROOT_DIR, 'pyutils', 'datasets'))
from refer import REFER
refer = REFER(dataset, splitBy = splitBy)

# add ground-truth sents
sents = []
for sent_id, ref in refer.sentToRef.items():
	# if ref['split'] != split:
	# 	continue
	sent = refer.sents[sent_id]  # sent
	tokens = refer.sentToTokens[sent_id]  # tokens
	ann = refer.refToAnn[ref['ref_id']]
	box = ann['bbox']
	sents += [{'sent': sent, 'tokens': tokens, 'ref_id': ref['ref_id'], 'image_id': ref['image_id'], 
	'box': box, 'split': ref['split']}]
print 'sents_gd prepared.'

# save
file_folder = 'cache/box/'+dataset_splitBy
if not osp.isdir(file_folder):
	os.mkdir(file_folder)
file_name = osp.join(file_folder, 'sents_gd.json')
json.dump(sents, open(file_name, 'w'))
