"""
This code is to generate ground-truth object proposals for each image.
It provides candidate boxes to evaluate "sent --> box"
1) we first feed each box feature (global+referr+location) to the model.
2) Then we rank the output loss of each box, and pick up the lowest one. 
That's the box most relevant to the sent accoring to our model.

During evaluation, we could use ground-truth object proposals or most possible object proposals.
This code is to generate the former ones for each image.
"""
import os.path as osp
import os
import sys
import json
import argparse

# input
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_splitBy', default='refcoco_unc', help='name of dataset+splitBy')
# parser.add_argument('--model_id', default=0, help='model_id to be loaded')
# parser.add_argument('--split', default='test', help='split name, val|test|train')
args = parser.parse_args()
params = vars(args)

dataset_splitBy = params['dataset_splitBy']   # in Lua, we simply use dataset denoting detaset_splitBy
i = dataset_splitBy.find('_')
dataset, splitBy = dataset_splitBy[:i], dataset_splitBy[i+1:]
# model_id = params['model_id']
# split = params['split']

# load refer
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(ROOT_DIR, 'pyutils', 'datasets'))
from refer import REFER
refer = REFER(dataset, splitBy = splitBy)

# pick up images appeared in refer dataset
imageToBoxes = {}
image_ids = []
for _, ref in refer.Refs.items():
	image_ids += [ref['image_id']]
image_ids = list(set(image_ids))
print 'related %d image_ids computed.' % len(image_ids)

# compute image_id --> boxes
for image_id in image_ids:
	anns = refer.imgToAnns[image_id]
	for ann in anns:
		imageToBoxes[image_id] = imageToBoxes.get(image_id, []) + [{'ann_id': ann['id'], 'box':ann['bbox']}]
print 'image_id --> boxes computed.'

# save
file_folder = 'cache/box/'+dataset_splitBy
if not osp.isdir(file_folder):
	os.mkdir(file_folder)
file_name = osp.join(file_folder, 'imageToBoxes_gd.json')
json.dump(imageToBoxes, open(file_name, 'w'))


