"""
This code is used to evaluate "sent --> box" accuracy.

Only one input is needed, i.e., model_idx_split_boxes(gd)_sent(gd).json
It should contains the following:
[{'sent', 'tokens', 'ref_id', 'image_id', 'box', 'split', 'pred_box', 'pred_loss'}]

We will compare 'box' and 'pred_box' if there IoU >= 0.5, and we will write html later on.
"""
import os.path as osp
import os
import sys
import json
import argparse

# input
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_splitBy', default='refcoco_licheng', help='name of dataset+splitBy')
parser.add_argument('--model_id', default='0', help='model_id to be loaded')
parser.add_argument('--split', default='testA', help='split name, val|test|train')
parser.add_argument('--boxes_type', default='gd', help='boxes generation type, gd or pred')
parser.add_argument('--sents_type', default='gd', help='sentences generation type, gd or id0, id1, ...')
args = parser.parse_args()
params = vars(args)

dataset_splitBy = params['dataset_splitBy']
i = dataset_splitBy.find('_')
dataset, splitBy = dataset_splitBy[:i], dataset_splitBy[i+1:]
model_id = params['model_id']
split = params['split']
boxes_type = params['boxes_type']
sents_type = params['sents_type']

# load predictions
if sents_type == 'gd': 
	# this is sent->box evaluation
	pred_path = osp.join('cache/box', dataset_splitBy, 'model_id'+model_id+'_'+split+'_boxes('+
		boxes_type+')_sents(gd).json')
else:
	# this is box->sent->box evaluation
	pred_path = osp.join('cache/langToBox', dataset_splitBy, 'model_id'+model_id+'_'+split+'_boxes('+
		boxes_type+')_sents(id'+model_id+').json')
	
predictions = json.load(open(pred_path, 'r'))['predictions']

# IoU function
def computeIoU(box1, box2):
	# each box is of [x1, y1, w, h]
	inter_x1 = max(box1[0], box2[0])
	inter_y1 = max(box1[1], box2[1])
	inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
	inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

	if inter_x1 < inter_x2 and inter_y1 < inter_y2:
		inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
	else:
		inter = 0
	union = box1[2]*box1[3] + box2[2]*box2[3] - inter
	return inter, union

# compute accuracy
acc = 0
for sent in predictions:
	gd_box = sent['box']
	pred_box = sent['pred_box']
	inter, union = computeIoU(gd_box, pred_box)
	IoU = float(inter)/union
	if IoU >= 0.5:
		acc += 1

# print results
print('\nHere is the sent->box result:')
print(' dataset_splitBy: %s' % dataset_splitBy)
print(' model_id: %s' % model_id)
print(' split: %s' % split)
print(' boxes_type: %s' % boxes_type)
print(' sents_type: %s' % sents_type)
print(' sent->box accuracy is %.2f%%\n' % (acc*100.0/len(predictions)))

