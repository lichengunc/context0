__author__ = 'licheng'

"""
Interface for accessing the referit game dataset.
# The following API functions are defined:
# REFER       - refer api class that loads clef, coco, and coco+, and prepare data structures
# getRefIds   - Get ref_id(s) that satisfy given filter conditions
# getAnnIds   - Get ann_id(s) that satisfy given filter conditions
# loadRefs    - Load ref(s) with the specified ids
# loadAnns    - Load ann(s) with the specified ids
# loadImgs    - Load img(s) with the specified ids
# getMask     - Get mask given ref

Some constructors:
# Refs        - dictionary of {ref_id: ref}
# imgToRefs   - dictionary of {image_id: refs}
# catToRefs   - dictionary of {category_id: refs}
# sentToParse - dictionary of {sent_id: parse}
# anns        - dictionary of {ann_id: ann}
# imgs        - dictionary of {image_id: image}
# cats        - dictionary of {category_id: category name}
# imgToAnns   - dictionary of {image_id: anns}
# refToAnn    - dictionary of {ref_id: ann}
# annToRef    - dictionary of {ann_id: ref}
# sentToRef   - dictionary of {sent_id: ref}
# sents       - dictionary of {sent_id: sent}
# sentToTokens- dictionary of {sent_id: tokens}
"""

import sys
import os.path as osp
import json
import cPickle as pickle
import itertools
import skimage.io as io
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pprint import pprint
import numpy as np
from pycocotools import mask
from skimage.measure import label, regionprops

class REFER:
    def __init__(self, dataset, with_parses=False, splitBy='unc'):
        '''
        prepare dataset: dataset(name), refs, parses, images, annotations
        '''
        print 'loading dataset %s into memory...' % dataset
        self.ROOT_DIR  = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
        self.DATA_DIR  = osp.join(self.ROOT_DIR, 'data', dataset)
        self.CACHE_DIR = osp.join(self.ROOT_DIR, 'cache', dataset)
        self.IMAGE_DIR = osp.join(self.ROOT_DIR, 'data', 'images', 'mscoco', 'images', 'train2014') if dataset in ['refcoco', 'refcoco+', 'refcocoG'] \
            else osp.join(self.ROOT_DIR, 'data', 'images', 'saiapr_tc-12')
        self.PROPOSAL_DIR = osp.join(self.ROOT_DIR, 'data', 'images', 'mscoco', 'proposals', 'train2014') if dataset in ['refcoco', 'refcoco+', 'refcocoG'] \
            else None

        # load dataset
        refer_file   = osp.join(self.DATA_DIR, 'cleaned('+splitBy+').p')
        self.data = {}
        self.data['dataset']    = dataset

        # load list of refs
        self.data['refs'] = pickle.load(open(refer_file, 'r'))

        # load parses {sent_id: parse}
        if with_parses:
            parse_file = osp.join(self.CACHE_DIR, 'sentToParse.p')
            self.data['sentToParse'] = pickle.load(open(parse_file, 'r'))

        # load image annotations
        annotation_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(annotation_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()

    def createIndex(self):
        # create Refer index
        print 'creating index...'
        Refs = {}  # This is a dictionary {ref_id: ref}
        imgToRefs = {}
        catToRefs = {}
        sentToParse = {}
        anns = {}
        imgs = {}
        cats = {}
        imgToAnns = {}
        refToAnn = {}
        annToRef = {}
        sentToRef = {}
        sents = {}
        sentToTokens = {}
        if 'refs' in self.data:
            imgToRefs  = {ref['image_id']:    [] for ref in self.data['refs']}
            catToRefs  = {ref['category_id']: [] for ref in self.data['refs']}
            Refs       = {ref['ref_id']:      [] for ref in self.data['refs']}
            for ref in self.data['refs']:
                # get id
                ref_id = ref['ref_id']
                category_id = ref['category_id']
                image_id = ref['image_id']
                # add mapping
                imgToRefs[image_id] += [ref]
                catToRefs[category_id] += [ref]
                Refs[ref_id] = ref
                # add sent_id -> ref
                for sent_id in ref['sent_ids']:
                    sentToRef[sent_id] = ref
                # add sent_id -> sent
                for sent in ref['sentences']:
                    sents[sent['sent_id']] = sent['sent']
                # add sent_id -> tokens
                for sent in ref['sentences']:
                    sentToTokens[sent['sent_id']] = sent['tokens']

        if 'parses' in self.data:
            sentToParse = self.data['sentToParse']

        if 'images' in self.data:
            imgs = {im['id']:  [] for im in self.data['images']}
            for img in self.data['images']:
                if self.PROPOSAL_DIR:
                    img['proposal_file_name'] = str(img['id'])+'.npy'  # if proposal computed, add its filename to each img
                imgs[img['id']] = img

        if 'annotations' in self.data:
            imgToAnns = {ann['image_id']: [] for ann in self.data['annotations']}
            anns = {ann['id']: [] for ann in self.data['annotations']}
            for ann in self.data['annotations']:
                imgToAnns[ann['image_id']] += [ann]
                anns[ann['id']] = ann

        if 'categories' in self.data:
            cats = {cat['id']: cat['name'] for cat in self.data['categories']}

        if 'refs' in self.data and 'annotations' in self.data:
            for ref_id, ref in Refs.items():
                refToAnn[ref_id] = anns[ref['ann_id']]
                annToRef[ref['ann_id']] = ref                   # Note there are more anns than refs!

        # create class members
        self.Refs         = Refs
        self.imgToRefs    = imgToRefs
        self.catToRefs    = catToRefs
        self.sentToParse  = sentToParse
        self.anns         = anns
        self.imgs         = imgs
        self.cats         = cats
        self.imgToAnns    = imgToAnns
        self.refToAnn     = refToAnn
        self.annToRef     = annToRef
        self.sentToRef    = sentToRef
        self.sents        = sents
        self.sentToTokens = sentToTokens

    def getRefIds(self, imgIds=[], catIds=[], refIds=[], split=''):
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]
        refIds = refIds if type(refIds) == list else [refIds]

        if len(imgIds) == len(catIds) == len(refIds) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(imgIds) == 0:
                refs = [self.imgToRefs[image_id] for image_id in imgIds]
            else:
                refs = self.data['refs']
            if not len(catIds) == 0:
                refs = [ref for ref in refs if ref['category_id'] in catIds]
            if not len(refIds) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in refIds]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider 'testA' in 'testAB'
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print 'No such split [%s]' % split
                    sys.exit()
        ids = [ref['ref_id'] for ref in refs]
        return ids

    def getAnnIds(self, imgIds=[], catIds=[], refIds=[]):
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]
        refIds = refIds if type(refIds) == list else [refIds]

        if len(imgIds) == len(catIds) == len(refIds) == 0:
            ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[image_id] for image_id in imgIds if image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists)) # convert above to list
            else:
                anns = self.data['annotations']
            if not len(catIds) == 0:
                anns = [ann for ann in anns if ann['category_id'] in catIds]
            ids = [ann['id'] for ann in anns]
            if not len(refIds) == 0:
                ids = set(ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in refIds]))
        return ids

    def loadAnns(self, ids=[]):
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadRefs(self, ids=[]):
        if type(ids) == list:
            return [self.Refs[id] for id in ids]
        elif type(ids) == int:
            return [self.Refs[ids]]

    def loadImgs(self, ids=[]):
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showRef(self, ref):
        # show image
        image = self.imgs[ref['image_id']]
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        plt.figure()
        plt.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print '%s. %s' % (sid+1, sent['sent'])
        # show annotation
        ann_id = ref['ann_id']
        ann    = self.anns[ann_id]
        ax = plt.gca()
        polygons = []
        color = []
        # c = np.random.random((1, 3)).tolist()[0]
        c = 'none'
        if type(ann['segmentation'][0]) == list:
            # polygon
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((len(seg)/2, 2))
                polygons.append(Polygon(poly, True, alpha=0.4))
                color.append(c)
            p = PatchCollection(polygons, facecolors=color, edgecolors=(1,1,0,0), linewidths=3, alpha=1)
            ax.add_collection(p)  # yellow polygon
            p = PatchCollection(polygons, facecolors=color, edgecolors=(1,0,0,0), linewidths=1, alpha=1)
            ax.add_collection(p)  # red polygon
        else:
            # mask
            rle = ann['segmentation']
            m = mask.decode(rle)
            img = np.ones( (m.shape[0], m.shape[1], 3) )
            color_mask = np.array([2.0,166.0,101.0])/255
            for i in range(3):
                img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack( (img, m*0.5) ))
            # p = PatchCollection(polygons, facecolors=color, edgecolors=(0,0,0,1), linewidths=3, alpha=0.4)
            # ax.add_collection(p)
        plt.show()

    def getMask(self, ref):
        '''
        :return: mask, mask-area, mask-center
        '''
        ann = self.refToAnn[ref['ref_id']]
        image = self.imgs[ref['image_id']]
        if type(ann['segmentation'][0]) == list: # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else: # mask
            rle = ann['segmentation']
        m = mask.decode(rle)
        m = np.sum(m, axis=2)   # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # area
        area = sum(mask.area(rle))              # very close to ann['area']
        # position
        position_x = np.mean(np.where(m==1)[1]) # [1] means columns (matlab style) -> x (c++ style)
        position_y = np.mean(np.where(m==1)[0]) # [0] means rows (matlab style)    -> y (c++ style)
        # mass position (If there were multiple regions, we use the largest one.)
        label_m = label(m, connectivity=m.ndim)
        regions = regionprops(label_m)
        if len(regions) > 0:
            largest_id = np.argmax(np.array([props.filled_area for props in regions]))
            largest_props = regions[largest_id]
            mass_y, mass_x = largest_props.centroid
        else:
            mass_x, mass_y = position_x, position_y
        # if centroid is not in mask, we find the closest point to it from mask
        if m[mass_y, mass_x] != 1:
            print 'Finding closest mask point...'
            kernel = np.ones((10, 10),np.uint8)
            me = cv2.erode(m, kernel, iterations = 1)
            points = zip(np.where(me == 1)[0].tolist(), np.where(me == 1)[1].tolist())  # row, col style
            points = np.array(points)
            dist   = np.sum((points - (mass_y, mass_x))**2, axis=1)
            id     = np.argsort(dist)[0]
            mass_y, mass_x = points[id]
        # return
        return {'mask': m, 'area': area, 'position_x': position_x, 'position_y': position_y, 'mass_x': mass_x, 'mass_y': mass_y}
        # # show image and mask
        # I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        # plt.figure()
        # plt.imshow(I)
        # ax = plt.gca()
        # img = np.ones( (m.shape[0], m.shape[1], 3) )
        # color_mask = np.array([2.0,166.0,101.0])/255
        # for i in range(3):
        #     img[:,:,i] = color_mask[i]
        # ax.imshow(np.dstack( (img, m*0.5) ))
        # plt.show()

if __name__ == '__main__':
    refer = REFER('refcoco', splitBy='unc')
    # ann_ids = refer.getAnnIds(catIds=1)
    # print len(ann_ids)
    #
    # ref_ids = refer.getRefIds(catIds=1)
    # print len(ref_ids)
    #
    # ann_ids = refer.getAnnIds(refIds=ref_ids)
    # print len(ann_ids)

    # ref_id   = refer.getRefIds(catIds=1)[0]
    # ref      = refer.loadRefs(ref_id)[0]
    # refer.showRef(ref)
    #
    # ref_id   = refer.getRefIds(catIds=1)[1]
    # ref      = refer.loadRefs(ref_id)[0]
    # refer.showRef(ref)
    print len(refer.imgs)
    print len(refer.imgToRefs)

    ref_ids = refer.getRefIds(split='test')
    print 'There are %s test referred objects.' % len(ref_ids)

    for ref_id in refer.getRefIds(split='test'):
        ref = refer.loadRefs(ref_id)[0]
        pprint(ref)
        print 'The label is %s.' % (refer.cats[ref['category_id']])
        refer.showRef(ref)
        refer.getMask(ref)

    ref_ids = refer.getRefIds(split='test')
    print len(ref_ids)


