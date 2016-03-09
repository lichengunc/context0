__author__ = 'licheng'

'''
Parse the sentences given dataset.
The result {sent_id: parse} is saved to ROOT_DIR/cache/dataset/sentToParse.p
Each parse is composed of {'parsetree', 'text', 'words', 'dependencies'}
The parser can be downloaded at https://github.com/dasmith/stanford-corenlp-python
'''

import cPickle as pickle
import os.path as osp
import sys
import string
from multiprocessing import Process, Pool
import argparse

# set up nlp parser
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(ROOT_DIR, 'lib', 'utils'))
parser_path = osp.join(ROOT_DIR, 'lib', 'utils', 'corenlp', 'stanford-corenlp-full-2015-01-30')
from corenlp.corenlp import StanfordCoreNLP

# tmp folder
tmp_folder = osp.join(ROOT_DIR, 'cache', 'tmp')

def parse_each_sent(worker_id, refs):
    parser = StanfordCoreNLP(parser_path)
    parse_result = {}
    for ref in refs:
        for sent in ref['sentences']:
            sent_id = sent['sent_id']
            to_be_parse = sent['sent']
            parse_result[sent_id] = parser.raw_parse(to_be_parse)['sentences'][0]
            print 'mpId_%s, refId_%s, sentId_%s done.' % (worker_id, ref['ref_id'], sent['sent_id'])
    with open(osp.join(tmp_folder, 'parse_result_'+str(worker_id)+'.p'), 'w') as outfile:
        pickle.dump(parse_result, outfile)

def combine(num_workers, output):
    parse_result = {}
    for worker_id in range(num_workers):
        this_result = pickle.load(open(osp.join(tmp_folder, 'parse_result_'+str(worker_id)+'.p'), 'r'))
        parse_result.update(this_result)
    with open(output, 'w') as outfile:
        pickle.dump(parse_result, outfile)

def parse_refs(refs, batch_size, output, mode='multiprocess'):
    '''
    :param refs: [ref], where each ref contains 'ref_id', 'sentences', 'ann_id', 'image_id', etc,.
    :return: sentId_to_parse
    '''
    if mode == 'multiprocess':  # multi-thread
        jobs = []
        for worker_id, b in enumerate(range(0, len(refs), batch_size)):
            sub_refs = refs[b: min(b+batch_size, len(refs))]
            p = Process(target=parse_each_sent, args = (worker_id, sub_refs))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()
        # combine results
        combine(len(range(0, len(refs), batch_size)), output)
    else:
        parse_each_sent(0, refs)  # single-thread
        combine(1, output)

def main(params):
    DATA_DIR = osp.join(ROOT_DIR, 'data', params['dataset'])
    refs = pickle.load(open(osp.join(DATA_DIR, 'cleaned.p')))
    batch_size = len(refs)/10
    output = osp.join(ROOT_DIR, 'cache', params['dataset'], 'sentToParse.p')
    parse_refs(refs, batch_size, output, mode=params['mode'])

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description='Parse sentence using StanfordNLPparser')
    argParser.add_argument(dest='dataset', help='dataset name, e.g., refclef')
    argParser.add_argument('--mode', dest='mode', default='multiprocess', help='multiprocess/singleprocess')
    args = argParser.parse_args()
    params = vars(args)
    main(params)


