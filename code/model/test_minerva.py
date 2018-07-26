from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import time
import os
import logging
import numpy as np
import tensorflow as tf
from options import read_options
from environment import env
import codecs
from collections import defaultdict
import gc
import resource
import sys
from minerva_api import Minerva_Triple_Api
import codecs

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# read command line options
options = read_options()
# Set logging
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
logfile = logging.FileHandler(options['log_file_name'], 'w')
logfile.setFormatter(fmt)
logger.addHandler(logfile)
# read the vocab files, it will be used by many classes hence global scope
logger.info('reading vocab files...')
options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
logger.info('Reading mid to name map')
mid_to_word = {}
# with open('/iesl/canvas/rajarshi/data/RL-Path-RNN/FB15k-237/fb15k_names', 'r') as f:
#     mid_to_word = json.load(f)
logger.info('Done..')
logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))

minerva = Minerva_Triple_Api(options)
#minerva.train_api(options)
save_path = "/data/xuht/minerva/output/umls/4ae0_2_0.05_50_0.05/model/model.ckpt"
path_logger_file = "/data/xuht/minerva"
#os.mkdir(path_logger_file + "/" + "test_beam")
minerva.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
minerva.test_api(options, restore=save_path)
paths = minerva.infer_api(options, restore=save_path, input_triples=[{"source_entity":"tissue",
                                  "relation":"consists_of"}
                                   ])
print(paths)
with codecs.open("/data/xuht/minerva/infer_test.txt", "w", "utf-8") as fwobj:
    for q in paths:
        for p in paths[q]:
            fwobj.write(p)

        