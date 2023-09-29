from logadempirical.PLELog.data.Embedding import *
from logadempirical.PLELog.data.DataLoader import *
import logging
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Specify logger
logger = logging.getLogger('embedding')
logger.setLevel(logging.INFO)
dataset = 'bgl' #  bgl, tdb, spirit, hdfs
save_path = '../../dataset/bgl'
templatesDir = '../../dataset/bgl'
log_file = 'BGL.log'
logID2Temp, templates = load_templates_from_structured(templatesDir, logger, dataset,
                                                       log_file=log_file)
templateVocab = nlp_emb_mergeTemplateEmbeddings_BGL(save_path, templates, dataset, logger)

with open(os.path.join(save_path, 'templates_BGL.vec'), 'r', encoding='utf-8') as reader:
    templateVocab = {}
    line_num = 0
    for line in reader.readlines():
        if line_num == 0:
            vocabSize, embedSize = [int(x) for x in line.strip().split()]
        else:
            items = line.strip().split()
            if len(items) != embedSize + 1: continue
            template_word, template_embedding = items[0], np.asarray(items[1:], dtype=np.float64)
            for logID, temp in logID2Temp.items():
                if temp == template_word:
                    templateVocab[logID] = template_embedding
        line_num += 1
    replica_logIDs = []
    for logId in logID2Temp.keys():
        if logID not in templateVocab.keys():
            replica_logIDs.append(logID)

    for logID in replica_logIDs:
        temp = logID2Temp[logID]
        line_num = 0
        for line in reader.readlines():
            if line_num == 0:
                vocabSize, embedSize = [int(x) for x in line.strip().split()]
            else:
                items = line.strip().split()
                if len(items) != embedSize + 1: continue
                template_word, template_embedding = items[0], np.asarray(items[1:], dtype=np.float64)
                if temp == template_word:
                    templateVocab[logID] = template_embedding
            line_num += 1

with open(os.path.join(save_path, 'embeddings.json'), 'w') as writer:
    json.dump(templateVocab, writer, cls=NumpyEncoder)