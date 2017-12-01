import re
import numpy as np
import scipy as sp
from collections import Counter
from scipy.sparse import csr_matrix
import cPickle as pickle
from collections import defaultdict


def clean(raw):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw)
    cleanr = re.compile('[^a-zA-Z0-9]')
    cleantext = re.sub(cleanr, ' ', cleantext)
    cleanr = re.compile(r'\W*\b\w{1,2}\b')
    cleantext = re.sub(cleanr, '', cleantext)
    return cleantext.lower()

def group(inp, n = 2):
    for i in xrange(len(inp) - (n - 1)):
        yield inp[i:i+n]

def group2words(inp):
    comb_2_words = []
    for f, s in group(inp, 2):
        comb = f + " "+s
        comb_2_words.append(comb)
    return comb_2_words

def group3words(inp):
    comb_3_words = []
    for f, s, t in group(inp, 3):
        comb = f + " "+s + " "+t
        comb_3_words.append(comb)
    return comb_3_words

def getKmers(inp):
    kmers=[]
    comb_2_words = group2words(inp)
    for comb in comb_2_words:
        kmers.append(comb)
    comb_3_words = group3words(inp)
    for comb in comb_3_words:
        kmers.append(comb)
    return kmers

def build_matrix(docs):
    r""" Build sparse matrix from a list of documents,
    each of which is a list of word/terms in the document.
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    # Remove all ratings
    for d in docs:
        #d = d[1:]
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)

    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        #d = d[1:]
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()

    return mat


# scale matrix and normalize its rows
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat

def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as infile:
        matrix = pickle.load(infile)
    return matrix