from itertools import product # for iterator
#import tqdm # imported conditionally below
# Multiprocessing based on this example: https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html#:~:text=The%20multiprocessing%20package%20provides%20the%20following%20sharable%20objects%3A,to%20share%20a%20matrix%2C%20we%20will%20use%20RawArray.
from multiprocessing import Pool, RawArray, get_context # parallel computation
import logging
from math import ceil # round up for tqdm total counter 
from rbeta.rbeta import rbeta_mean_only
import numpy as np
    
## First, we probably need to output pairs
# this helps us not convert a billion row np array to a list
#def yield_pairs(chunk, n):
#    i = n
#    while i < chunk.shape[0]:
#        yield([idx for idx in range(i-n,i)])
#        i += n
#    yield([idx for idx in range(i-n, chunk.shape[0])])

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(data_raw, data_shape, corrs_raw, corrs_shape):
    """ creates a dictionary of the shared memory"""
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['data_raw'] = data_raw
    var_dict['data_shape'] = data_shape
    var_dict['corrs_raw'] = corrs_raw
    var_dict['corrs_shape'] = corrs_shape


#def yield_pairs2(indices): # adds the index at the beginning of the tuple
#    for p1, p2 in product(indices, repeat=2):
#        yield(((indices.index(p1),) + p1, (indices.index(p2),) + p2))


# for some reason chunksize doesn't seem to be working, so letting the iterator do it....
#def yield_pairs3(indices, chunksize): # adds the index at the beginning of the tuple
#    i = 0
#    ret = list()
#    for p1, p2 in product(indices, repeat=2):
#        i += 1
#        ret.append(((indices.index(p1),) + p1, (indices.index(p2),) + p2))
#        if i >= chunksize:
#            yield(ret)
#            ret = list()
#            i = 0
#    yield(ret)



def yield_pairs3b(indices, chunksize): # adds the index at the beginning of the tuple
    """Yields lists of pairs of indices for correlation"""
    i = 0
    ret = list()
    for p1, p2 in product(enumerate(indices), repeat=2):
        i += 1
        ret.append(((p1[0],) + p1[1], (p2[0],) + p2[1]))
        if i >= chunksize:
            yield(ret)
            ret = list()
            i = 0
    yield(ret)



#def correlate_pairs2(pair):
#    corrs = np.frombuffer(var_dict['corrs_raw'], dtype=np.float64).reshape(var_dict['corrs_shape'])
#    data_shared = np.frombuffer(var_dict['data_raw'], dtype=np.float64).reshape(var_dict['data_shape'])
#
#    corrs[pair[0][0], pair[1][0]] = rbeta.rbeta_means_only(data_shared[pair[0][1:]], data_shared[pair[1][1:]] )



def correlate_pairs3(pairs):
    """given a list of pairs of indices, reads the indices' data and stores the resulting correlation in shared memory"""
    corrs = np.frombuffer(var_dict['corrs_raw'], dtype=np.float64).reshape(var_dict['corrs_shape'])
    data_shared = np.frombuffer(var_dict['data_raw'], dtype=np.float64).reshape(var_dict['data_shape'])

    for pair in pairs:
        #ts1 = data_shared[pair[0][1:]] # so we're referencing into a data space the function doesn't have. parallelized this may not work
        #ts2 = data_shared[pair[1][1:]] 
        #corrs[pair[0][0], pair[1][0]] = rbeta.rbeta(ts1, ts2)
        corrs[pair[0][0], pair[1][0]] = rbeta_mean_only(data_shared[pair[0][1:]], data_shared[pair[1][1:]] )


def prbeta(data, indices, chunksize=100000, pools=2, progress_bar=False):
    '''function to run rbeta correlations between a list of indices in a matrix

        INPUT
            data: a numpy array where the last dimension is a time series
            indices: a list of tuples of dimensions one less than that of the data
                     representing the voxels to compare
            chunksize: an integer representing the number of comparisons to do per
                        chunk. Chunks are sent to workers so this controls the 
                        amount of work each worker does at a time.
            pools: an integer reprersenting the number of workers to use. This should
                   be no greater than the number of threads on your computer.

       OUTPUT
            A numpy array with two dimensions the length of the input indices.
            Each cell represents the correlation of the column, seeded by the row.

    '''
    if not progress_bar:
        logging.info("Please note, this can take a LONG time. If you're concerned about if it's finishing, you may want to enable the progress bar")
    else:
        import tqdm

    # So this line alone ate like 200GB of ram v.  Yikes.  Maybe we can run it for the voxels of interest?
    # from https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    #chunk = np.array(np.meshgrid(indices, indices)).T.reshape(-1,2)
    # 671 million comparisons per run.  Computers are fast right?
    #chunk = chunk[chunk[:, 0] > chunk[:, 1]] # filter out self and duplicative pairs # removing for rbeta as it's 1-directional
    # add the output column.  330 million ish rows.
    #chunk = np.concatenate((chunk, np.zeros((chunk.shape[0], 1), dtype=chunk.dtype)), axis=1)


    # Create Shared Output 
    global corrs_raw
    corrs_shape = (len(indices), len(indices))
    corrs_raw = RawArray('d', len(indices)**2)
    corrs = np.frombuffer(corrs_raw, dtype=np.float64).reshape(corrs_shape)

        
    # Create shared memory
    global data_raw
    data_raw = RawArray('d', int(np.prod(data.shape)))
    data_shared = np.frombuffer(data_raw, dtype=np.float64).reshape(data.shape)
    np.copyto(data_shared, data) # dst, src

    # Create shared indices memory
    #global indices_raw
    #indices_dims = {}
    #indices_raw = RawArray('d', len(indices * ))
    #indices_shared = np.fromBuffer(indices_raw, dtype=np.float64)

    #chunk_share[:, 2] = np.zeros(chunk.shape[0], dtype=chunk.dtype) # wipe out correlation data
    with get_context("spawn").Pool(processes=pools, initializer=init_worker, initargs=(data_raw, data.shape, corrs_raw, corrs_shape)) as p: # define parallel pools. 
        logging.debug("Starting parallelized loop")
        if progress_bar:
            # https://github.com/tqdm/tqdm/issues/484#issuecomment-352463240
            for _ in tqdm.tqdm(
                    ## switching chunksize into the iterator because it's being ignored in imap_unordered
    #                p.imap_unordered(correlate_pairs2, 
    #                                 yield_pairs2(indices), chunksize=chunksize),
                    p.imap_unordered(correlate_pairs3, 
                                     yield_pairs3b(indices, chunksize), chunksize=1), 
                    total=np.ceil(float(len(indices))**2/chunksize), desc="chunks"):
                pass
        else:
            ## switching chunksize into the iterator because it's being ignored in imap_unordered
#            for _ in p.imap_unordered(correlate_pairs2, yield_pairs2(indices), chunksize=chunksize):
            _ = p.map(correlate_pairs2, yield_pairs2(indices), chunksize=chunksize)

    return corrs