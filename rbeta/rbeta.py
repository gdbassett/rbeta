import numpy as np
from scipy import stats # for zscore

def findboldpeaks(ts1, threshold):
    '''Get the peaks in a single TS
    
        INPUT:
            ts1: signal. a single time series array
            threshold: the z-normalized threshold used to determine peaks
            
        OUTPUT:
            a numpy array of indexes of peaks
    '''
    
    # H = signal > threshold
    H=np.heaviside(ts1-threshold,0)
    ls = np.where(H==1)
    ls = ls[0]
    ll = np.where(H[ls-1]==0)
    locs=ls[ll]
    if locs.size != 0 and locs[0]==0:
        locs = locs[1:len(locs)]
    return locs

def rbeta_events(ts1, ts2, thr=1, past=2, future=8):
    '''Get the correlated events from ts1 and ts2
    
    INPUT
      ts1: seed time series
      ts2: time series to comare to ts1
      thr: is theshold in units of standard deviations
      past and future are positive integers describing window around time to consider (in indices)

    OUTPUT
     events_ts1:  an array of seed events, sized:
                   (number of seed events) x (past+future+1)
     events_ts2:  array of target events, sized:
                   (number of seed events) x (past+future+1)
    '''
    
    events=list()
    # z-normalize seed
    ts1 = stats.zscore(ts1, ddof=1)
    # rbdim is the dimensions of A
    #rbdim = np.shape(A) # USES A
    locs=findboldpeaks(ts1,thr)
    if locs.size == 0:
        events_ts1 = events_ts2 = np.array([[],[]])
    else:
        #idx = np.where(((locs+future<=rbdim[1])&(locs-past>0))) # USES A
        idx = np.where(((locs+future<=ts1.shape[0])&(locs-past>0))) # USES A
        locs=locs[idx]-1
        T=list() # T must end up being the event indices in ts1
        for t in locs:
            T.append(np.arange(t-past,t+future+1)) 
        #storing seed event
        a = past+future+1 # length of window
        b = locs.size # the product of the dimensions of the array, so overall length.
        idx = [item for sublist in T for item in sublist]
        events_ts1 = ts1[idx].reshape(b,a) 
        ## this loop generates a list with an item for each row (other voxel) in A
        ## that row is the events corresponding to ts1
        ## we're just going to do it for the 1 time series ts2
        #for k in range(rbdim[0]): #  USES A
        ##store target events
        #    ev = A[k,np.asarray(T)].reshape(b,a) # USES A
        #    events.append(ev)   # USES A   
        events_ts2 = ts2[np.asarray(T)].reshape(b, a)    
    
    return events_ts1, events_ts2


def rbeta_corrs(events_ts1, events_ts2, pastD=4, futureD=2):
    '''Get the correlated mean and correlations between ts1 events
    and ts2 events
    
    INPUT
     events_ts1:       array of target events, sized:
                   (number of seed events) x (past+future+1)
     events_ts2:  an array of seed events, sized:
                   (number of seed events) x (past+future+1)
     pastD:       Positive integer describing the number of time 
                   slizes into the event to start correlation
     futureD:     Positive integer describing the  number of time 
                   slices from the end  of the event to end correlation
    
    OUTPUT corrs_mean, corrs
    corrs_mean:  correlation between ts1 mean rBeta and ts2
    corrs:        correlations between 
                  rBeta events, sized:
                  (number of voxels) x (number of seed events)
    '''
    
    # a = len(events_ts2)  # assumes list.  we won't have that
    b = np.shape(events_ts1)[0]
    l = np.shape(events_ts1)[1] 
    events_ts1 = events_ts1[:,pastD:l-futureD].T
    ts1_mean = events_ts1.mean(1)
    corrs=list()
    #for k in range(a): # don't need since we don't have a list of events
    #ev = events2[k]
    l = np.shape(events_ts2)[1]
    events_ts2=events_ts2[:,pastD:l-futureD]
    #print(ev.mean(0))
    #compute correlations of mean rBetas
    corr_mean = np.corrcoef(ts1_mean, events_ts2.mean(0))
    for t in range(b):
        #print(np.shape(events_seed[:,t]))
        #print(np.shape(ev[t,:]))
        ct = np.corrcoef(events_ts1[:,t].T,events_ts2[t,:])
        corrs.append(ct[0,1])
        

    return corr_mean, corrs


def rbeta_corrs_mean_only(events_ts1, events_ts2, pastD=4, futureD=2):
    '''Get just the mean. We don't need the extra loop for corrs, but
    leaving that function for Dante's implementation
    
    INPUT
     events_ts1:  an array of target events, sized:
                   (number of seed events) x (past+future+1)
     events_ts2:  an array of seed events, sized:
                   (number of seed events) x (past+future+1)
     pastD:       Positive integer describing the number of time 
                   slizes into the event to start correlation
     futureD:     Positive integer describing the  number of time 
                   slices from the end  of the event to end correlation
    
    OUTPUT (optionally: one or both variables)
     corrs_mean:  correlation between ts1 mean rBeta and ts2
    '''
    
    # a = len(events_ts2)  # assumes list.  we won't have that
    b = np.shape(events_ts1)[0]
    l = np.shape(events_ts1)[1]  # Not used
    events_ts1 = events_ts1[:,pastD:l-futureD].T
    ts1_mean = events_ts1.mean(1)
    corrs=list()
    #for k in range(a): # don't need since we don't have a list of events
    #ev = events2[k]
    l = np.shape(events_ts2)[1] # not used
    events_ts2=events_ts2[:,pastD:l-futureD]
    #print(ev.mean(0))
    #compute correlations of mean rBetas
    return np.corrcoef(ts1_mean, events_ts2.mean(0))[0, 1]

def rbeta_mean_only(ts1, ts2, pastD=4, futureD=2, thr=1, past=2, future=8):
    '''
    Calculate the rbeta correlation between two time series


    INPUT
     ts1: seed time series
     ts2: time series to comare to ts1
     thr: is theshold in units of standard deviations
     past and future are positive integers describing window around time to consider (in indices)
     pastD:       Positive integer describing the number of time 
                   slizes into the event to start correlation
     futureD:     Positive integer describing the  number of time 
                   slices from the end  of the event to end correlation

    OUTPUT
     corrs_mean:  correlation between ts1 mean rBeta and ts2
    '''
    events = rbeta_events(ts1, ts2, thr=thr, past=past, future=future)
    if events[0].size == 0:
        return 0
    else:
        return rbeta_corrs_mean_only(events[0], events[1], pastD=pastD, futureD=futureD)


def rbeta(ts1, ts2, pastD=4, futureD=2, thr=1, past=2, future=8):
    '''
    Calculate the rbeta correlation between two time series


    INPUT
     ts1: seed time series
     ts2: time series to comare to ts1
     thr: is theshold in units of standard deviations
     past and future are positive integers describing window around time to consider (in indices)
     pastD:       Positive integer describing the number of time 
                   slizes into the event to start correlation
     futureD:     Positive integer describing the  number of time 
                   slices from the end  of the event to end correlation

    OUTPUT corrs_mean, corrs
    corrs_mean:  correlation between ts1 mean rBeta and ts2
    corrs:        correlations between 
                  rBeta events, sized:
                  (number of voxels) x (number of seed events)
    '''
    events = rbeta_events(ts1, ts2, thr=thr, past=past, future=future)
    if events[0].size == 0:
        return 0
    else:
        return rbeta_corrs(events[0], events[1], pastD=pastD, futureD=futureD), 
    