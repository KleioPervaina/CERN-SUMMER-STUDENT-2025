#
# LHC Filling Pattern information from CALS database.
#
# Relevant classes:
#   - LHCFillingPattern (fno)
#
# Created : 20.04.2020 - Ilias Efthymiopoulos
#

version = '4.0 - November 25, 2023 (IE)'

from pathlib import Path
import numpy as np
import pandas as pd
from collections import OrderedDict
import itertools 

import dask.dataframe as dd


###############################################################################
class LHCFillingPattern:
    ''' 
        Get the filling pattern info from the data    
    '''
    intensity_threshold = 2.0e10    # bunch intensity threshold to identify filled bunches
    bmode               = 'RAMP'    # cycle mode to search for filled bunches
    dt                  = pd.Timedelta('0s')    # time in bmode to fetch FBCT data
    bblr_max            = 20        # maximum BBLR encounters to consider 20*3.747 (25ns*c/2) = 74.948 m
    
    def __init__(self, fno, datadir, 
                 bunch_spacing=1):
        self.fill_number            = fno
        self.datadir                = datadir
        self.bunch_spacing          = bunch_spacing # bunch spacing in units of 25ns slots
        self.setBunchPatternAtBmode()
        self.setInjections()
        self.setHeadOn()
        self.setBunchTrains()
        self.setLongRangeEncounters()
        return

    def setBunchPatternAtBmode(self):
        _tmp = FilledBunchesAtBmode(self.fill_number, self.datadir, 
                                    LHCFillingPattern.bmode, 
                                    LHCFillingPattern.intensity_threshold)
        self.fbunchesDF     = _tmp
        self.nobunches      = {'B1': _tmp['B1']['nobunches'], 'B2' : _tmp['B2']['nobunches'] }
        self.bunches        = {'B1': _tmp['B1']['bids'], 'B2' : _tmp['B2']['bids'] }
        self.pattern        = {'B1': _tmp['B1']['pattern'], 'B2' : _tmp['B2']['pattern'] }
        self.nobunches_b1   = _tmp['B1']['nobunches']
        self.bunches_b1     = _tmp['B1']['bids']
        self.pattern_b1     = _tmp['B1']['pattern']
        self.nobunches_b2   = _tmp['B2']['nobunches']
        self.bunches_b2     = _tmp['B2']['bids']
        self.pattern_b2     = _tmp['B2']['pattern']
        return 

    def setInjections(self):
        _injb1 = FillInjectionsForBeam(self.fill_number, self.datadir, 'B1', 
                                       LHCFillingPattern.intensity_threshold)
        _injb2 = FillInjectionsForBeam(self.fill_number, self.datadir, 'B2', 
                                       LHCFillingPattern.intensity_threshold)
        self.injectionsDF = {'B1' : _injb1, 'B2' : _injb2}
        self.injectionsDF_b1    = _injb1
        self.injectionsDF_b2    = _injb2
        self.ninjections_b1     = len(_injb1)
        self.ninjections_b2     = len(_injb2)
        return
    
    def setHeadOn(self):
        _tmp = HeadOnPattern(self.pattern_b1, self.pattern_b2)
        self.headOnDF = _tmp
        self.ncolliding_ip15 = _tmp[(_tmp.ip == 'ip1') & (_tmp.beam == 'B1')].shape[0]
        self.ncolliding_ip2 = _tmp[(_tmp.ip == 'ip2') & (_tmp.beam == 'B1')].shape[0]
        self.ncolliding_ip8 = _tmp[(_tmp.ip == 'ip8') & (_tmp.beam == 'B1')].shape[0]
        self.collbid_ip15 = {'B1' : _tmp[(_tmp.ip == 'ip1') & (_tmp.beam == 'B1')].hobunch.values,
                             'B2' : _tmp[(_tmp.ip == 'ip1') & (_tmp.beam == 'B2')].hobunch.values}
        self.collbid_ip2 = {'B1' : _tmp[(_tmp.ip == 'ip2') & (_tmp.beam == 'B1')].hobunch.values,
                            'B2' : _tmp[(_tmp.ip == 'ip2') & (_tmp.beam == 'B2')].hobunch.values}
        self.collbid_ip8 = {'B1' : _tmp[(_tmp.ip == 'ip8') & (_tmp.beam == 'B1')].hobunch.values,
                            'B2' : _tmp[(_tmp.ip == 'ip8') & (_tmp.beam == 'B2')].hobunch.values}
        self.collpat_ip15 = {'B1' : bunch2pattern(self.collbid_ip15['B1']),
                             'B2' : bunch2pattern(self.collbid_ip15['B2'])}
        self.collpat_ip2 = {'B1' : bunch2pattern(self.collbid_ip2['B1']),
                             'B2' : bunch2pattern(self.collbid_ip2['B2'])}
        self.collpat_ip8 = {'B1' : bunch2pattern(self.collbid_ip8['B1']),
                             'B2' : bunch2pattern(self.collbid_ip8['B2'])}
        
        self.noncollbid_ip15 = {'B1' : np.array(list(set(self.bunches_b1) ^ set(self.collbid_ip15['B1']))),
                               'B2' : np.array(list(set(self.bunches_b2) ^ set(self.collbid_ip15['B2'])))}
        self.noncollbid_ip2 = {'B1' : np.array(list(set(self.bunches_b1) ^ set(self.collbid_ip2['B1']))),
                               'B2' : np.array(list(set(self.bunches_b2) ^ set(self.collbid_ip2['B2'])))}
        self.noncollbid_ip8 = {'B1' : np.array(list(set(self.bunches_b1) ^ set(self.collbid_ip8['B1']))),
                               'B2' : np.array(list(set(self.bunches_b2) ^ set(self.collbid_ip8['B2'])))}
        self.noncollpat_ip15 = {'B1' : bunch2pattern(self.noncollbid_ip15['B1']),
                                'B2' : bunch2pattern(self.noncollbid_ip15['B2'])}
        self.noncollpat_ip2 = {'B1' : bunch2pattern(self.noncollbid_ip2['B1']),
                                'B2' : bunch2pattern(self.noncollbid_ip2['B2'])}
        self.noncollpat_ip8 = {'B1' : bunch2pattern(self.noncollbid_ip8['B1']),
                                'B2' : bunch2pattern(self.noncollbid_ip8['B2'])}
        return 

    def setBunchTrains(self):
        self.bunchtrainsDF_b1       = BeamBunchTrains(self.bunches_b1, self.bunch_spacing)
        self.bunchtrainsDF_b2       = BeamBunchTrains(self.bunches_b2, self.bunch_spacing)
        self.bunchtrainsDF = {'B1': self.bunchtrainsDF_b1, 'B2': self.bunchtrainsDF_b2}
        return

    def setLongRangeEncounters(self):
        
        _aux = LongRangeEncounters(self.bunches_b1, self.bunches_b2, 
                                   self.pattern_b1, self.pattern_b2, 
                                   LHCFillingPattern.bblr_max)
        self.lrencountersDF = _aux
        self.lrencounters = {
            'B1' : { 'ip1' : np.zeros(3564), 'ip2' : np.zeros(3564), 'ip5' : np.zeros(3564), 'ip8' : np.zeros(3564)},
            'B2' : { 'ip1' : np.zeros(3564), 'ip2' : np.zeros(3564), 'ip5' : np.zeros(3564), 'ip8' : np.zeros(3564)}}
        
        for ib in ['B1', 'B2']: 
            for row in _aux[_aux.beam == ib.lower()].itertuples():
                self.lrencounters[ib]['ip1'][row.bid] = row.lrip1enc_no
                self.lrencounters[ib]['ip2'][row.bid] = row.lrip2enc_no
                self.lrencounters[ib]['ip5'][row.bid] = row.lrip5enc_no
                self.lrencounters[ib]['ip8'][row.bid] = row.lrip8enc_no
        
        return

    def info(self):
        print (f'''>>>>> LHC Filling pattern for fil {self.fill_number}''')
        # mypprint('name ',self.name)
        mypprint('bunch spacing ', self.bunch_spacing)
        mypprint('bunches : B1, B2', f'{self.nobunches_b1}, {self.nobunches_b2}')
        print('Collision scheme HO:')
        mypprint('bunches at IP1/5 ', self.ncolliding_ip15)
        mypprint('bunches at IP2 ', self.ncolliding_ip2)
        mypprint('bunches at IP8 ', self.ncolliding_ip8)
        mypprint('non-colliding at IP1/5', len(self.noncollbid_ip15))
        mypprint('non-colliding at IP2', len(self.noncollbid_ip2))
        mypprint('non-colliding at IP8', len(self.noncollbid_ip8))
        # mypprint('bunches per injection ', self.bunches_per_injection)
        mypprint('injections B1 : ', f' {self.ninjections_b1} {set(self.injectionsDF_b1.ninjected_bunches.values)}')
        mypprint('injections B2 : ', f' {self.ninjections_b2} {set(self.injectionsDF_b2.ninjected_bunches.values)}')
        mypprint('bunch trains B1 : ', f' {self.bunchtrainsDF_b1.shape[0]} {set(self.bunchtrainsDF_b1.nbunches.values)}' )
        mypprint('bunch trains B2 : ', f' {self.bunchtrainsDF_b2.shape[0]} {set(self.bunchtrainsDF_b2.nbunches.values)}' )

def FillInjectionSheme(fno : int, ddir : Path): 
    var = 'LHC.STATS:LHC:INJECTION_SCHEME'
    _dfy = pd.read_parquet(f'{ddir}/HX:FILLN={fno}',  columns=list(var.split(',')))
    if _dfy.empty :
        return ' ' 
    else : 
        return _dfy[var].unique()

def FilledBunchesAtBmode(fno, ddir, bmode, thres):
    fbdict = {}
    for b in ['B1', 'B2']:
        var = [f'LHC.BCTFR.A6R4.{b}:BUNCH_INTENSITY']
        fldbdf = dd.read_parquet(f'{ddir}/HX:FILLN={fno}/HX:BMODE={bmode}', columns=var).dropna().compute()
        bunch_intensity = fldbdf.iloc[0].values[0]
        filled_pattern = np.zeros(3564).astype(int)
        filled_bunches = np.where(bunch_intensity>thres)[0]
        filled_pattern[filled_bunches] = 1
        fbdict[b] = {'nobunches' : np.sum(filled_pattern).astype(int),
                     'pattern' : filled_pattern,
                     'bids' : filled_bunches}
    return fbdict

def FillInjectionsForBeam(fno, ddir, beam, threshold):
    '''
        Get the number of injections per beam for the selected fill(s)
        Returns separately the probe and physics bunch injections.

        For fills with multiple injection periods the last is considered and
        the numbers have a negative sign.

        Filling pattern from fast-BCT device: 
            'LHC.BCTFR.A6R4.B1:BUNCH_FILL_PATTERN'
            'LHC.BCTFR.A6R4.B2:BUNCH_FILL_PATTERN'
    '''
    dflist = []
    for bmode in ['INJPROB', 'INJPHYS'] : 
        var = f'LHC.BCTFR.A6R4.{beam.upper()}:BUNCH_INTENSITY'
        try : 
            _dfy = dd.read_parquet(f'{ddir}/HX:FILLN={fno}/HX:BMODE={bmode}', columns=list(var.split(','))).dropna().compute().sort_index()
            _dfy['filled_bunches'] = _dfy[var].apply(lambda r: np.array([1 if w>threshold else 0 for w in r ]))
            _dfy['no_bunches'] = _dfy['filled_bunches'].apply(np.sum)
            _dfy['injected_bunches'] = np.diff(_dfy['filled_bunches'].values,axis=0,prepend=0)
            _dfy['ninjected_bunches'] = _dfy['injected_bunches'].apply(np.sum)
            _dfy.drop(var, axis=1, inplace=True)
            _aux = _dfy[_dfy.ninjected_bunches != 0].copy()
            _aux['bmode'] = bmode
            dflist.append(_aux)
        except:
            pass
    ddff = pd.concat(dflist)
    assert np.all(np.diff(ddff.index)>0), f' >>> ERROR : injdf for {fno=} {beam=} not in order'
    return pd.concat(dflist)

def _FilledSlotsAtTime(tt):
    '''
        Obtain the list of filled BID slots at a time (as array and as filled slots with 0/1)

        Returns:
            fb1/fb2 [n]     : array with the filled bucket IDs
            b1/b2   [3564]  : arrays with the filled buckets 
            fslots          : dictionary with all the data
    '''
    # vlist = ['LHC.BQM.B1:NO_BUNCHES','LHC.BQM.B2:NO_BUNCHES','LHC.BQM.B1:FILLED_BUCKETS','LHC.BQM.B2:FILLED_BUCKETS']
    # - use the BCT data that are more reliable
    vlist = ['LHC.BCTFR.A6R4.B1:BUNCH_FILL_PATTERN','LHC.BCTFR.A6R4.B2:BUNCH_FILL_PATTERN']

    _fbct = importData.cals2pd(vlist,tt,'last')
    beam1 = _fbct.iloc[0]['LHC.BCTFR.A6R4.B1:BUNCH_FILL_PATTERN']
    beam2 = _fbct.iloc[0]['LHC.BCTFR.A6R4.B2:BUNCH_FILL_PATTERN']

    if _fbct.index.year <= 2015 : # --- it seems for 2015 the B1 and B2 had a +1 difference
        beam2 = np.roll(beam2, -1)
    b1 = np.array(beam1)
    b2 = np.array(beam2)
    fb1 = np.where(b1>0)[0]
    fb2 = np.where(b2>0)[0]

    fslots = {}
    fslots['B1'] = {}
    fslots['B2'] = {}
    fslots['B1']['Filled'] = np.array(beam1)
    fslots['B2']['Filled'] = np.array(beam2)
    fslots['B1']['FilledBID'] = fb1
    fslots['B2']['FilledBID'] = fb2
    return fb1, fb2, b1, b2, fslots

def offsetB1toB2(ip):
    offset = {'IP1':0, 'IP5':0, 'IP2':-891, 'IP8':891+3}
    return offset[ip.upper()]

def bid2pat(abid):
    bidpat = np.zeros(3564)
    bidpat[np.transpose(abid)] = 1 
    return bidpat

def pat2bid(apat, flag):
    return np.where(apat>flag)[0]

def bcollPattern(bs1, bs2):

    b1ho1, b2ho1 = headon(bs1, bs2, 'IP1')
    b1ho2, b2ho2 = headon(bs1, bs2, 'IP2')
    b1ho8, b2ho8 = headon(bs1, bs2, 'IP8')

    b1coll = bs1.copy()
    b1coll[b1ho1] += 2**1 + 2**5
    b1coll[b1ho2] += 2**2
    b1coll[b1ho8] += 2**8

    b2coll = bs2.copy()
    b2coll[b2ho1] += 2**1 + 2**5
    b2coll[b2ho2] += 2**2
    b2coll[b2ho8] += 2**8

    return b1coll, b2coll

def headonBeamPairIP(hobids, ip, beam='B1'):
    ip = ip.upper()
    iof_bcid = offsetB1toB2(ip)
    if beam == 'B2' : iof_bcid = -iof_bcid
    return np.array([(i-iof_bcid)%3564 for i in hobids])

def BeamBunchTrains(fbids, bunchspacing=1):
    btrains = group_consecutives(fbids, step=bunchspacing)
    btrainA = [x[0] for x in btrains]
    btrainZ = [x[-1] for x in btrains]
    nobunch = [len(x) for x in btrains]
    deltatr = np.roll(btrainZ,1)
    deltatr[0] = deltatr[0]-3564
    trdat = {'id':np.arange(0,len(btrains)),
            'bid_first':btrainA,
            'bid_last':btrainZ,
            'bids':btrains,
            'nbunches':nobunch,
            'gap':btrainA-deltatr}
    return pd.DataFrame(trdat)

def LongRangeEncounters(fbid_b1, fbid_b2, fpat1, fpat2, nmax):
    bidB1df = pd.DataFrame({'bid':fbid_b1, 'beam':'b1'})
    bidB2df = pd.DataFrame({'bid':fbid_b2, 'beam':'b2'})
    for ip in ['ip1', 'ip2', 'ip5', 'ip8']:
        ho1_ip, ho2_ip = headon(fpat1, fpat2, ip.upper())
        bidB1df['ho'+ip]            = bidB1df.apply(lambda row: 1 if row['bid'] in ho1_ip else 0, axis=1)
        bidB1df['lr'+ip+'enc']      = bidB1df.apply(lambda row: bidlrencounters(row['bid'], fbid_b2, ip, nmax), axis=1)
        bidB1df['lr'+ip+'enc_pos']  = bidB1df.apply(lambda row: bidlrencpos(row['lr'+ip+'enc'],nmax), axis=1)
        bidB1df['lr'+ip+'enc_no']   = bidB1df.apply(lambda row: len(row['lr'+ip+'enc_pos']), axis=1)

        bidB2df['ho'+ip]            = bidB2df.apply(lambda row: 1 if row['bid'] in ho2_ip else 0, axis=1)
        bidB2df['lr'+ip+'enc']      = bidB2df.apply(lambda row: bidlrencounters(row['bid'], fbid_b1, ip, nmax), axis=1)
        bidB2df['lr'+ip+'enc_pos']  = bidB2df.apply(lambda row: bidlrencpos(row['lr'+ip+'enc'],nmax), axis=1)
        bidB2df['lr'+ip+'enc_no']   = bidB2df.apply(lambda row: len(row['lr'+ip+'enc_pos']), axis=1)
    return pd.concat([bidB1df, bidB2df])

def bidlrencounters(bid, bid2, ip, nmax):
    offset = -1*offsetB1toB2(ip)
        
    lr_left = np.zeros(nmax)
    lr_right = np.zeros(nmax)
    for j in np.arange(1, nmax):
        if (bid+offset+j)%3564 in bid2 :
            lr_right[j]= 1
        if (bid+offset-j)%3564 in bid2 :
            lr_left[j] = 1
    ip_left = lr_left[1:]
    ip_right = lr_right[1:]
    return np.concatenate((ip_left[::-1],ip_right))

def bidlrencpos(enc, nmax):
    spos = np.concatenate(((np.arange(1,nmax)*-1)[::-1], np.arange(1,nmax)))
    _enc = np.where(enc>0)[0]
    return spos[_enc]

def headon(bpatb1, bpatb2, ip):
    ip = ip.upper()
    iof_bcid = offsetB1toB2(ip)
    
    bpatb2a = np.roll(bpatb2, iof_bcid)
    tmp1 = bpatb1 + bpatb2a
    hob1 = pat2bid(tmp1,1)
    
    bpatb1a = np.roll(bpatb1, -iof_bcid)
    tmp2 = bpatb1a + bpatb2
    hob2 = pat2bid(tmp2, 1)
    return hob1, hob2

def cflagID():
    ips = [1, 2, 5, 8]
    d2 = {}
    _cflagdict = {}
    for i in [1,2,3,4]:
        for j in itertools.combinations(ips,i):
            key = ''
            value = 1
            for k in j:
                key += 'ip'+str(k)+'-'
                value += 2**k
            key = key[:-1]
            _cflagdict[key] = value
    _cflagdict['nc'] = 1
    cflagID = OrderedDict(sorted(_cflagdict.items(), key=lambda t: t[1]))
    cflagIDinv = dict(map(reversed, cflagID.items()))
    return cflagID, cflagIDinv

def HeadOnPattern(bpatb1, bpatb2):
    ip = ['ip1', 'ip2','ip5','ip8']
    ipflag = {'ip1':2**1, 'ip2':2**2, 'ip5':2**5, 'ip8':2**8}
    
    cpattB1AllIPs, cpattB2AllIPs = bcollPattern(bpatb1, bpatb2)

    dflist = []
    for i in ip:
        hob1, hob2 = headon(bpatb1, bpatb2, i)
        hob1p = headonBeamPairIP(hob1, i, beam='B1')
        hob2p = headonBeamPairIP(hob2, i, beam='B2')
        
        _b1  = [cpattB1AllIPs[j] for j in hob1]
        _b1p = [cpattB2AllIPs[j] for j in hob1p]
        _aux = pd.DataFrame({'hobunch':hob1, 
                             'hopartner':hob1p,
                             'cflag':_b1,
                             'cflagp':_b1p})
        _aux['beam'] = 'B1'
        _aux['ip'] = i
        dflist.append(_aux)
        
        _b2  = [cpattB2AllIPs[j] for j in hob2]
        _b2p = [cpattB1AllIPs[j] for j in hob2p]
        _aux = pd.DataFrame({'hobunch':hob2,
                             'hopartner':hob2p,
                             'cflag':_b2,
                             'cflagp':_b2p})
        _aux['beam'] = 'B2'
        _aux['ip'] = i

        dflist.append(_aux)

    hoPatDF = pd.concat(dflist)

    flagID, flagIDinv = cflagID()

    hoPatDF['cflagID']  = hoPatDF['cflag'].apply(lambda x : flagIDinv[int(x)])
    hoPatDF['cflagIDp'] = hoPatDF['cflagp'].apply(lambda x : flagIDinv[int(x)])

    return hoPatDF

def flatten(l):
    ''' Flattens a list of lists to a single list '''
    return [item for sublist in l for item in sublist]

def check_bunch_classes(fillpat, bclasses=[8, 36]):
    ''' Check remaining bunches outside the defined classes '''
    _allb1 = fillpat.pattern_b1
    _allb2 = fillpat.pattern_b2
    for c in bclasses: 
        _xx = flatten(fillpat.bunchtrainsDF_b1[fillpat.bunchtrainsDF_b1.nbunches==c].bids)
        _aa = np.zeros(3564)
        _aa[_xx] = 1
        _allb1 = np.subtract(_allb1, _aa)
    
        _xx = flatten(fillpat.bunchtrainsDF_b2[fillpat.bunchtrainsDF_b2.nbunches==c].bids)
        _aa = np.zeros(3564)
        _aa[_xx] = 1
        _allb2 = np.subtract(_allb2, _aa)
    _resb1 = np.sum(_allb1).astype(int)
    _presb1 = np.where(_allb1>0)[0].astype(int)
    
    _resb2 = np.sum(_allb2).astype(int)
    _presb2 = np.where(_allb2>0)[0].astype(int)
    
    return (_resb1, _presb1), (_resb2, _presb2)

def commonel(list1, list2):
	'''
		Return list of common elements in the two input lists
	'''
	return [element for element in list1 if element in list2]

def group_consecutives(vals, step=1):
	'''Return list of consecutive lists of numbers from vals (number list) '''
	run = []
	result = [run]
	expect = None
	for v in vals:
		if (v == expect) or (expect is None):
			run.append(v)
		else:
			run = [v]
			result.append(run)
		expect = v + step
	return result

def bucket2slot(buck):
	'''
		LHC bucket to slot number converter
	'''
	return [(x-1)/10 for x in buck]

def slot2bucket(slot):
	'''
		LHC slot to bucket number converter
	'''
	return [x*10+1 for x in slot]

def bunch2pattern(x):
    _tmp = np.zeros(3564)
    _tmp[x] = 1
    return _tmp
 
mypprint = lambda txt,val : print (f'''{txt:_<35s} {val}''')
