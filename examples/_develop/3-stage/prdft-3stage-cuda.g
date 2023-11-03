
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);
ImportAll(realdft);

Debug(true);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

#N := 256*16;
N := 256;

batch := 2;
name := "batch_dft_"::StringInt(batch)::"x"::StringInt(N);

t := TFCall(TTensorI(PRDFT1(N, -1), batch, APar, APar), rec(fname := name, params := []));
#t := TFCall(TTensorI(IPRDFT1(N, -1), batch, APar, APar), rec(fname := name, params := []));

opts := conf.getOpts(t);
#opts.breakdownRules.PRDFT[3].allChildren := P -> Filtered(PRDFT1_CT.allChildren(P), i -> i[1].params[1] <= 16);
#opts.breakdownRules.IPRDFT[3].allChildren := P -> Filtered(IPRDFT1_CT.allChildren(P), i -> i[1].params[1] <= 16);

tt := opts.tagIt(t);

# ==
_tt := opts.preProcess(tt);
rt := opts.search(_tt);
ss := opts.sumsRuleTree(rt);
c := opts.codeSums(ss);
# ==

c := opts.fftxGen(tt);
opts.prettyPrint(c);

cyc := CMeasure(c, opts);


