
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);
ImportAll(realdft);

Debug(true);

## ToDO
# For PRDFT bigger surgery is needed: 
# 1) upgrade CT rules to NewRulesFor() to guard against tags, and 
# 2) tspl_CT version of the PRDFT_CT rule                    


# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

N := 1024;
batch := 2;
name := "batch_dft_"::StringInt(batch)::"x"::StringInt(N);

t := TFCall(TRC(TTensorI(PRDFT1(N, -1), batch, APar, APar)), rec(fname := name, params := []));
#t := TFCall(TRC(PRDFT1(N, -1)), rec(fname := name, params := []));

opts := conf.getOpts(t);
opts.breakdownRules.PRDFT := [ PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT_, PRDFT_PD ];
opts.breakdownRules.IPRDFT := [ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT_, IPRDFT_PD ];

tt := opts.tagIt(t);

_tt := opts.preProcess(tt);
rt := opts.search(_tt);

# ==
c := opts.fftxGen(tt);
opts.prettyPrint(c);

cyc := CMeasure(c, opts);

mt := MatSPL(tt);
mc := CMatrix(c, opts);

diff := InfinityNormMat(mc-mt);

