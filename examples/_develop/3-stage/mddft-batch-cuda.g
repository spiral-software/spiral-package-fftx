
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Load(fftx);
ImportAll(fftx);
Import(simt);

Debug(true);


# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

nbatch := 2;
szcube := 768;
d := 3;
ns := Replicate(d, szcube);

PrintLine("mddft-batch-cuda: batch = ", nbatch, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(batch := nbatch,
    apat := When(true, APar, AVec),
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d_batch",  
    TFCall(TRC(TTensorI(MDDFT(ns, k), batch, apat, apat)), 
        rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

_tt := opts.preProcess(tt);
rt := opts.search(_tt);
#ss := opts.sumsRuleTree(rt);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

# does it run?
cyc := CMeasure(c, opts);


