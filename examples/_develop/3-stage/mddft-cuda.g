
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

Debug(true);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

d := 3;
#szcube :=  Replicate(d, 768);
#szcube :=  Replicate(d, 1024);
#szcube :=  Replicate(d, 1280);

szcube := [ 24, 32, 40 ];

name := "mddft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

t := TFCall(MDDFT(szcube, 1), 
        rec(fname := name, params := []));

opts := conf.getOpts(t);
Add(opts.breakdownRules.TTensorI, fftx.platforms.cuda.IxA_SIMT_peelof3);

tt := opts.tagIt(t);

_tt := opts.preProcess(tt);
rt := opts.search(_tt);
ss := opts.sumsRuleTree(rt);

ImportAll(fftx.platforms.cuda);
ss := FixUpCUDASigmaSPL(ss, opts);
c := opts.codeSums(ss);
#c := opts.fftxGen(tt);
c.ruletree:=rt;

opts.prettyPrint(c);

# does it run?
cyc := CMeasure(c, opts);


