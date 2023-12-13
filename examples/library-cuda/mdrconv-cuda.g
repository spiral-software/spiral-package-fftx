
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional real DFTs

Load(fftx);
ImportAll(fftx);

Debug(true);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

n := 189;
d := 3;
szcube :=  Replicate(d, n);
    
name := "mdrconv"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

symvar := var("sym", TPtr(TReal));

t := TFCall(IMDPRDFT(szcube, 1) * RCDiag(FDataOfs(symvar, MDPRDFT(szcube, -1).dims()[1], 0)) * MDPRDFT(szcube, -1), 
        rec(fname := name, params := [symvar])
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

# _tt := opts.preProcess(tt);
# rt := opts.search(_tt);
# sp := SPLRuleTree(rt);
# 
# ss := opts.sumsRuleTree(rt);
# c := opts.codeSums(ss);
# 




c := opts.fftxGen(tt);
opts.prettyPrint(c);

