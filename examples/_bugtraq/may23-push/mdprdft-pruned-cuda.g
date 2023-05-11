
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional real DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

fwd := true;
#fwd := false;

d := 3;
n := 128;
szcube := Replicate(d, n);
pat := Replicate(d, [0..n/2-1]);

if fwd then
    prdft := PrunedMDPRDFT;
    k := 1;
else
    prdft := PrunedIMDPRDFT;
    k := -1;
fi;

name := When(fwd, "", "i")::"pmdprdft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

PrintLine("prunedmdprdft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

t := TFCall(ApplyFunc(prdft, [szcube, pat, k]), 
        rec(fname := name, params := []));

opts := conf.getOpts(t);


tt := opts.tagIt(t);

Debug(true);
_tt := opts.preProcess(tt);
rt := opts.search(_tt);

tt1 := FindUnexpandableNonterminal(_tt, opts);

#s := SPLRuleTree(rt);
#ss := opts.sumsRuleTree(rt);

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));
