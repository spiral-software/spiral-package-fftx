
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.confGPU();

szcube := [64, 64, 64];

symvar := var("amplitudes", TPtr(TReal));
name := "exaefl_kernel1";
domain := 2*Product(DropLast(szcube, 1))* (Last(szcube)/2+1);

t := TFCall(IMDPRDFT(szcube, 1) * ExaFEL_Pointwise(domain, symvar) * MDPRDFT(szcube, -1), 
    rec(fname := name, params := [symvar]));

opts := conf.getOpts(t);
tt := opts.tagIt(t);

#_tt := opts.preProcess(tt);
#rt := opts.search(_tt);
#ss := opts.sumsRuleTree(rt);
#--
#c := opts.codeSums(ss);
c := opts.fftxGen(tt);






opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));




