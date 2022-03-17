
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.confGPU();

szcube := [48, 48, 48];
#szcube := [81, 81, 81];

symvar := var("amplitudes", TPtr(TReal));
name := "exaefl_kernel1";
domain := MDPRDFT(szcube, -1).dims()[1];

t := TFCall(IMDPRDFT(szcube, 1) * ExaFEL_Pointwise(domain, symvar) * MDPRDFT(szcube, -1), 
    rec(fname := name, params := [symvar]));

opts := conf.getOpts(t);
tt := opts.tagIt(t);

_tt := opts.preProcess(tt);
rt := opts.search(_tt);

Debug(true);
ss := opts.sumsRuleTree(rt);

pp := Collect(ss, Pointwise)[1];
pp.free();

#--
#c := opts.codeSums(ss);
c := opts.fftxGen(tt);

opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));




