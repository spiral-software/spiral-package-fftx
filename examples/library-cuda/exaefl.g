
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.confGPU();

szcube := [64, 64, 64];

prdft := MDPRDFT(szcube, 1);
iprdft := IMDPRDFT(szcube, -1);

symvar := var("amplitudes", TPtr(TReal));
name := "exaefl_kernel1";

domain := 2*Product(DropLast(szcube, 1))* (Last(szcube)/2+1);
i := Ind(domain);
ampli := FDataOfs(symvar, domain, 0);

func := Lambda(i, cond(eq(i, V(0), 1, ampli.at(i))));

t := TFCall(IMDPRDFT(szcube, 1) * RCDiag(func) * MDPRDFT(szcube, -1), 
    rec(fname := name, params := [symvar]));

opts := conf.getOpts(t);
tt := opts.tagIt(t);

#_tt := opts.preProcess(tt);

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));
