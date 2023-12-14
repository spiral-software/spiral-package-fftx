
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

#sizes := [
#     [ 96, 96, 320],
#     [ 100, 100, 100],
#     [ 224, 224, 100],
#     [270, 270, 270],
#     [272, 272, 272],
szcube :=    [648, 648, 648 ];

# for szcube in sizes do
#     var.flush();
d := Length(szcube);

name := "mddft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

t := TFCall(MDDFT(szcube, 1), 
        rec(fname := name, params := []));

opts := conf.getOpts(t);
tt := opts.tagIt(t);

_tt := opts.preProcess(tt);
rt := opts.search(_tt);
ss := opts.sumsRuleTree(rt);
c := opts.codeSums(ss);

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));
#od;

CMeasure(c, opts);

