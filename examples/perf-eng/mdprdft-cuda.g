
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional real DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

sizes := [
#     [ 4, 4, 4],
#     [ 100, 100, 100],
#     [ 224, 224, 100],
#     [ 96, 96, 320],
     [ 80, 80, 80 ]
];

#fwd := true;
fwd := false;

if fwd then
    prdft := MDPRDFT;
    k := 1;
else
    prdft := IMDPRDFT;
    k := -1;
fi;


for szcube in sizes do
    var.flush();
    d := Length(szcube);
    
    name := "mdprdft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    
    PrintLine("mdprdft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

    t := TFCall(ApplyFunc(prdft, [szcube, k]), 
            rec(fname := name, params := []));
    
    opts := conf.getOpts(t);
    tt := opts.tagIt(t);
    
    c := opts.fftxGen(tt);
    opts.prettyPrint(c);
    PrintTo(name::".cu", opts.prettyPrint(c));
od;
