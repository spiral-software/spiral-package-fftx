
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

sizes := [
#     [ 96, 96, 320],
#     [ 100, 100, 100],
#     [ 224, 224, 100],
#     [270, 270, 270],
#     [272, 272, 272],
   [ 80, 96, 64 ]
];

#for szcube in sizes do
#    var.flush();
    szcube := sizes[1];
    d := Length(szcube);
    
    name := "mddftf"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    
    PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");
    
    t := TFCallF(TRC(MDDFT(szcube, 1)), 
            rec(fname := name, params := [],
                XType := TArrayNDF(TComplex, szcube), YType := TArrayNDF(TComplex, szcube)));
    
    opts := conf.getOpts(t);
    tt := opts.tagIt(t);
    
    _tt := opts.preProcess(tt);
    
    c := opts.fftxGen(tt);
    opts.prettyPrint(c);
    #CMeasure(c, opts);
    PrintTo(name::".cu", opts.prettyPrint(c));
#od;
