
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := FFTXGlobals.defaultHIPConf();

# sizes := [
# #[130,130,130]];
# #     [ 4, 4, 4],
# #     [ 96, 96, 320],
#      [ 100, 100, 100]#,
# #     [ 224, 224, 100],
# #     [ 80, 80, 80 ],
# ];

# for szcube in sizes do
#     var.flush();

    d := Length(szcube);
    
    name := "mddft"::StringInt(d)::"d";
    sfilname := "mddft"::StringInt(d)::"d-"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    
    PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, "; func name = ", name, "; source file = ", sfilname, ";\t\t##PICKME##");
    
    if 1 = 1 then
        t := TFCall(MDDFT(szcube, 1), 
                    rec(fname := name, params := []));
    
        opts := conf.getOpts(t);
        tt := opts.tagIt(t);
    
        c := opts.fftxGen(tt);
        opts.prettyPrint(c);
        PrintTo("srcs/"::sfilname::".cu", opts.prettyPrint(c));
    fi;

# od;
