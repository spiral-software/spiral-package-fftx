# 1d and multidimensional real iDFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

d := 3;
szcube := 4;
name := "imdprdft"::StringInt(d)::"d";

PrintLine("imdprdft-cuda: d = ", d, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(ns := Replicate(3, szcube),
    TFCall(IMDPRDFT(ns, 1), 
        rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".c", opts.prettyPrint(c));

