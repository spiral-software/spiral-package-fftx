# Batch Real nD convolution

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();


sizes := [
     [ 4, 4, 4],
     [ 80, 80, 80 ],
     [ 100, 100, 100],
     [ 224, 224, 100],
     [ 96, 96, 320],
];

for szcube in sizes do
    var.flush();
    d := Length(szcube);
    
    name := "mdrconv"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    PrintLine("mdrconv-cuda-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

    symvar := var("sym", TPtr(TReal));

    t := TFCall(IMDPRDFT(szcube, 1) * RCDiag(FDataOfs(symvar, 2*Product(DropLast(szcube, 1))* (Last(szcube)/2+1), 0)) * MDPRDFT(szcube, -1), 
            rec(fname := name, params := [symvar])
    );

    opts := conf.getOpts(t);
    tt := opts.tagIt(t);
    c := opts.fftxGen(tt);
    opts.prettyPrint(c);
    
    PrintLine("mdrconv-cuda: codegen test only (no compiled test with 'symbol')\t\t##PICKME##");
od;
