
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

sizes := [
     [ 4, 4, 4],
     [ 96, 96, 320],
     [ 100, 100, 100],
     [ 224, 224, 100],
     [ 80, 80, 80 ],
];

i := 1;
szcube := sizes[i];
var.flush();
d := Length(szcube);

name := "mddft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

t := TFCall(MDDFT(szcube, 1), 
    rec(fname := name, params := []));

opts := conf.getOpts(t);

Import(fftx.platforms.cuda);
Import(simt);

opts.breakdownRules.MDDFT := [fftx.platforms.cuda.MDDFT_tSPL_Pease_SIMT];
#opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimZ, ASIMTBlockDimY, ASIMTBlockDimX];

opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimX];
opts.tags := [ASIMTKernelFlag(ASIMTGridDimX)];


opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), fftx.platforms.cuda.L_IxA_SIMT]::opts.breakdownRules.TTensorI;
opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true))]::opts.breakdownRules.DFT;

tt := opts.tagIt(t);

_tt := opts.preProcess(tt);
rt := opts.search(_tt);

xx := FindUnexpandableNonterminal(_tt, opts);

spl := SPLRuleTree(rt);

tm := MatSPL(_tt);
sm := MatSPL(spl);
InfinityNormMat(tm-sm);


c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));
