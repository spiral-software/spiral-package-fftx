
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

i := 2;
szcube := sizes[i];
var.flush();
d := Length(szcube);

name := "mddft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

t := TFCall(TRC(MDDFT(szcube, 1)), 
    rec(fname := name, params := []));

opts := conf.getOpts(t);

Import(fftx.platforms.cuda);
Import(simt);

opts.breakdownRules.MDDFT := [fftx.platforms.cuda.MDDFT_tSPL_Pease_SIMT];
opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
#opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimZ, ASIMTBlockDimY, ASIMTBlockDimX];
#opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimX];
opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];

opts.globalUnrolling := 33;

opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), fftx.platforms.cuda.L_IxA_SIMT]::opts.breakdownRules.TTensorI;
opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, filter := e-> ForAll(e, i -> i in [10, 14, 16])))]::opts.breakdownRules.DFT;

opts.unparser.simt_synccluster := opts.unparser.simt_syncblock;

tt := opts.tagIt(t);

_tt := opts.preProcess(tt);
rt := opts.search(_tt);

##xx := FindUnexpandableNonterminal(_tt, opts);
#
spl := SPLRuleTree(rt);
ss := opts.sumsRuleTree(rt);
#c:= opts.codeSums(ss);
#opts.prettyPrint(c);
#
#
#
#tm := MatSPL(_tt);
#sm := MatSPL(spl);
#InfinityNormMat(tm-sm);
#

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));

PrintTo(name::".rt.g", c.ruletree);

ss := opts.sumsRuleTree(c.ruletree);
PrintTo(name::".ss.g", ss);

PrintTo(name::".spl.g", spl);


##-------------------
#224 x 224 x 100
#
#10 x 10 x 224 x 224
#
#kernel(10) * warp(10) x groupY(8) x  block(16 x 14 x 28) 
#10*10*8* 16*2*2 = 50 kB shared memory
#
#
#14 x 16 x 100 x 224
#
#kernel(14) x warp(16) x groupX(5) x block(20 x 14 x 16)
#14*5*16 * 2*16*2 = 70 kB
#
#kernel(16) x warp(14) x groupX(5) x block(20 x 14 x 16)
#
#
#DFT(100 x I(50176)
#
#n := 100;
#m := 50176;
#mem := 1024*96;
#mem_per_pt := 2*8*2*2;
#max_threads := 2048;
#
#_peelof := (n,m) -> Maximum(Filtered(mem_per_pt * Filtered(n*DivisorsInt(m), e-> e<max_threads), f -> f < mem))/(mem_per_pt*n);
#
#_peelof(100, 224*224);
#
#_peelof(224, 100*224);
#
#
#
#
#DFT(224) x I(22400)
