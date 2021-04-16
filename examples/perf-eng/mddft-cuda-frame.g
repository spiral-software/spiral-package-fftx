
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

##  Script to generate code, will be driven by a size specification and will write the
##  CUDA code to a file.  The code will be compiled along with a test harness to run the
##  code, timing it against a cufft specification of the same size, and validating that
##  the results are the same for both.

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

Import(fftx.platforms.cuda);
Import(simt);

var.flush();
##  szcube is specified outside this module    szcube := sizes[i];
d := Length(szcube);

name := "mddft"::StringInt(d)::"d";
PrintLine("mddft-cuda-frame: name = ", name, ", cube = ", szcube, ", size = ",
          StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1),
                                                                    s->" x "::StringInt(s))),
          ";\t\t##PICKME##");

if 1 = 1 then
    t := TFCall(TRC(MDDFT(szcube, 1)), 
            rec(fname := name, params := []));

    opts := conf.getOpts(t);

    opts.breakdownRules.MDDFT := [fftx.platforms.cuda.MDDFT_tSPL_Pease_SIMT];
    opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
    #opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimZ, ASIMTBlockDimY, ASIMTBlockDimX];
    #opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimX];
    opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];


    _thold := 16;
    opts.globalUnrolling := 2*_thold + 1;

    opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), fftx.platforms.cuda.L_IxA_SIMT]::opts.breakdownRules.TTensorI;
    #opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, filter := e-> ForAll(e, i -> i in [8..20])))]::opts.breakdownRules.DFT;
    opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, 
                                                        filter := e-> When(e[1]*e[2] <= _thold^2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold)))]::opts.breakdownRules.DFT;

    opts.unparser.simt_synccluster := opts.unparser.simt_syncblock;

    opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
                                         FixUpCUDASigmaSPL_3Stage(s1, opts)); 


    tt := opts.tagIt(t);

    _tt := opts.preProcess(tt);
    rt := opts.search(_tt);

    ss := opts.sumsRuleTree(rt);


    c:= opts.codeSums(ss);
    c.ruletree := rt;

    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo(name::".cu", opts.prettyPrint(c));
    PrintTo(name::".rt.g", c.ruletree);

    ss := opts.sumsRuleTree(c.ruletree);
    PrintTo(name::".ss.g", ss);

    PrintTo(name::".spl.g", spl);
fi;

