
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Import(fftx.platforms.cuda);
Import(fftx.codegen);
Import(simt);

Class(FFTCUDADeviceOpts, FFTXCUDADeviceOpts, rec(
#    tags := [ASIMTKernelFlag(ASIMTGridDimY), ASIMTBlockDimX, ASIMTLoopDim() ],
#    tags := [ASIMTGridDimX, ASIMTBlockDimX, ASIMTKernelFlag(ASIMTLoopDim()) ],
#    tags := [ASIMTBlockDimZ, ASIMTKernelFlag(ASIMTBlockDimX), ASIMTKernelFlag(ASIMTBlockDimY) ],
    tags := [ASIMTKernelFlag(ASIMTGridDimY), ASIMTGridDimX],

    operations := rec(Print := s -> Print("<FFTX FFT CUDA Device options record>"))    
));

PRDFT_PD_MAX := 17;
DFT_RADER_MIN := 17;
DFT_PD_MAX := 13;

fftCUDADeviceOpts := function(arg) # specific to FFT size 100...
    local opts;
    
    opts := Copy(FFTCUDADeviceOpts);
    opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
    opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, CopyFields(PRDFT1_CT, 
            rec(allChildren := P ->Filtered(PRDFT1_CT.allChildren(P), i->When(P[1] = 100, Cols(i[1]) = 4, true)))), 
        CopyFields(PRDFT_PD, rec(maxSize := PRDFT_PD_MAX))], _noT);
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, CopyFields(IPRDFT_PD, rec(maxSize := PRDFT_PD_MAX)) ], _noT);
    opts.breakdownRules.IPRDFT2 := List([ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT ], _noT);
    opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);
    opts.breakdownRules.DFT := [ DFT_Base, 
        CopyFields(DFT_CT, rec(children := nt ->Filtered(DFT_CT.children(nt), i->When(nt.params[1] = 100, Cols(i[1]) = 4, true)))), 
        CopyFields(DFT_PD, rec(maxSize := DFT_PD_MAX)), CopyFields(DFT_Rader, rec(minSize := DFT_RADER_MIN)) ];

    opts.breakdownRules.TTensorInd := [TTensorInd_SIMT];    
    opts.breakdownRules.TL := [L_SIMT];
    opts.breakdownRules.TIterHStack := [TIterHStack_SIMT];
    opts.breakdownRules.TIterVStack := [TIterVStack_SIMT];
    opts.breakdownRules.MDPRDFT := [MDPRDFT_tSPL_RowCol_SIMT];
    opts.breakdownRules.IMDPRDFT := [IMDPRDFT_tSPL_RowCol_SIMT];
    opts.breakdownRules.TRC := [TRC_SIMT];
    opts.breakdownRules.MDDFT := [ MDDFT_Base, CopyFields(MDDFT_tSPL_RowCol, rec(switch := true)), MDDFT_RowCol, MDDFT_tSPL_RowCol_SIMT];
    opts.breakdownRules.TTensor := [ AxI_IxB, IxB_AxI ];
    opts.breakdownRules.TTensorI := [ IxA_SIMT,  AxI_SIMT ];
    opts.breakdownRules.TSparseMat := [CopyFields(TSparseMat_base, rec(max_rows := 1)), TSparseMat_VStack];
    
    opts.globalUnrolling := 2 * Maximum([PRDFT_PD_MAX, DFT_RADER_MIN, DFT_PD_MAX]) + 1;
    opts.codegen.GathPtr := fftx.codegen.MultiPtrCodegenMixin.GathPtr;
    opts.codegen.ScatPtr := fftx.codegen.MultiPtrCodegenMixin.ScatPtr;
    opts.codegen.OO := (self, o, y, x, opts) >> skip();
    
    opts.sumsgen.IterHStack := MultiPtrSumsgenMixin.IterHStack;
    opts.preProcess := t -> ApplyStrategy(t, 
                    [ MergedRuleSet(RulesFuncSimp, RulesSums), 
                      RulesFFTXPromoteNT, RulesF2C, 
                      MergedRuleSet(RulesFuncSimp, RulesSums) ],
        BUA, opts);
        
    opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
        FixUpCUDASigmaSPL(s1, opts)); 

#    opts.useDeref := false;
    # for debugging...
#    opts.generateInitFunc := false;
#    opts.codegen := DefaultCodegen;
#    opts.arrayBufModifier := "";
#    opts.arrayDataModifier := "static const";
    opts.max_threads := 1024;
        
    return opts;
end;


confFFTCUDADevice := rec(
    defaultName := "confFFTCUDADevice",
    defaultOpts := (arg) >> rec(useFFTCUDADevice := true),
    confHandler := fftCUDADeviceOpts 
);

fftx.FFTXGlobals.registerConf(confFFTCUDADevice);



