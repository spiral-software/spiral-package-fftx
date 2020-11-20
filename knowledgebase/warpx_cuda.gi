Import(fftx.platforms.cuda);
Import(fftx.codegen);
Import(simt);

Class(WarpXCUDADeviceOpts, FFTXCUDADeviceOpts, rec(
#    tags := [ASIMTKernelFlag(ASIMTGridDimY), ASIMTBlockDimX, ASIMTLoopDim() ],
#    tags := [ASIMTGridDimX, ASIMTBlockDimX, ASIMTKernelFlag(ASIMTLoopDim()) ],
#    tags := [ASIMTBlockDimZ, ASIMTKernelFlag(ASIMTBlockDimX), ASIMTKernelFlag(ASIMTBlockDimY) ],
    tags := [ASIMTKernelFlag(ASIMTGridDimY), ASIMTGridDimX, ASIMTBlockDimZ],

    operations := rec(Print := s -> Print("<FFTX WarpX CUDA Device options record>"))    
));

warpXCUDADeviceOpts := function(arg) # specific to WarpX size 80 and 100... # the magic list [20, 80, 100] needs to be exposed from the API level
    local opts, rfs;
    
    rfs := Copy(RulesFuncSimp);
    Unbind(rfs.rules.TensorIdId);    
    Unbind(rfs.rules.Const_fbase);
    
    opts := Copy(WarpXCUDADeviceOpts);
    opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
    opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, CopyFields(PRDFT1_CT, 
            rec(allChildren := P ->Filtered(PRDFT1_CT.allChildren(P), i->When(P[1] in [20, 80, 100], Cols(i[1]) = 4, true)))), 
        PRDFT_PD], _noT);
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, CopyFields(IPRDFT1_CT,
            rec(allChildren := P ->Filtered(IPRDFT1_CT.allChildren(P), i->When(P[1] in [20, 80, 100], Cols(i[2]) = 4, true)))), 
        IPRDFT_PD], _noT);
    opts.breakdownRules.IPRDFT2 := List([ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT ], _noT);
    opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);
    opts.breakdownRules.DFT := [ DFT_Base, 
        CopyFields(DFT_CT, rec(children := nt ->Filtered(DFT_CT.children(nt), i->When(nt.params[1] in [20, 80, 100], Cols(i[1]) = 4, true)))), 
        DFT_PD ];

    opts.breakdownRules.TTensorInd := [TTensorInd_SIMT];    
    opts.breakdownRules.TL := [L_SIMT];
    opts.breakdownRules.TIterHStack := [TIterHStack_SIMT];
    opts.breakdownRules.TIterVStack := [TIterVStack_SIMT];
    opts.breakdownRules.MDPRDFT := [MDPRDFT_3D_SIMT];
    opts.breakdownRules.IMDPRDFT := [IMDPRDFT_3D_SIMT];
    opts.breakdownRules.TRC := [TRC_SIMT];
    opts.breakdownRules.MDDFT := [ MDDFT_Base, CopyFields(MDDFT_tSPL_RowCol, rec(switch := true)), MDDFT_RowCol];
    opts.breakdownRules.TTensor := [ AxI_IxB, IxB_AxI ];
    opts.breakdownRules.TTensorI := [ IxA_SIMT,  AxI_SIMT ];
    opts.breakdownRules.TSparseMat := [CopyFields(TSparseMat_base, rec(max_rows := 1)), TSparseMat_VStack];
    
    opts.globalUnrolling := 12; #23; ## 12 for size 80, 23 for size 100--needs to be exposed at the API level
    opts.codegen.GathPtr := fftx.codegen.MultiPtrCodegenMixin.GathPtr;
    opts.codegen.ScatPtr := fftx.codegen.MultiPtrCodegenMixin.ScatPtr;
    opts.codegen.OO := (self, o, y, x, opts) >> skip();
    
    opts.sumsgen.IterHStack := MultiPtrSumsgenMixin.IterHStack;
    opts.preProcess := t -> ApplyStrategy(t, 
                    [ RulesFFTXPromoteWarpX1, 
                      MergedRuleSet(rfs, RulesSums, RulesFFTXPromoteWarpX2), 
                      RulesFFTXPromoteNT, 
                      MergedRuleSet(rfs, RulesSums, RulesFFTXPromoteWarpX3) ],
        BUA, opts);
        
    opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(rfs, RulesSums, RulesSIMTFission) ], BUA, opts),
        FixUpCUDASigmaSPL(s1, opts)); 

#    opts.useDeref := false;
    # for debugging...
#    opts.generateInitFunc := false;
#    opts.codegen := DefaultCodegen;
#    opts.arrayBufModifier := "";
#    opts.arrayDataModifier := "static const";

    # lets see if that gets all the temps into device memory
    opts.max_shmem_size := 16384;
    opts.use_shmem := false;
    opts.max_threads := 1024;
    opts.fixUpZ := true;
    
    return opts;
end;


confWarpXCUDADevice := rec(
    defaultName := "confWarpXCUDADevice",
    defaultOpts := (arg) >> rec(useWarpXCUDADevice := true),
    confHandler := warpXCUDADeviceOpts 
);

fftx.FFTXGlobals.registerConf(confWarpXCUDADevice);



