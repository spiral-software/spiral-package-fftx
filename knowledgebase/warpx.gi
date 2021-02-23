
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Class(WarpXOpts, FFTXConvOpts, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX WarpX options record>"))    
));

warpxOpts := function(arg) # specific to WarpX size 100...
    local opts, rfs;
    
    rfs := Copy(RulesFuncSimp);
    Unbind(rfs.rules.TensorIdId);
    
    opts := Copy(WarpXOpts);
    opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
    opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, CopyFields(PRDFT1_CT, 
            rec(allChildren := P ->Filtered(PRDFT1_CT.allChildren(P), i->When(P[1] = 100, Cols(i[1]) = 4, true)))), 
        PRDFT_PD], _noT);
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD ], _noT);
    opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);
    opts.breakdownRules.IPRDFT2 := List([ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT ], _noT);
    opts.breakdownRules.DFT := [ DFT_Base, 
        CopyFields(DFT_CT, rec(children := nt ->Filtered(DFT_CT.children(nt), i->When(nt.params[1] = 100, Cols(i[1]) = 4, true)))), 
        DFT_PD ];
    opts.breakdownRules.TTensorInd := [dsA_base, L_dsA_L_base, dsA_L_base, L_dsA_base];    
    opts.breakdownRules.MDDFT := [ MDDFT_Base, MDDFT_RowCol ];
    opts.breakdownRules.TL := [L_base];
    opts.breakdownRules.TSparseMat := [ TSparseMat_base ];
    
    opts.globalUnrolling := 23;
    opts.codegen := CopyFields( MultiPtrCodegenMixin, spiral.libgen.VecRecCodegen);
    opts.sumsgen.IterHStack := MultiPtrSumsgenMixin.IterHStack;
    opts.preProcess := t -> ApplyStrategy(t, 
                    [ RulesFFTXPromoteWarpX1, 
                      MergedRuleSet(rfs, RulesSums, RulesFFTXPromoteWarpX2), 
                      RulesFFTXPromoteNT, 
                      MergedRuleSet(rfs, RulesSums, RulesFFTXPromoteWarpX3) ],
        BUA, opts);
    opts.useDeref := false;
    # for debugging...
    opts.generateInitFunc := false;
#    opts.codegen := DefaultCodegen;
    opts.arrayBufModifier := "";
    opts.arrayDataModifier := "static";
    
    return opts;
end;


warpxConf := rec(
    defaultName := "defaultWarpXConf",
    defaultOpts := (arg) >> rec(useWarpX := true),
    confHandler := warpxOpts 
);

fftx.FFTXGlobals.registerConf(warpxConf);



