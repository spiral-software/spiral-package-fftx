Class(HockneyOpts, FFTXConvOpts, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX Hockney options record>"))    
));

hockneyOpts := function(arg) 
    local opts, maxSize, argrec;
    opts := Copy(FFTXConvOpts);

    argrec := rec(
        globalUnrolling := opts.globalUnrolling, 
        prunedBasemaxSize := opts.globalUnrolling / 2
    );

    if IsBound(arg[1].args) and Length(arg[1].args) >= 2 and IsRec(arg[1].args[2]) then
           argrec := CopyFields(argrec, arg[1].args[2]);
    fi;
    opts.globalUnrolling := argrec.globalUnrolling;
   
    opts.useDeref := false;
    opts.breakdownRules.PRDFT := List([ PRDFT1_Base2, PRDFT1_CT, PRDFT_PD], _noT);
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD], _noT);
    opts.breakdownRules.IPRDFT2 := List([ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT], _noT);
    opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT, PRDFT3_OddToPRDFT1], _noT);
    opts.breakdownRules.URDFT := List([ URDFT1_Base1, URDFT1_Base2, URDFT1_Base4, URDFT1_CT ], _noT);
    opts.breakdownRules.DFT := [DFT_Base, DFT_PD, DFT_CT];
    opts.breakdownRules.IOPrunedMDRConv := [IOPrunedMDRConv_3D_2trip_zyx_freqdata];
    opts.breakdownRules.MDDFT := [ MDDFT_Base, MDDFT_RowCol ];
    opts.breakdownRules.GT := [ GT_NthLoop ];
    opts.breakdownRules.DFT := [ DFT_Base, DFT_CT, DFT_Rader, CopyFields(DFT_GoodThomas, rec(maxSize := 15)), DFT_PD ];
    opts.topTransforms := [DFT, MDDFT, PrunedDFT, PrunedIDFT, IOPrunedConv, PrunedPRDFT, PrunedIPRDFT, TTensorI, IOPrunedMDRConv, 
        PRDFT, IPRDFT, PRDFT1, IPRDFT1, PRDFT2, IPRDFT2, PRDFT3, IPRDFT3, TTensorI ];
    opts.breakdownRules.MDRConv := [ MDRConv_3D_2trip_zyx_freqdata ]; # [MDRConv_Base]

    opts.breakdownRules.PrunedPRDFT := [ CopyFields(PrunedPRDFT_base, rec(maxSize := argrec.prunedBasemaxSize)), PrunedPRDFT_CT_rec_block ];
    opts.breakdownRules.PrunedIPRDFT := [ CopyFields(PrunedIPRDFT_base, rec(maxSize := argrec.prunedBasemaxSize)), PrunedIPRDFT_CT_rec_block ];
    opts.breakdownRules.PrunedDFT := [ CopyFields(PrunedDFT_base, rec(maxSize := argrec.prunedBasemaxSize)), PrunedDFT_CT_rec_block ];

    return opts;
end;


hockneyConf := rec(
    defaultName := "defaultHockneyConf",
    defaultOpts := (arg) >> rec(useHockney := true, args := arg),
    confHandler := hockneyOpts 
);

fftx.FFTXGlobals.registerConf(hockneyConf);



