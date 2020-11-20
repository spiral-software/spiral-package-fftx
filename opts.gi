# FFTX Baseline options

_noT := r -> CopyFields(r, rec(forTransposition := false));

Class(FFTXOpts, SpiralDefaults, FFTXGenMixin, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX options record>")),
    breakdownRules := Copy(SpiralDefaults.breakdownRules),
    codegen := Copy(SpiralDefaults.codegen)
));

FFTXOpts.codegen.GathPtr := fftx.codegen.MultiPtrCodegenMixin.GathPtr;
FFTXOpts.codegen.ScatPtr := fftx.codegen.MultiPtrCodegenMixin.ScatPtr;
FFTXOpts.codegen.OO := (self, o, y, x, opts) >> skip();
FFTXOpts.sumsgen.IterHStack := MultiPtrSumsgenMixin.IterHStack;

Class(FFTXConvOpts, FFTXOpts, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX Conv options record>")),    
    breakdownRules := Copy(SpiralDefaults.breakdownRules),
    codegen := CopyFields(MultiPtrCodegenMixin, SpiralDefaults.codegen)
));


Class(FFTXGlobals, rec(
    registerConf := meth(arg)
        local self, key;
        self := arg[1];
        self.(arg[2].defaultName) := arg[2].defaultOpts;
        key := Filtered(RecFields(arg[2].defaultOpts()), i -> not IsSystemRecField(i));
        self.supportedConfs.(key[1]) := arg[2].confHandler;
    end,
    supportedConfs := rec(
        mdRConv := 
            function(arg)
                local opts;
                opts := Copy(FFTXConvOpts);
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
                return opts;
            end,
    ),
    defaultConf := (arg) >> rec(),
    mdRConv := (arg) >> rec(mdRConv := true),
    getOpts := 
        meth(arg) 
            local opts, rfields;
            if Length(arg) >= 2 then
                rfields := Filtered(RecFields(arg[2]), i -> not IsSystemRecField(i));
                if Length(rfields) = 1 and IsBound(arg[1].supportedConfs.(rfields[1])) then
                    return ApplyFunc(arg[1].supportedConfs.(rfields[1]), Drop(arg, 1));
                fi;
            fi;    
            opts := Copy(FFTXOpts);
            opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
            opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT, PRDFT1_PF, PRDFT_PD, PRDFT_Rader], _noT);
            opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD, IPRDFT_Rader ], _noT);
            opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);
            opts.breakdownRules.MDDFT := [ MDDFT_Base, MDDFT_RowCol ];
            opts.breakdownRules.DFT := [ DFT_Base, DFT_CT, DFT_Rader, CopyFields(DFT_GoodThomas, rec(maxSize := 15)), DFT_PD ];
            opts.breakdownRules.GT := [ GT_NthLoop ];
            opts.breakdownRules.MDRConv := [ MDRConv_3D_2trip_zyx_freqdata ]; # [MDRConv_Base]
            return opts;  
        end
));



