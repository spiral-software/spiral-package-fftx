
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# FFTX Baseline options

_noT := r -> CopyFields(r, rec(forTransposition := false));

Declare(ParseOpts);

Class(FFTXDefaultConf, LocalConfig, rec(
    getOpts := (self, t) >> ParseOpts(self, t),
    operations := rec(Print := s -> Print("<FFTX Default Configuration>")),
));


Class(FFTXOpts, SpiralDefaults, FFTXGenMixin, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX options record>")),
    breakdownRules := Copy(SpiralDefaults.breakdownRules),
    codegen := Copy(SpiralDefaults.codegen),
    tagIt := (self, t) >> t.withTags(self.tags)
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
                opts.breakdownRules.DFT := [DFT_Base, DFT_PD, DFT_CT, CopyFields(DFT_Rader, rec(minSize := 17))];
                opts.breakdownRules.IOPrunedMDRConv := [IOPrunedMDRConv_3D_2trip_zyx_freqdata];
                opts.breakdownRules.MDDFT := [ MDDFT_Base, MDDFT_RowCol ];
                opts.breakdownRules.GT := [ GT_NthLoop ];
                opts.breakdownRules.DFT := [ DFT_Base, DFT_CT, CopyFields(DFT_GoodThomas, rec(maxSize := 15)), DFT_PD ];
                opts.topTransforms := [DFT, MDDFT, PrunedDFT, PrunedIDFT, IOPrunedConv, PrunedPRDFT, PrunedIPRDFT, TTensorI, IOPrunedMDRConv, 
                    PRDFT, IPRDFT, PRDFT1, IPRDFT1, PRDFT2, IPRDFT2, PRDFT3, IPRDFT3, TTensorI ];
                #opts.breakdownRules.MDRConv := [ MDRConv_3D_2trip_zyx_freqdata ]; # [MDRConv_Base]
                return opts;
            end,
    ),
    defaultConf := (arg) >> FFTXDefaultConf,
    mdRConv := (arg) >> rec(mdRConv := true),
    getOpts := 
        meth(arg) 
            local opts, rfields;
            if Length(arg) >= 2 then
                rfields := Filtered(RecFields(arg[2]), i -> not IsSystemRecField(i));
                if Length(rfields) >= 1 and IsBound(arg[1].supportedConfs.(rfields[1])) then
                    return ApplyFunc(arg[1].supportedConfs.(rfields[1]), Drop(arg, 1));
                fi;
            fi;    
            opts := Copy(FFTXOpts);
            opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
            opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT, PRDFT1_PF, CopyFields(PRDFT_PD, rec(maxSize := 17)), PRDFT_Rader], _noT);
            opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, CopyFields(IPRDFT_PD, rec(maxSize := 17)), IPRDFT_Rader ], _noT);
            opts.breakdownRules.IPRDFT2 := List([ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT], _noT);
            opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);
            opts.breakdownRules.MDDFT := [ MDDFT_Base, MDDFT_RowCol ];
            opts.breakdownRules.DFT := [ DFT_Base, DFT_CT, CopyFields(DFT_GoodThomas, rec(maxSize := 15)), DFT_PD, CopyFields(DFT_Rader, rec(minSize := 17)) ];
            opts.breakdownRules.GT := [ GT_NthLoop ];
            #opts.breakdownRules.MDRConv := [ MDRConv_3D_2trip_zyx_freqdata ]; # [MDRConv_Base]
            return opts;  
        end
));


_promote1 := t >> let(t1 := RulesFFTXPromoteNT(Copy(t)), RulesFFTXPromoteNT_Cleanup(t1));


# this is a first experimental opts-deriving logic. This needs to be done extensible and properly
ParseOpts := function(conf, t)
    local tt, _tt, _conf, _opts;
    
    if conf = FFTXDefaultConf then 
        # promote with default conf rules
        tt := _promote1(Copy(t));

        if ObjId(tt) = TFCall then
            _tt := tt.params[1];
            # check for convolution
            if (ObjId(_tt) = MDRConv) or ((ObjId(_tt) = TTensorI) and (ObjId(_tt.params[1]) = MDRConv)) then 
                _conf := FFTXGlobals.mdRConv();
                _opts := FFTXGlobals.getOpts(_conf);
                return _opts;
            fi;
            # check for Hockney. This is for N=128
            if ObjId(_tt) = IOPrunedMDRConv then
                _conf := FFTXGlobals.defaultHockneyConf(rec(globalUnrolling := 16, prunedBasemaxSize := 7));
                _opts := FFTXGlobals.getOpts(_conf);
                return _opts;
            fi;
        fi;

        # check for WarpX
        _conf := FFTXGlobals.defaultWarpXConf();
        _opts := FFTXGlobals.getOpts(_conf);
        tt := _opts.preProcess(Copy(t));
        if ObjId(tt) = TFCall and ObjId(tt.params[1]) = TCompose then
            _tt := tt.params[1].params[1];
            # detect promoted WarpX
            if IsList(_tt) and Length(_tt) = 3 and List(_tt, ObjId) = [ TNoDiagPullinRight, TRC, TNoDiagPullinLeft ] then
                return _opts;
            fi;
        fi;
        # we are doing nothing special
        _opts := FFTXGlobals.getOpts();
        _opts.breakdownRules.SkewDTT := _opts.breakdownRules.SkewDTT{[1..2]};

        return _opts;
    fi;
    
    # Here we have to handle GPU configs
    Error("Don't know how to derive opts!\n");
end; 




