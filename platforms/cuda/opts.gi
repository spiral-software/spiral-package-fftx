
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Import(simt);

Class(FFTXCUDAOpts, FFTXOpts, simt.TitanVDefaults, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX CUDA options record>")),    
    max_threads := 2048,
    max_blocks := 1024,
    max_heap := 1024 * 1024 * 1024
));

cudaOpts := function(arg)
    local opts;
    opts := Copy(FFTXCUDAOpts);
    opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
    opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT, PRDFT1_PF, PRDFT_PD, PRDFT_Rader], _noT);
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD, IPRDFT_Rader ], _noT);
    opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);
    return opts;  

    return opts;
end;

Declare(ParseOptsCUDA);

Class(FFTXCUDADefaultConf, rec(
    getOpts := (self, t) >> ParseOptsCUDA(self, t),
    operations := rec(Print := s -> Print("<FFTX CUDA Default Configuration>")),
    useCUDA := true
));

Class(FFTXCUDADeviceDefaultConf, rec(
    getOpts := (self, t) >> ParseOptsCUDA(self, t),
    operations := rec(Print := s -> Print("<FFTX CUDA Device Default Configuration>")),
    useCUDADevice := true
));


cudaConf := rec(
    defaultName := "defaultCUDAConf",
    defaultOpts := (arg) >> FFTXCUDADefaultConf,
    devFunc := true,
    confHandler := cudaOpts 
);

fftx.FFTXGlobals.registerConf(cudaConf);

getTargetOS := function()
    local tgt;
    
    if LocalConfig.osinfo.isWindows() then
        tgt := "win-x64-cuda";
    elif LocalConfig.osinfo.isLinux() then
        tgt := "linux-cuda";
    elif LocalConfig.osinfo.isDarwin() then
        tgt := "linux-cuda";    ## may work
    fi;
    return tgt;
end;

#--
Class(FFTXCUDADeviceOpts, FFTXCUDAOpts, simt.TitanVDefaults, rec(
    tags := [],
    devFunc := true,
    target := rec ( name := getTargetOS() ),
    operations := rec(Print := s -> Print("<FFTX CUDA Device options record>"))    
));

cudaDeviceOpts := function(arg) # specific to WarpX size 100...
    local opts;
    opts := Copy(FFTXCUDADeviceOpts);
    opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
    opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, CopyFields(PRDFT1_CT, 
            rec(allChildren := P ->Filtered(PRDFT1_CT.allChildren(P), i->When(P[1] = 100, Cols(i[1]) = 4, true)))), 
        PRDFT_PD], _noT);
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD ], _noT);
    opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);
    opts.breakdownRules.DFT := [ DFT_Base, 
        CopyFields(DFT_CT, rec(children := nt ->Filtered(DFT_CT.children(nt), i->When(nt.params[1] = 100, Cols(i[1]) = 4, true)))), 
        DFT_PD, CopyFields(DFT_Rader, rec(minSize := 17)) ];
    opts.breakdownRules.TTensorInd := [dsA_base, L_dsA_L_base, dsA_L_base, L_dsA_base];    
    return opts;
end;


cudaDeviceConf := rec(
    defaultName := "defaultCUDADeviceConf",
    defaultOpts := (arg) >> FFTXCUDADeviceDefaultConf,
    confHandler := cudaDeviceOpts 
);

fftx.FFTXGlobals.registerConf(cudaDeviceConf);


# this is a first experimental opts-deriving logic. This needs to be done extensible and properly
ParseOptsCUDA := function(conf, t)
    local tt, _tt, _tt2, _conf, _opts, _HPCSupportedSizesCUDA, _thold,
    MAX_KERNEL, MAX_PRIME, MIN_SIZE, MAX_SIZE, size1, filter;
    
    # all dimensions need to be inthis array for the high perf MDDFT conf to kick in for now
    # size 320 is problematic at this point and needs attention. Need support for 3 stages to work first
    MAX_KERNEL := 23;
    MAX_PRIME := 17;
    MIN_SIZE := 32;
    MAX_SIZE := 680;

    _thold := MAX_KERNEL;
    filter := (e) -> When(e[1] * e[2] <= _thold ^ 2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold);
    size1 := Filtered([MIN_SIZE..MAX_SIZE], i -> ForAny(DivisorPairs(i), filter) and ForAll(Factors(i), j -> not IsPrime(j) or j <= MAX_PRIME));
    _HPCSupportedSizesCUDA := size1;

#    _HPCSupportedSizesCUDA := [80, 96, 100, 224, 320];
#    _thold := 16;
    
    if IsBound(conf.useCUDADevice) then 
#        # detect real MD convolution
#        _tt := Collect(t, RCDiag)::Collect(t, MDPRDFT)::Collect(t, IMDPRDFT)::Collect(t, TTensorI);
#        if Length(_tt) = 4 then
#            _conf := FFTXGlobals.confWarpXCUDADevice();
#            _opts := FFTXGlobals.getOpts(_conf);        
#            return _opts;
#        fi;        

        # detect batch of DFT/PRDFT
        if ((Length(Collect(t, TTensorInd)) >= 1) or let(lst := Collect(t, TTensorI), (Length(lst) >= 1) and ForAll(lst, l->l.params[2] > 1))) and 
            ((Length(Collect(t, DFT)) = 1) or (Length(Collect(t, PRDFT)) = 1) or (Length(Collect(t, IPRDFT)) = 1)) then
            _conf := FFTXGlobals.confBatchFFTCUDADevice();
            _opts := FFTXGlobals.getOpts(_conf);

            # opts for high performance CUDA cuFFT
            if ForAll(Flat(List(Collect(t, @(1, [DFT, PRDFT, IPRDFT])), j-> j.params[1])), i -> i in _HPCSupportedSizesCUDA)  then
                _opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
                _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];
                
                _opts.globalUnrolling := 2*_thold + 1;

                _opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), 
                    fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT]::_opts.breakdownRules.TTensorI;
                _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, 
                    filter := e-> When(e[1]*e[2] <= _thold^2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold)))]::_opts.breakdownRules.DFT;
                
                _opts.unparser.simt_synccluster := _opts.unparser.simt_syncblock;
                _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
                    When(Collect(t, PRDFT)::Collect(t, IPRDFT) = [], 
                        FixUpCUDASigmaSPL(FixUpCUDASigmaSPL_3Stage(s1, opts), opts),
                        FixUpCUDASigmaSPL_3Stage_Real(s1, opts))); 
                _opts.postProcessCode := (c, opts) -> FixUpTeslaV_Code(c, opts);    
#                _opts.postProcessCode := (c, opts) -> FixUpTeslaV_Code(PingPong_3Stages(c, opts), opts);    
                _opts.fixUpTeslaV_Code := true;

                _opts.operations.Print := s -> Print("<FFTX CUDA HPC Batch DFT options record>");

            fi;
            return _opts;
        fi;
       
        # detect 3D DFT/Batch DFT
        _tt := Collect(t, MDDFT)::Collect(t, MDPRDFT)::Collect(t, IMDPRDFT);
        if Length(_tt) = 1 and Length(_tt[1].params[1]) = 3 then
            _conf := FFTXGlobals.confFFTCUDADevice();
            _opts := FFTXGlobals.getOpts(_conf);
#                Error();

            # opts for high performance CUDA cuFFT
            if ForAll(_tt[1].params[1], i-> i in _HPCSupportedSizesCUDA) then
                _opts.breakdownRules.MDDFT := [fftx.platforms.cuda.MDDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.MDPRDFT := [fftx.platforms.cuda.MDPRDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.IMDPRDFT := [fftx.platforms.cuda.IMDPRDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
                
                _opts.globalUnrolling := 2*_thold + 1;

                _opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), 
                    fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT]::_opts.breakdownRules.TTensorI;
                _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, 
                    filter := e-> When(e[1]*e[2] <= _thold^2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold)))]::_opts.breakdownRules.DFT;
                
                _opts.unparser.simt_synccluster := _opts.unparser.simt_syncblock;
#                _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
#                    FixUpCUDASigmaSPL_3Stage(s1, opts)); 
                _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
                    When(Collect(t, MDPRDFT)::Collect(t, IMDPRDFT) = [], 
                        FixUpCUDASigmaSPL_3Stage(s1, opts),
                        FixUpCUDASigmaSPL_3Stage_Real(s1, opts))); 


                _opts.postProcessCode := (c, opts) -> FixUpTeslaV_Code(c, opts);    
#                _opts.postProcessCode := (c, opts) -> FixUpTeslaV_Code(PingPong_3Stages(c, opts), opts);    
                _opts.fixUpTeslaV_Code := true;

                if ((Length(Collect(t, TTensorInd)) >= 1) or let(lst := Collect(t, TTensorI), (Length(lst) >= 1) and ForAll(lst, l->l.params[2] > 1))) then
                    _opts.operations.Print := s -> Print("<FFTX CUDA HPC Batch MDDFT/MDPRDFT/MDIPRDFT options record>");
                    _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTGridDimY, ASIMTBlockDimY, ASIMTBlockDimX];
                else
                    _opts.operations.Print := s -> Print("<FFTX CUDA HPC MDDFT/MDPRDFT/MDIPRDFT options record>");
                    _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];
                fi;

                _opts.HPCSupportedSizesCUDA := _HPCSupportedSizesCUDA;

            fi;
            
            return _opts;
        fi;

        # detect 3D DFT/iDFT but non-convolution case
        _tt := Collect(t, MDDFT);
        if Length(_tt) = 2 and ForAll(_tt, i->Length(i.params[1]) = 3) and Sum(List(_tt, i->i.params[2])) = Product(_tt[1].params[1]) then
            _conf := FFTXGlobals.confFFTCUDADevice();
            _opts := FFTXGlobals.getOpts(_conf);

            # opts for high performance CUDA cuFFT
            if Length(Filtered(_tt, i -> ObjId(i) = MDDFT)) > 0 and ForAll(_tt[1].params[1], i-> i in _HPCSupportedSizesCUDA) then
                _opts.breakdownRules.MDDFT := [fftx.platforms.cuda.MDDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.MDPRDFT := [fftx.platforms.cuda.MDPRDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.IMDPRDFT := [fftx.platforms.cuda.IMDPRDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
                _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];
                
                _opts.globalUnrolling := 2*_thold + 1;

                _opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), 
                    fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT]::_opts.breakdownRules.TTensorI;
                _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, 
                    filter := e-> When(e[1]*e[2] <= _thold^2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold)))]::_opts.breakdownRules.DFT;
                
                _opts.unparser.simt_synccluster := _opts.unparser.simt_syncblock;
                _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
                    FixUpCUDASigmaSPL_3Stage(s1, opts)); 
                _opts.postProcessCode := (c, opts) -> FixUpTeslaV_Code(PingPong_3Stages(c, opts), opts);    
                _opts.fixUpTeslaV_Code := true;

                _opts.operations.Print := s -> Print("<FFTX CUDA HPC MDDFT options record>");

            fi;
            
            return _opts;
        fi;
    
        # promote with default conf rules
        tt := _promote1(Copy(t));

        if ObjId(tt) = TFCall then
            _tt := tt.params[1];
            # check for convolution
            if (ObjId(_tt) in [MDRConv, MDRConvR, IOPrunedMDRConv]) or ((ObjId(_tt) in [TTensorI, TTensorInd]) and (ObjId(_tt.params[1]) in [MDRConv, MDRConvR])) then 
                _conf := FFTXGlobals.confMDRConvCUDADevice();
                _opts := FFTXGlobals.getOpts(_conf);

           
                # opts for high performance CUDA cuFFT
                if (ObjId(_tt) in [MDRConv, MDRConvR, IOPrunedMDRConv] and ForAll(_tt.params[1], i-> i in _HPCSupportedSizesCUDA)) or
                    (ObjId(_tt) in [TTensorI, TTensorInd] and ForAll(_tt.params[1].params[1], i-> i in _HPCSupportedSizesCUDA)) then
                    _opts.breakdownRules.MDDFT := [fftx.platforms.cuda.MDDFT_tSPL_Pease_SIMT];
                    _opts.breakdownRules.MDPRDFT := [fftx.platforms.cuda.MDPRDFT_tSPL_Pease_SIMT];
                    _opts.breakdownRules.IMDPRDFT := [fftx.platforms.cuda.IMDPRDFT_tSPL_Pease_SIMT];
                    _opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
                    
                    # handle IOPrunedMDRConv in CUDA -- TBD
                    _opts.breakdownRules.PrunedMDPRDFT := [PrunedMDPRDFT_tSPL_Base, PrunedMDPRDFT_tSPL_RowCol1];
                    _opts.breakdownRules.PrunedIMDPRDFT := [PrunedIMDPRDFT_tSPL_Base, PrunedIMDPRDFT_tSPL_RowCol1];
                    _opts.breakdownRules.PrunedMDDFT := [PrunedMDDFT_tSPL_Base, PrunedMDDFT_tSPL_RowCol];
                    _opts.breakdownRules.PrunedIMDDFT := [PrunedIMDDFT_tSPL_Base, PrunedIMDDFT_tSPL_RowCol];
                    _opts.breakdownRules.IOPrunedMDRConv := [IOPrunedMDRConv_tSPL_InvDiagFwd];
                    
                    _opts.globalUnrolling := 2*_thold + 1;
    
                    _opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), 
                        fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT]::_opts.breakdownRules.TTensorI;
                    _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, 
                        filter := e-> When(e[1]*e[2] <= _thold^2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold)))]::_opts.breakdownRules.DFT;
                    
                    _opts.unparser.simt_synccluster := _opts.unparser.simt_syncblock;
    #                _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
    #                    FixUpCUDASigmaSPL_3Stage(s1, opts)); 
                    _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
                        When(Collect(t, MDPRDFT)::Collect(t, IMDPRDFT) = [], 
                            FixUpCUDASigmaSPL_3Stage(s1, opts),
                            FixUpCUDASigmaSPL_3Stage_Real(s1, opts))); 
    
    
                    _opts.postProcessCode := (c, opts) -> FixUpTeslaV_Code(c, opts);    
    #                _opts.postProcessCode := (c, opts) -> FixUpTeslaV_Code(PingPong_3Stages(c, opts), opts);    
                    _opts.fixUpTeslaV_Code := true;
    
                    if ((Length(Collect(t, TTensorInd)) >= 1) or let(lst := Collect(t, TTensorI), (Length(lst) >= 1) and ForAll(lst, l->l.params[2] > 1))) then
                        _opts.operations.Print := s -> Print("<FFTX CUDA HPC Batch MDRConv/MDRConvR/IOPrunedMDRConv options record>");
                        _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTGridDimY, ASIMTBlockDimY, ASIMTBlockDimX];
                    else
                        _opts.operations.Print := s -> Print("<FFTX CUDA HPC MDRConv/MDRConvR/IOPrunedMDRConv options record>");
                        _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];
                    fi;
    
                    _opts.HPCSupportedSizesCUDA := _HPCSupportedSizesCUDA;
    
                fi;

                return _opts;
            fi;
            # check for Hockney. This is for N=130
            if ObjId(_tt) = IOPrunedMDRConv  and _tt.params[1] = [130,130,130] then
                _conf := FFTXGlobals.confHockneyMlcCUDADevice();
                _opts := FFTXGlobals.getOpts(_conf);
                return _opts;
            fi;
            # check for general Hockney. 
            if ObjId(_tt) = IOPrunedMDRConv then
                _conf := FFTXGlobals.confMDRConvCUDADevice();
                _opts := FFTXGlobals.getOpts(_conf);
                _opts.tags := [ASIMTKernelFlag(ASIMTGridDimY), ASIMTGridDimX, ASIMTBlockDimZ];
                return _opts;
            fi;
        fi;

        # check for WarpX
        _conf := FFTXGlobals.confWarpXCUDADevice();
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
        return FFTXGlobals.getOpts(conf); 
    fi;
    if IsBound(conf.useCUDA) then 
        return FFTXGlobals.getOpts(conf); 
    fi;
    
    # Here we have to handle GPU configs
    Error("Don't know how to derive opts!\n");
end; 
