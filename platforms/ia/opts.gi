
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Class(FFTXIAOpts, FFTXOpts, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX IA options record>")),    
    max_threads := 1024
));

iaOpts := function(arg)
    local opts, optrec, sseopts, smpopts;
    
    optrec := rec(dataType := T_Real(64), globalUnrolling := 32);
    smpopts := rec(numproc := 4, api := "OpenMP");
    sseopts := rec(svct := true, splitL := false, oddSizes := false, stdTTensor := true, tsplPFA := false);

    if Length(arg) >= 1 then optrec := CopyFields(optrec, arg[1]); fi;
    if Length(arg) >= 2 then smpopts := CopyFields(smpopts, arg[2]); fi;
    if Length(arg) >= 3 then sseopts := CopyFields(sseopts, arg[3]); fi;
    
    opts := CopyFields(FFTXOpts, IAGlobals.getOpts(optrec, smpopts, sseopts));
    opts.breakdownRules.TFCall := FFTXOpts.breakdownRules.TFCall;
    
    # FFTX specific breakdown rules
    opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
    opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT, PRDFT1_PF, PRDFT_PD, PRDFT_Rader], _noT);
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD, IPRDFT_Rader ], _noT);
    opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);
    Add(opts.includes, "\"mm_malloc.h\"");

    return opts;
end;

Declare(ParseOptsIA);

Class(FFTXIADefaultConf, rec(
    __call__ := self >> self,
    getOpts := (self, t) >> ParseOptsIA(self, t),
    operations := rec(Print := s -> Print("<FFTX IA Default Configuration>")),
    useOMP := true,
    useSIMD := true
));

iaConf := rec(
    defaultName := "defaultIAConf",
    defaultOpts := (arg) >> FFTXIADefaultConf,
    useOMP := true,
    useSIMD := true,
    confHandler := iaOpts 
);

fftx.FFTXGlobals.registerConf(iaConf);

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

# this is a first experimental opts-deriving logic. This needs to be done extensible and properly
ParseOptsIA := function(conf, t)
    local tt, _tt, _conf, _opts;
    
    _opts := iaOpts();
    return _opts;
    
#    if IsBound(conf.useCUDADevice) then 
#        # detect real MD convolution
#        _tt := Collect(t, RCDiag)::Collect(t, MDPRDFT)::Collect(t, IMDPRDFT)::Collect(t, TTensorI);
#        if Length(_tt) = 4 then
#            _conf := FFTXGlobals.confWarpXCUDADevice();
#            _opts := FFTXGlobals.getOpts(_conf);        
#            return _opts;
#        fi;        
#
#        # detect batch of DFT/PRDFT/MDDFT/MDPRDFT
#        if ((Length(Collect(t, TTensorInd)) >= 1) or (Length(Collect(t, TTensorI)) >= 1)) and 
#            ((Length(Collect(t, DFT)) = 1) or (Length(Collect(t, PRDFT)) = 1) or (Length(Collect(t, IPRDFT)) = 1) or
#              (Length(Collect(t, MDDFT)) >= 1) or (Length(Collect(t, MDPRDFT)) >= 1) or (Length(Collect(t, IMDPRDFT)) >= 1)) then
#            _conf := FFTXGlobals.confBatchFFTCUDADevice();
#            _opts := FFTXGlobals.getOpts(_conf);
#            return _opts;
#        fi;
#       
#        # detect 3D DFT
#        _tt := Collect(t, MDDFT)::Collect(t, MDPRDFT)::Collect(t, IMDPRDFT);
#        if Length(_tt) = 1 and Length(_tt[1].params[1]) = 3 then
#            _conf := FFTXGlobals.confFFTCUDADevice();
#            _opts := FFTXGlobals.getOpts(_conf);
#            return _opts;
#        fi;
#    
#        # promote with default conf rules
#        tt := _promote1(Copy(t));
#
#        if ObjId(tt) = TFCall then
#            _tt := tt.params[1];
#            # check for convolution
#            if (ObjId(_tt) = MDRConv) or ((ObjId(_tt) = TTensorI) and (ObjId(_tt.params[1]) = MDRConv)) then 
#                _conf := FFTXGlobals.confMDRConvCUDADevice();
#                _opts := FFTXGlobals.getOpts(_conf);
#                return _opts;
#            fi;
#            # check for Hockney. This is for N=130
#            if ObjId(_tt) = IOPrunedMDRConv  and _tt.params[1] = [130,130,130] then
#                _conf := FFTXGlobals.confHockneyMlcCUDADevice();
#                _opts := FFTXGlobals.getOpts(_conf);
#                return _opts;
#            fi;
#        fi;
#
#        # check for WarpX
#        _conf := FFTXGlobals.confWarpXCUDADevice();
#        _opts := FFTXGlobals.getOpts(_conf);
#        tt := _opts.preProcess(Copy(t));
#        if ObjId(tt) = TFCall and ObjId(tt.params[1]) = TCompose then
#            _tt := tt.params[1].params[1];
#            # detect promoted WarpX
#            if IsList(_tt) and Length(_tt) = 3 and List(_tt, ObjId) = [ TNoDiagPullinRight, TRC, TNoDiagPullinLeft ] then
#                return _opts;
#            fi;
#        fi;
#        # we are doing nothing special
#        return FFTXGlobals.getOpts(conf); 
#    fi;
#    if IsBound(conf.useCUDA) then 
#        return FFTXGlobals.getOpts(conf); 
#    fi;
#    
#    # Here we have to handle GPU configs
    Error("Don't know how to derive opts!\n");
end; 

