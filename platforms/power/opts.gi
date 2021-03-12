
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Class(FFTXPOWER9Opts, FFTXOpts, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX POWER9 options record>")),    
));

power9Opts := function(arg)
    local opts, optrec, vsxopts;
    
    optrec := rec(dataType := T_Real(64), globalUnrolling := 32);
    vsxopts := rec(svct := true, splitL := false, oddSizes := false, stdTTensor := true, tsplPFA := false);
    
    opts := CopyFields(FFTXOpts, SIMDGlobals.getOpts(POWER9_2xf, vsxopts));
    opts.breakdownRules.TFCall := FFTXOpts.breakdownRules.TFCall;
    
    opts.globalUnrolling := 32;
    
    # FFTX specific breakdown rules
    opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
    opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT, PRDFT1_PF, PRDFT_PD, PRDFT_Rader], _noT);
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD, IPRDFT_Rader ], _noT);
    opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);

    return opts;
end;

Declare(ParseOptsPOWER);

Class(FFTXPOWER9DefaultConf, rec(
    __call__ := self >> self,
    getOpts := (self, t) >> ParseOptsPOWER(self, t),
    operations := rec(Print := s -> Print("<FFTX POWER9 Default Configuration>")),
    useOMP := false,
    useSIMD := true
));

power9Conf := rec(
    defaultName := "defaultPOWER9Conf",
    defaultOpts := (arg) >> FFTXPOWER9DefaultConf,
    useOMP := false,
    useSIMD := true,
    confHandler := power9Opts 
);

fftx.FFTXGlobals.registerConf(power9Conf);

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
ParseOptsPOWER := function(conf, t)
    local tt, _tt, _conf, _opts;
    
    _opts := power9Opts();
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

