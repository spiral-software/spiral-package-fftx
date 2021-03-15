
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

ImportAll(paradigms.smp);
ImportAll(paradigms.vector);

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
    
    opts.globalUnrolling := optrec.globalUnrolling;
    
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

#-----------------------------------------
# OpenMP + VSX

Class(FFTXPOWER9OMPOpts, FFTXOpts, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX POWER9 OpenMP options record>")),    
));

power9OMPOpts := function(arg)
    local opts, optrec, vsxopts, smpopts, tid;
    
    optrec := rec(dataType := T_Real(64), globalUnrolling := 32);
    vsxopts := rec(svct := true, splitL := false, oddSizes := false, stdTTensor := true, tsplPFA := false);
    smpopts := rec(numproc := LocalConfig.cpuinfo.cores, api := "OpenMP");
    
    opts := CopyFields(FFTXOpts, SIMDGlobals.getOpts(POWER9_2xf, vsxopts));
    opts.breakdownRules.TFCall := FFTXOpts.breakdownRules.TFCall;
    
    #-- OpenMP opts merging-- 
    opts.unparser := When(IsBound(smpopts.OmpMode) and smpopts.OmpMode = "for", 
                            OpenMP_POWERUnparser_ParFor, 
                            OpenMP_POWERUnparser);
    opts.codegen := spiral.libgen.VecRecCodegen;
    
    opts.breakdownRules.GT := [ GT_Base, GT_NthLoop, 
        CopyFields(GT_Par, rec(parEntireLoop := false, splitLoop := true)), GT_Par_odd,
        GT_Vec_AxI, GT_Vec_IxA, GT_Vec_IxA_Push, GT_Vec_IxA_L, GT_Vec_L_IxA        
    ];
    opts.breakdownRules.TTensorI := Concat([ 
    CopyFields(TTensorI_toGT, rec(
        applicable := (self, t) >> t.hasTags() and ObjId(t.getTags()[1])=AParSMP ))], 
        opts.breakdownRules.TTensorI);
        
    opts.breakdownRules.TTensorInd := 
        Concat([dsA_base_smp, dsA_smp, L_dsA_L_base_smp, L_dsA_L_smp], 
            opts.breakdownRules.TTensorInd);    
    
    tid := When(smpopts.api = "OpenMP", threadId(), var("tid", TInt));
    opts.tags := Concat([ AParSMP(smpopts.numproc, tid)  ], opts.tags);
    #----------
    opts.globalUnrolling := optrec.globalUnrolling;
    Add(opts.includes, "\"mm_malloc.h\"");
    
    # FFTX specific breakdown rules
    opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
    opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, PRDFT1_CT, PRDFT1_PF, PRDFT_PD, PRDFT_Rader], _noT);
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD, IPRDFT_Rader ], _noT);
    opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);

    return opts;
end;

Declare(ParseOptsPOWER);

Class(FFTXPOWER9OMPConf, rec(
    __call__ := self >> self,
    getOpts := (self, t) >> ParseOptsPOWER(self, t),
    operations := rec(Print := s -> Print("<FFTX POWER9 OpenMP Configuration>")),
    useOMP := true,
    useSIMD := true
));

power9OMPConf := rec(
    defaultName := "defaultPOWER9OMPConf",
    defaultOpts := (arg) >> FFTXPOWER9OMPConf,
    useOMP := true,
    useSIMD := true,
    confHandler := power9OMPOpts 
);

fftx.FFTXGlobals.registerConf(power9OMPConf);

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

# this is a first experimental opts-deriving logic. This needs to be done extensible and properly
ParseOptsPOWER := function(conf, t)
    local tt, _tt, _conf, _opts;
    
    if conf.useOMP then
        _opts := power9OMPOpts();
    else
        _opts := power9Opts();
    fi;
    return _opts;
    
    Error("Don't know how to derive opts!\n");
end; 

