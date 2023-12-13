
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Import(simt);
ImportAll(dct_dst);
ImportAll(realdft);
ImportAll(dtt);


#knownFactors := [];
#knownFactors[30000] := [ 20, 10, 10, 15 ];
#
#
#factorInto := function(N, stages)
#    local stageval, fct, n, mapping, factors, m, buckets, j, sad, mn, idx, bestFct;
#    stageval := exp(log(N)/stages).v;
#    
#    fct := Factors(N);
#    n := Length(fct);
#    mapping := ApplyFunc(Cartesian, Replicate(n, [1..stages]));
#    
#    factors := [];
#    for m in mapping do
#        buckets := Replicate(stages, 1);
#        for j in [1..n] do
#            buckets[m[j]] := buckets[m[j]] * fct[j];
#           Add(factors, buckets);
#        od;
#    od;
#    
#    sad := List(factors, m -> Sum(List(m, i -> AbsFloat(i - stageval))));
#    mn := Minimum(sad);
#    idx := Position(sad, mn);
#    bestFct := factors[idx];
#
#    return bestFct;
#end;
#
#bestFactors := function(N, max_factor)
#    local factors, i, f, bestf;
#    
#    if IsBound(knownFactors[N]) then return knownFactors[N]; fi;
#    
#    factors := List([2..4], i -> factorInto(N, i));
#    
#    bestf := Filtered(factors, f -> ForAll(f, i -> i < 26))[1];
#    knownFactors[N] := bestf;
#    return bestf;
#end;


peelFactor := (n, max_factor) -> Filtered(DivisorPairs(n), e->e[1] <= max_factor);
expandFactors := (f, max_factor) -> let(lst := List(f, e -> DropLast(e, 1)::peelFactor(Last(e), max_factor)), 
    rlst := ApplyFunc(Concatenation, List(lst, a -> let(ll := Length(Filtered(a, v -> not IsList(v))), List(Drop(a, ll), v ->a{[1..ll]}::v)))),
    When(IsList(rlst[1]), rlst, [rlst]));
findFactors := (lst, max_factor) -> When(IsInt(lst), When(Last(Factors(lst)) > max_factor, [], 
    findFactors(peelFactor(lst, max_factor), max_factor)), When(not ForAny(lst, lst -> Last(lst) <= max_factor), 
    findFactors(expandFactors(lst, max_factor), max_factor), lst));   
factorize := (n, max_factor, max_prime) -> When(n <= max_factor, [n], let(fct := When(Last(Factors(n)) > max_prime, [[n]], findFactors(peelFactor(n, max_factor), max_factor)), 
    nroot := exp(log(n)/Length(fct[1])).v, sad := List(fct, m -> Sum(List(m, i -> AbsFloat(i -nroot)))),  mn := Minimum(sad), 
    Sort(fct[Position(sad, mn)])));
isSupported := (n, max_factor, max_prime) -> let(
    fct := factorize(n, max_factor, max_prime),
    Length(fct) > 1 and Length(fct) <= 4 or (Length(fct) = 1 and When(IsPrime(fct[1]), fct[1] < max_prime, fct[1] < max_factor)));



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

_globalSize1 := [];

# this is a first experimental opts-deriving logic. This needs to be done extensible and properly
ParseOptsCUDA := function(conf, t)
    local tt, _tt, _tt2, _conf, _opts, _HPCSupportedSizesCUDA, _thold, _thold_prdft, _ThreeStageSizesCUDA, _FourStageSizesCUDA,
    MAX_TWOPOWER, MAX_KERNEL, MAX_PRIME, MIN_SIZE, MAX_SIZE, size1, filter, MAGIC_SIZE, _isHPCSupportedSizesCUDA, _isSupportedAtAll, _isHPCSupportedSizesOld;
    
    # all dimensions need to be inthis array for the high perf MDDFT conf to kick in for now
    # size 320 is problematic at this point and needs attention. Need support for 3 stages to work first
    MAX_KERNEL := 26;
    MAX_PRIME := 17;
    MIN_SIZE := 32;
    MAX_SIZE := 680;
    MAGIC_SIZE := 4096;
    MAX_TWOPOWER := 16;

    _thold := MAX_KERNEL;
    _thold_prdft := MAX_PRIME;
    
    filter := (e) -> When(e[1] * e[2] <= _thold ^ 2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold);
    size1 := Filtered([MIN_SIZE..MAX_SIZE], i -> ForAny(DivisorPairs(i), filter) and ForAll(Factors(i), j -> not IsPrime(j) or j <= MAX_PRIME));
    _HPCSupportedSizesCUDA := size1;
    _globalSize1 := size1;
    
    _isHPCSupportedSizesCUDA := n -> isSupported(n, MAX_KERNEL, MAX_PRIME);
    _isHPCSupportedSizesOld := n -> n in _HPCSupportedSizesCUDA;
    
    _isSupportedAtAll := (t, n) -> When(t in [DFT, MDDFT, PRDFT, IPRDFT, MDPRDFT, IMDPRDFT], _isHPCSupportedSizesCUDA(n), n in _HPCSupportedSizesCUDA);
    
    # -- initial guard for 3 stages algorithm
    _ThreeStageSizesCUDA := e -> e >= MAX_KERNEL^2 or e in [512];
    _FourStageSizesCUDA := e -> e >= MAX_KERNEL^3 or e in [8192, 16384]; 

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

#Error();  
# -- 3 stage algorithm detection here --            
        if ((Length(Collect(t, DFT)) = 1) or (Length(Collect(t, PRDFT)) = 1) or (Length(Collect(t, IPRDFT)) = 1)) and
            ForAll(Flat(List(Collect(t, @(1, [DFT, PRDFT, IPRDFT])), j -> rec(n:=j.params[1], id := ObjId(j)))), j -> _isSupportedAtAll(j.id, j.n)) then
          
            _conf := FFTXGlobals.confBatchFFTCUDADevice();
            _opts := FFTXGlobals.getOpts(_conf);

            _opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
            _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];
#                _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimX];
            
            _opts.globalUnrolling := When(Collect(t, PRDFT)::Collect(t, IPRDFT) = [], 2*_thold + 1, 8 * MAX_TWOPOWER);

            # handle weird out of ressources problem for DFT(8k) and beyond
            if Length(Collect(t, DFT)) = 1 and Collect(t, DFT)[1].params[1] > MAGIC_SIZE then _opts.max_threads := _opts.max_threads / 2; fi;
            _opts.max_blocks := 32768;
            
            _opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), CopyFields(L_IxA_split, rec(switch := true)),
#                    CopyFields(TTensorI_vecrec, rec(switch := true, minSize := 16, supportedNTs := [DFT], numTags := 2)),
# FIX-FOR-NOW: disable tiling for now
#                fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT
            ]::DropLast(_opts.breakdownRules.TTensorI, 1);

            # support for single kernel batches                
            if (Length(Collect(t, DFT)) = 1 and Collect(t, DFT)[1].params[1] <= MAX_KERNEL) or 
               (Length(Collect(t, PRDFT)) = 1 and Collect(t, PRDFT)[1].params[1] <= MAX_KERNEL) or
               (Length(Collect(t, IPRDFT)) = 1 and Collect(t, IPRDFT)[1].params[1] <= MAX_KERNEL) then 
                Add(_opts.breakdownRules.TTensorI, fftx.platforms.cuda.IxA_SIMT_peelof3);
#                _opts.breakdownRules.TTensorI := _opts.breakdownRules.TTensorI{[1,2,5,6,7]};
            fi;                
                
            _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, # here we need to make sure to get the right decomposition fo r3 and 4 stages, TBD/FIXME
                filter := e-> let(factors := factorize(e[1]*e[2], MAX_KERNEL, MAX_PRIME), 
                    When(Length(factors) <= 3, e[1] = factors[1], e[1] = factors[1]*factors[2])))),
                CopyFields(DFT_Rader, rec(minSize := DFT_PD.maxSize + 1, maxSize := MAX_PRIME, switch := true))]::
                _opts.breakdownRules.DFT;
 
# For PRDFT bigger surgery is needed: 1) upgrade CT rules to NewRules to guard against tags, and 2) tspl_CT version of the PRDFT_CT rule                    
            _opts.breakdownRules.PRDFT := [ PRDFT1_Base1, PRDFT1_Base2, CopyFields(PRDFT1_CT, rec(
                    allChildren := P -> Filtered(PRDFT1_CT.allChildren(P), 
                        e-> let(factors := factorize(e[1].params[1]*e[3].params[1], MAX_KERNEL, MAX_PRIME), 
                            Cond(Length(factors) = 1, true, ForAny(factors, u->e[1].params[1] = u))))
                        )), 
                CopyFields(PRDFT_PD, rec(maxSize := MAX_PRIME)) ];        
            _opts.breakdownRules.IPRDFT := [ IPRDFT1_Base1, IPRDFT1_Base2, CopyFields(IPRDFT1_CT, rec(
                    allChildren := P -> Filtered(IPRDFT1_CT.allChildren(P), 
                        e-> let(factors := factorize(e[1].params[1]*e[3].params[1], MAX_KERNEL, MAX_PRIME), 
                            Cond(Length(factors) = 1, true, ForAny(factors, u->e[1].params[1] = u))))
                        )), 
                CopyFields(IPRDFT_PD, rec(maxSize := MAX_PRIME)) ];
            
# _opts.breakdownRules.PRDFT[3].allChildren := P -> Filtered(PRDFT1_CT.allChildren(P), 
#     e-> let(factors := factorize(e[1].params[1]*e[3].params[1], MAX_KERNEL, MAX_PRIME), 
#         When(Length(factors) <= 3, e[1].params[1] = factors[1], e[1].params[1] = factors[1]*factors[2])));
# 
# _opts.breakdownRules.IPRDFT[3].allChildren := P -> Filtered(IPRDFT1_CT.allChildren(P), 
#     e-> let(factors := factorize(e[1].params[1]*e[3].params[1], MAX_KERNEL, MAX_PRIME), 
#         When(Length(factors) <= 3, e[1].params[1] = factors[1], e[1].params[1] = factors[1]*factors[2])));
            
# this is a quick hack to get correct code for up to 1024, but the code is slow as all parallelism is dropped by the legacy PRDFT rule
#            _opts.breakdownRules.PRDFT[3].allChildren := P -> Filtered(PRDFT1_CT.allChildren(P), i -> i[1].params[1] <= _thold_prdft);    
#            _opts.breakdownRules.IPRDFT[3].allChildren := P -> Filtered(IPRDFT1_CT.allChildren(P), i -> i[1].params[1] <= _thold_prdft);
           
            _opts.unparser.simt_synccluster := _opts.unparser.simt_syncblock;
            _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
                When(Collect(t, PRDFT)::Collect(t, IPRDFT) = [], 
                    FixUpCUDASigmaSPL(FixUpCUDASigmaSPL_3Stage(s1, opts), opts),
                    FixUpCUDASigmaSPL(
                    FixUpCUDASigmaSPL_3Stage_Real(
                    s1, opts)
                    , opts)
                    )); 
            _opts.postProcessCode := (c, opts) -> FixUpTeslaV_Code(c, opts);    
#                _opts.postProcessCode := (c, opts) -> FixUpTeslaV_Code(PingPong_3Stages(c, opts), opts);    
            _opts.fixUpTeslaV_Code := true;

            _opts.operations.Print := s -> Print("<FFTX CUDA HPC (PR)DFT 3+ stages options record>");

            return _opts;
        fi;
# -- end 3 stage algo --           

        # detect batch of DFT/PRDFT
        # this branch is rightnow dead code, it seems 
        # - the DFT branch catches the batch DFT
        # - not clear if the bathc specific stuff here is still needed (for performance)
        if ((Length(Collect(t, TTensorInd)) >= 1) or let(lst := Collect(t, TTensorI), (Length(lst) >= 1) and ForAll(lst, l->l.params[2] > 1))) and 
            ((Length(Collect(t, DFT)) = 1) or (Length(Collect(t, PRDFT)) = 1) or (Length(Collect(t, IPRDFT)) = 1)) then
            _conf := FFTXGlobals.confBatchFFTCUDADevice();
            _opts := FFTXGlobals.getOpts(_conf);

            # opts for high performance CUDA cuFFT
            if #ForAll(Flat(List(Collect(t, @(1, [DFT, PRDFT, IPRDFT])), j-> j.params[1])), i -> i in _HPCSupportedSizesCUDA)  then
                ForAll(Flat(List(Collect(t, @(1, [DFT, PRDFT, IPRDFT])), j -> rec(n:=j.params[1], id := ObjId(j)))), j -> _isSupportedAtAll(j.id, j.n)) then
            
                _opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
                
                if Length(Collect(t, TTensorI)) = 2 then
                    _opts.tags := [ ASIMTKernelFlag(ASIMTGridDimX), ASIMTGridDimY, ASIMTBlockDimY, ASIMTBlockDimX ];
                else
                    _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];
                fi;
                
                _opts.globalUnrolling := 2*_thold + 1;

                _opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), 
                    fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT]::_opts.breakdownRules.TTensorI;

#                _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, 
#                    filter := e-> When(e[1]*e[2] <= _thold^2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold)))]::_opts.breakdownRules.DFT;

                _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, # here we need to make sure to get the right decomposition fo r3 and 4 stages, TBD/FIXME
                    filter := e-> let(factors := factorize(e[1]*e[2], MAX_KERNEL, MAX_PRIME), 
                        When(Length(factors) <= 3, e[1] = factors[1], e[1] = factors[1]*factors[2])))),
                    CopyFields(DFT_Rader, rec(minSize := DFT_PD.maxSize + 1, maxSize := MAX_PRIME, switch := true))]::
                    _opts.breakdownRules.DFT;
             
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
        _tt := Collect(t, MDDFT)::Collect(t, MDPRDFT)::Collect(t, IMDPRDFT)::Collect(t, PrunedMDPRDFT)::Collect(t, PrunedIMDPRDFT);
        if Length(_tt) = 1 and Length(_tt[1].params[1]) = 3 then
#            if ObjId(_tt[1]) in [MDDFT] then
#                _conf := FFTXGlobals.confBatchFFTCUDADevice();
#            else
                _conf := FFTXGlobals.confFFTCUDADevice();
#            fi;
            _opts := FFTXGlobals.getOpts(_conf);
           
            # opts for high performance CUDA cuFFT
            if #ForAll(_tt[1].params[1], i-> _isHPCSupportedSizesCUDA(i)) then
               ForAll(Flat(List(_tt, j -> rec(n:=j.params[1], id := ObjId(j)))), j -> ForAll(j.n, k-> _isSupportedAtAll(j.id, k) and k > Minimum(_HPCSupportedSizesCUDA))) then
            
                _opts.breakdownRules.MDDFT := [fftx.platforms.cuda.MDDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.MDPRDFT := [fftx.platforms.cuda.MDPRDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.IMDPRDFT := [fftx.platforms.cuda.IMDPRDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
                _opts.breakdownRules.PrunedMDPRDFT := [ PrunedMDPRDFT_tSPL_Pease_SIMT ];
                _opts.breakdownRules.PrunedIMDPRDFT := [ PrunedIMDPRDFT_tSPL_Pease_SIMT ];
                _opts.breakdownRules.PrunedDFT := [ PrunedDFT_base, PrunedDFT_DFT, PrunedDFT_CT, PrunedDFT_CT_rec_block, 
                    CopyFields(PrunedDFT_tSPL_CT, rec(switch := true)) ];
                    
                if ObjId(_tt[1]) in [MDPRDFT, IMDPRDFT] and ForAll(_tt[1].params[1], i -> i > MAX_KERNEL) then
                    _opts.breakdownRules.TFCall := _opts.breakdownRules.TFCall{[1]};    # when is rule [2] needed?
                fi;
                
                _opts.globalUnrolling := 4 * MAX_TWOPOWER;
#                _opts.globalUnrolling := 2*_thold + 1;
                _opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), CopyFields(L_IxA_split, rec(switch := true)),
                    fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT]:: 
                    When(ForAny(_tt, _t -> ObjId(_t) in [PrunedMDPRDFT, PrunedIMDPRDFT]), 
                        [fftx.platforms.cuda.IxA_SIMT_peelof, fftx.platforms.cuda.IxA_SIMT_peelof2], [])::_opts.breakdownRules.TTensorI;

if ForAll(_tt[1].params[1], i-> _isHPCSupportedSizesOld(i)) then
                _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, # here we need to make sure to get the right decomposition fo r3 and 4 stages, TBD/FIXME
                    filter := e-> let(factors := factorize(e[1]*e[2], MAX_KERNEL, MAX_PRIME), 
                        When(Length(factors) <= 3, e[1] = factors[1], e[1] = factors[1]*factors[2]))
                    
                        )
                    ),
                    CopyFields(DFT_Rader, rec(minSize := DFT_PD.maxSize + 1, maxSize := MAX_PRIME, switch := true))]::
                    _opts.breakdownRules.DFT;
else

#===================
#             # support for single kernel batches                
#             if (Length(Collect(t, DFT)) = 1 and Collect(t, DFT)[1].params[1] <= MAX_KERNEL) or 
#                (Length(Collect(t, PRDFT)) = 1 and Collect(t, PRDFT)[1].params[1] <= MAX_KERNEL) or
#                (Length(Collect(t, IPRDFT)) = 1 and Collect(t, IPRDFT)[1].params[1] <= MAX_KERNEL) then 
#                 Add(_opts.breakdownRules.TTensorI, fftx.platforms.cuda.IxA_SIMT_peelof3);
# #                _opts.breakdownRules.TTensorI := _opts.breakdownRules.TTensorI{[1,2,5,6,7]};
#             fi;                
                
            _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, # here we need to make sure to get the right decomposition fo r3 and 4 stages, TBD/FIXME
                filter := e-> let(factors := factorize(e[1]*e[2], MAX_KERNEL, MAX_PRIME), 
                    Cond(Length(factors) = 1, true, Length(factors) <= 3, e[1] = factors[1], e[1] = factors[1]*factors[2])))),
                CopyFields(DFT_Rader, rec(minSize := DFT_PD.maxSize + 1, maxSize := MAX_PRIME, switch := true))]::
                _opts.breakdownRules.DFT;
 
            _opts.breakdownRules.PRDFT := [ PRDFT1_Base1, PRDFT1_Base2, CopyFields(PRDFT1_CT, rec(
                    allChildren := P -> Filtered(PRDFT1_CT.allChildren(P), 
                        e-> let(factors := factorize(e[1].params[1]*e[3].params[1], MAX_KERNEL, MAX_PRIME), 
                            Cond(Length(factors) = 1, true, ForAny(factors, u->e[1].params[1] = u))))
                        )), 
                CopyFields(PRDFT_PD, rec(maxSize := MAX_PRIME, switch := true)) ];        
            _opts.breakdownRules.IPRDFT := [ IPRDFT1_Base1, IPRDFT1_Base2, CopyFields(IPRDFT1_CT, rec(
                    allChildren := P -> Filtered(IPRDFT1_CT.allChildren(P), 
                        e-> let(factors := factorize(e[1].params[1]*e[3].params[1], MAX_KERNEL, MAX_PRIME), 
                            Cond(Length(factors) = 1, true, ForAny(factors, u->e[1].params[1] = u))))
                        )), 
                CopyFields(IPRDFT_PD, rec(maxSize := MAX_PRIME, switch := true)) ];

#===================

fi;
                
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
#                    _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];
                    if ObjId(_tt[1]) = MDDFT and ForAny(_tt[1].params[1], i -> i >= MAX_SIZE) then
                        _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimX];
                        _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
                            When(Collect(t, PRDFT)::Collect(t, IPRDFT) = [], 
                                FixUpCUDASigmaSPL(FixUpCUDASigmaSPL_3Stage(s1, opts), opts),
                                FixUpCUDASigmaSPL_3Stage_Real(s1, opts))); 
                    else
                        _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];
                    fi;
                    
                    
if ForAll(_tt[1].params[1], i-> _isHPCSupportedSizesOld(i)) then
                    
                    if ObjId(_tt[1]) = MDDFT and ForAny(_tt[1].params[1], i -> i <= MAX_KERNEL) then
                        Add(_opts.breakdownRules.TTensorI, fftx.platforms.cuda.IxA_SIMT_peelof3);
                        _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
                            FixUpCUDASigmaSPL(When(Collect(t, PRDFT)::Collect(t, IPRDFT) = [], 
                                FixUpCUDASigmaSPL(FixUpCUDASigmaSPL_3Stage(s1, opts), opts),
                                FixUpCUDASigmaSPL_3Stage_Real(s1, opts)), opts)); 
                    fi;
else                    
#==================

                    if ObjId(_tt[1]) in [MDPRDFT, IMDPRDFT] and ForAll(_tt[1].params[1], i -> i > MAX_KERNEL) then

                        _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
                            When(Collect(t, MDPRDFT)::Collect(t, IMDPRDFT) = [], 
                                FixUpCUDASigmaSPL(FixUpCUDASigmaSPL_3Stage(s1, opts), opts),
                                FixUpCUDASigmaSPL(
                                FixUpCUDASigmaSPL_3Stage_Real(
                                s1, opts)
                                , opts)
                                )); 
                        _opts.operations.Print := s -> Print("<FFTX CUDA HPC 3+ stage MDDFT/MDPRDFT/MDIPRDFT options record>");
                    fi;
fi;
#=======================                    
                    
                    
                fi;

#                _opts.HPCSupportedSizesCUDA := _HPCSupportedSizesCUDA;

            fi;
            
            return _opts;
        fi;
        # detect 3D DFT/iDFT but non-convolution case
        _tt := Collect(t, MDDFT);
        if Length(_tt) = 2 and ForAll(_tt, i->Length(i.params[1]) = 3) and Sum(List(_tt, i->i.params[2])) = Product(_tt[1].params[1]) then
            _conf := FFTXGlobals.confFFTCUDADevice();
            _opts := FFTXGlobals.getOpts(_conf);

            # opts for high performance CUDA cuFFT
            if Length(Filtered(_tt, i -> ObjId(i) = MDDFT)) > 0 and ForAll(_tt[1].params[1], i-> _isHPCSupportedSizesCUDA(i)) then
                _opts.breakdownRules.MDDFT := [fftx.platforms.cuda.MDDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.MDPRDFT := [fftx.platforms.cuda.MDPRDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.IMDPRDFT := [fftx.platforms.cuda.IMDPRDFT_tSPL_Pease_SIMT];
                _opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
                _opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];
                
                _opts.globalUnrolling := 2*_thold + 1;

                _opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), 
                    fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT]::_opts.breakdownRules.TTensorI;
#                _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, 
#                    filter := e-> When(e[1]*e[2] <= _thold^2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold)))]::_opts.breakdownRules.DFT;

                _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, # here we need to make sure to get the right decomposition fo r3 and 4 stages, TBD/FIXME
                    filter := e-> let(factors := factorize(e[1]*e[2], MAX_KERNEL, MAX_PRIME), 
                        When(Length(factors) <= 3, e[1] = factors[1], e[1] = factors[1]*factors[2]))
                    
                        )
                    ),
                    CopyFields(DFT_Rader, rec(minSize := DFT_PD.maxSize + 1, maxSize := MAX_PRIME, switch := true))]::
                    _opts.breakdownRules.DFT;
               
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

        if ObjId(tt) in [TFCall, TFCallF] then
            _tt := tt.params[1];
            # check for convolution
            if (ObjId(_tt) in [PrunedMDPRDFT, PrunedIMDPRDFT, MDRConv, MDRConvR, IOPrunedMDRConv]) or ((ObjId(_tt) in [TTensorI, TTensorInd]) and (ObjId(_tt.params[1]) in [MDRConv, MDRConvR])) then 
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
                    _opts.breakdownRules.PrunedMDPRDFT := [ PrunedMDPRDFT_tSPL_Pease_SIMT ];
                    _opts.breakdownRules.PrunedIMDPRDFT := [ PrunedIMDPRDFT_tSPL_Pease_SIMT ];
#                    _opts.breakdownRules.PrunedMDPRDFT := [PrunedMDPRDFT_tSPL_Base, PrunedMDPRDFT_tSPL_RowCol1];
#                    _opts.breakdownRules.PrunedIMDPRDFT := [PrunedIMDPRDFT_tSPL_Base, PrunedIMDPRDFT_tSPL_RowCol1];
                    _opts.breakdownRules.PrunedMDDFT := [PrunedMDDFT_tSPL_Base, PrunedMDDFT_tSPL_RowCol];
                    _opts.breakdownRules.PrunedIMDDFT := [PrunedIMDDFT_tSPL_Base, PrunedIMDDFT_tSPL_RowCol];
                    _opts.breakdownRules.IOPrunedMDRConv := [IOPrunedMDRConv_tSPL_InvDiagFwd];
                    
#                    _opts.globalUnrolling := 2*_thold + 1;
                    _opts.globalUnrolling := 4*MAX_TWOPOWER;
    
                    _opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)),
                        fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT,
                        fftx.platforms.cuda.IxA_SIMT_peelof, fftx.platforms.cuda.IxA_SIMT_peelof2]::_opts.breakdownRules.TTensorI;
                        
                        
                    _opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, 
                        filter := e-> When(e[1]*e[2] <= _thold^2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold)))]::_opts.breakdownRules.DFT;
                        
                    _opts.unparser.simt_synccluster := _opts.unparser.simt_syncblock;
    #                _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
    #                    FixUpCUDASigmaSPL_3Stage(s1, opts)); 
                    _opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesDiagStandalonePointwise, 
                            RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
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
        if ObjId(tt) in [TFCall, TFCallF] and ObjId(tt.params[1]) = TCompose then
            _tt := tt.params[1].params[1];
            # detect promoted WarpX
            if IsList(_tt) and Length(_tt) = 3 and List(_tt, ObjId) = [ TNoDiagPullinRight, TRC, TNoDiagPullinLeft ] then
                return _opts;
            fi;
        fi;
        
        # check for MDDST1
        _conf := FFTXGlobals.confFFTCUDADevice();
        _opts := FFTXGlobals.getOpts(_conf);
        tt := _opts.preProcess(Copy(t));
        if ObjId(tt) in [TFCall, TFCallF] and ObjId(tt.params[1]) = MDDST1 then
            _opts.breakdownRules.DCT3 := [ DCT3_toSkewDCT3 ];
            _opts.breakdownRules.DST1 := [ DST1_toDCT3 ];
            _opts.breakdownRules.SkewDTT := [ SkewDTT_Base2, SkewDCT3_VarSteidl ];
            _opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)),
                fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT,
                fftx.platforms.cuda.IxA_SIMT_peelof, fftx.platforms.cuda.IxA_SIMT_peelof2]::_opts.breakdownRules.TTensorI;
        
            _opts.operations.Print := s -> Print("<FFTX CUDA MDDST1 options record>");
            return _opts;
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
