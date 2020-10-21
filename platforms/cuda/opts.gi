Class(FFTXCUDAOpts, FFTXOpts, simt.TitanVDefaults, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX CUDA options record>")),    
    max_threads := 1024
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


cudaConf := rec(
    defaultName := "defaultCUDAConf",
    defaultOpts := (arg) >> rec(useCUDA := true),
    devFunc := false,
    confHandler := cudaOpts 
);

fftx.FFTXGlobals.registerConf(cudaConf);

#--
Class(FFTXCUDADeviceOpts, FFTXOpts, simt.TitanVDefaults, rec(
    tags := [],
    devFunc := true,
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
        DFT_PD ];
    opts.breakdownRules.TTensorInd := [dsA_base, L_dsA_L_base, dsA_L_base, L_dsA_base];    
    return opts;
end;


cudaDeviceConf := rec(
    defaultName := "defaultCUDADeviceConf",
    defaultOpts := (arg) >> rec(useCUDADevice := true),
    confHandler := cudaDeviceOpts 
);

fftx.FFTXGlobals.registerConf(cudaDeviceConf);


