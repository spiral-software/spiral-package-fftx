
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Import(fftx.platforms.cuda);
Import(fftx.codegen);
Import(simt);

Class(HockneyMLC_SIMTRules, RuleSet);


# Set of rules for applying Tile and Fission SIMT flags
RewriteRules(HockneyMLC_SIMTRules, rec(
	#Enable synchronization when different thread clusters are mapped to innermost sums
	# SUM(A*[S,IS], B*[S,IS]) => SUM(A, B)*SUM([S,IS],[S,IS])
    collect := Rule([SIMTSUM, [@(1, Compose),
                                @(2).cond(e -> not ObjId(e) in [Scat]), @(3).cond(s -> IsBound(s.simt_dim)) 
                            ], 
                            [@(6, Compose),
                                @(7).cond(e -> not ObjId(e) in [Scat]), @(8).cond(s -> IsBound(s.simt_dim))
                            ]
                    ], e -> let(dsum1 := @(3).val.dimensions[1], dsum2 := @(8).val.dimensions[1], 
                                Compose(    SIMTSUM(Copy(e.simt_dim),   @(2).val*Gath(H(dsum1+dsum2, dsum1, 0, 1)), 
                                                                    @(7).val*Gath(H(dsum1+dsum2, dsum2, dsum1, 1)) ),
                                            SIMTSUM(Copy(e.simt_dim),   Scat(H(dsum1+dsum2, dsum1, 0, 1))*@(3).val, 
                                                                    Scat(H(dsum1+dsum2, dsum2, dsum1, 1))*@(8).val )
                                    )
                            ),
                        "CollectCompose"
    ),

    tile    := Rule(@(1, SIMTISum, e->ObjId(e.simt_dim)=ASIMTTileFlag ),
                    e -> let(d1 := e.domain/e.simt_dim.tsize(),
                             v1 := var.fresh(e.var.id{[1]}, e.var.t, d1),
                             d2 := e.simt_dim.tsize(),
                             v2 := var.fresh(e.var.id{[1]}, e.var.t, d2),
                             newb := SubstVars(e.children()[1], rec( (e.var.id) := v1*d2 + v2 ) ),
                             simt_dlst := e.simt_dim.simt_dim(),
                             SIMTISum(Copy(simt_dlst.simt_dim(1)), v1, d1, 
                                SIMTISum(Copy(simt_dlst.simt_dim(2)), v2, d2, newb )
                                )
                        ),
                    "tile_SIMTISum" 
    ),

    fission_SIMTISum := Rule([@(1, SIMTISum, e->ObjId(e.simt_dim)=ASIMTFissionFlag ),
                            @(2, Compose, e-> not ForAny(e.children(), cch -> ObjId(cch) in [Gath, Scat]) )
                        ], function(e)
                            local inner_comp_list, ch, cch, cchlen, pos, simt_dlst, b, newb, v, sl, gl;

                            ch := @(2).val;
                            inner_comp_list := [];
                            cchlen :=  Length(ch.children());
                            for cch in ch.children() do
                                b := Copy( cch );
                                v := e.var.clone();
                                newb := SubstVars(b, rec( (e.var.id) := v ));
                                sl := newb.dimensions[1];
                                gl := newb.dimensions[2];
                                pos := Length(inner_comp_list);
                                simt_dlst := e.simt_dim.simt_dim();
                                Add(inner_comp_list,
                                    Cond(
                                        pos = 0,        SIMTISum(Copy(simt_dlst.simt_dim(pos+1)), v, e.domain, newb*Gath(H(e.domain*gl, gl, gl*v, 1)) ),
                                        pos = cchlen-1, SIMTISum(Copy(simt_dlst.simt_dim(pos+1)), v, e.domain, Scat(H(e.domain*sl, sl, sl*v, 1))*newb ),
                                        SIMTISum(Copy(simt_dlst.simt_dim(pos+1)), v, e.domain, Scat(H(e.domain*sl, sl, sl*v, 1))*newb*Gath(H(e.domain*gl, gl, gl*v, 1)) )
                                    )
                                );
                            od;
                            e := Compose( inner_comp_list );
                            return e;
                        end,
                        "fission_SIMTISum"
    ),

    # If outer loop is mapped to cluster (warp) dimension and it contains other block-parallel-loops then loop must be fissioned
    # to enable synchronization.
    xy_fission := Rule([@(1, SIMTISum, e->ObjId(e.simt_dim)=ASIMTBlockDimX ),
                            @(2, Compose, e->   not ForAny(e.children(), cch -> ObjId(cch) in [Gath, Scat]) and
                                                ForAny(List(Collect(e, @(3,[SIMTISum, SIMTSUM]) ), s -> s.simt_dim.simt_sync()), 
                                                        s -> s > @(1).val.simt_dim.simt_sync()) 
                            )
                        ], function(e)
                            local   inner_comp_list, sync_list, 
                                    ch, cch, cchlen, b, newb, v, sl, gl;

                            inner_comp_list := Collect(e.children(), 
                                                    @@(4, Compose, 
                                                        (expr, cx) -> IsBound(cx.SIMTSUM) and cx.SIMTSUM <> [] 
                                                                    or 
                                                                    IsBound(cx.SIMTISum) and cx.SIMTISum <> []
                                                                    or
                                                                    IsBound(cx.Compose) and cx.Compose <> []
                                                    )
                                                );
                            sync_list := List(Collect(inner_comp_list, @(5,[SIMTISum, SIMTSUM]) ), s -> s.simt_dim.simt_sync());
                            if ForAny(sync_list, s -> s > e.simt_dim.simt_sync()) then
                                Error("Cannot properly syncronize within following structure: ", e);
                            fi;

                            ch := @(2).val;
                            inner_comp_list := [];
                            cchlen :=  Length(ch.children());
                            for cch in ch.children() do
                                b := Copy( cch );
                                v := e.var.clone();
                                newb := SubstVars(b, rec( (e.var.id) := v ));
                                sl := newb.dimensions[1];
                                gl := newb.dimensions[2];
                                Add(inner_comp_list,
                                    Cond(
                                        Length(inner_comp_list) = 0,        SIMTISum(Copy(e.simt_dim), v, e.domain, newb*Gath(H(e.domain*gl, gl, gl*v, 1)) ),
                                        Length(inner_comp_list) = cchlen-1, SIMTISum(Copy(e.simt_dim), v, e.domain, Scat(H(e.domain*sl, sl, sl*v, 1))*newb ),
                                        SIMTISum(Copy(e.simt_dim), v, e.domain, Scat(H(e.domain*sl, sl, sl*v, 1))*newb*Gath(H(e.domain*gl, gl, gl*v, 1)) )
                                    )
                                );
                            od;
                            e := Compose( inner_comp_list );
                            return e;
                        end,
                        "xy_fission_SIMTISum" 
    ),    
)); 



HockneyMLC_fftxGen := function(t, opts)
    local sb, cb, rtb, symbl, tt, tb,
    nd3, nfreq, n2, ns3, nfreq, ns2, 
    tag_prdft_pd_loop1, rtp_prdft_pd_loop1, tag_dft_pd_loop1, rtp_dft_pd_loop1, tag_prdft_pd_loop2, rtp_prdft_pd_loop2, tag_dft_pd_loop2, rtp_dft_pd_loop2,
    tag_prdft_rec1, rtp_prdft_rec1, tag_prdft_rec2, rtp_prdft_rec2, tag_prdft_rec_z, rtp_prdft_rec_z, tag_dft_rec_y, rtp_dft_rec_y, BX, tag_idft_rec_x, rtp_idft_rec_x,
    tag_dft_rec_x, rtp_dft_rec_x, tag_conv_x, rtp_conv_x, tag_pridft_rec_z, rtp_pridft_rec_z, tag_idft_rec_y, rtp_idft_rec_y, TS_stg1, TS_stg2, TS_stg543, trstg5to1;
    
    # promote t
    tt := opts.preProcess(t);
    
    # parse parameters    
    symbl := tt.params[2].params[1];
    tb := tt.params[1];
    nd3 := Length(tt.params[1].params[4][1]);
    nfreq := tt.params[1].params[1][1]/2+1;
    n2 := tt.params[1].params[1][1];
    ns3 := Length(tt.params[1].params[6][1]);
    ns2 := Length(tt.params[1].params[6][1]);
    opts.symbol := [symbl];

	if IsBound(tt.params[2].fname) then
	   opts.cudasubName := tt.params[2].fname;
	fi;
    
    # code gen script
    rtb := RuleTreeMid(tb, opts);
    
    ## Pruned Real DFT vvvvv
    # rec1: PRDFT_PD_loop presents 4 parallelizable objects
    tag_prdft_pd_loop1 := [ASIMTBlockDimY(32,[0,0],[0,31]), ASIMTLoopDim(), ASIMTBlockDimY(32, [1,6]), ASIMTLoopDim()];
    # rec1: PRDFT_PD_loop produces 0 subrules 
    rtp_prdft_pd_loop1 := tag_rec(tag_prdft_pd_loop1, Ignore);
    
    # rec1: DFT_PD_loop presents 3 parallelizable objects
    tag_dft_pd_loop1 := [ASIMTBlockDimY(32,[7,7],[0,31]), ASIMTBlockDimY(32, [8,14]), ASIMTLoopDim()];
    # rec1: DFT_PD_loop produces 0 subrules 
    rtp_dft_pd_loop1 := tag_rec(tag_dft_pd_loop1, Ignore);
    
    # rec2: PRDFT_PD_loop presents 4 parallelizable objects
    tag_prdft_pd_loop2 := [ASIMTBlockDimY(32,[15,15],[0,31]), ASIMTLoopDim(), ASIMTBlockDimY(32, [16,21]), ASIMTLoopDim()];
    # rec2: PRDFT_PD_loop produces 0 subrules 
    rtp_prdft_pd_loop2 := tag_rec(tag_prdft_pd_loop2, Ignore);
    
    # rec2: DFT_PD_loop presents 3 parallelizable objects
    tag_dft_pd_loop2 := [ASIMTBlockDimY(32,[22,22],[0,31]), ASIMTBlockDimY(32, [23,29]), ASIMTLoopDim()];
    # rec2: DFT_PD_loop produces 0 subrules 
    rtp_dft_pd_loop2 := tag_rec(tag_dft_pd_loop2, Ignore);
    
    tag_prdft_rec1 := [ASIMTBlockDimY(32), ASIMTLoopDim(), ASIMTBlockDimY(32), ASIMTBlockDimY(32, [0,3]), ASIMTBlockDimY(32, [4,12])];
    rtp_prdft_rec1 := tag_rec(tag_prdft_rec1, [rtp_prdft_pd_loop1, rtp_dft_pd_loop1] :: Replicate(2, Ignore));
    
    tag_prdft_rec2 := [ASIMTBlockDimY(32), ASIMTLoopDim(), ASIMTBlockDimY(32), ASIMTBlockDimY(32, [13,15]), ASIMTBlockDimY(32, [16,25])];
    rtp_prdft_rec2 := tag_rec(tag_prdft_rec2, [rtp_prdft_pd_loop2, rtp_dft_pd_loop2] :: Replicate(2, Ignore));
    
    #Top prdft rec rule presents 3 parallelizable objects
    tag_prdft_rec_z := [ASIMTBlockDimX(16,[0,0],[0,15]), ASIMTBlockDimX(16), ASIMTBlockDimY(32)];
    #Top prdft rec rule produces 4 subrules 
    rtp_prdft_rec_z := tag_rec(tag_prdft_rec_z, [Ignore, Ignore, rtp_prdft_rec1, rtp_prdft_rec2]);
    
    ## Pruned Real DFT ^^^^^
    
    ## Pruned DFT vvvvv
    
    #Top dft rec rule presents 4 parallelizable objects
    tag_dft_rec_y := [ASIMTBlockDimY(13, [0,9]), ASIMTBlockDimY(13), ASIMTBlockDimY(13, [0,6]), ASIMTBlockDimY(13, [7,12])];
    #Top dft rec rule produces 3 subrules 
    rtp_dft_rec_y := tag_rec(tag_dft_rec_y, Replicate(3, Ignore));
    
    ## Pruned DFT ^^^^^
    
    ## Pruned Conv vvvvv
    BX := 16;
    #Top idft rec rule presents 4 parallelizable objects
    tag_idft_rec_x := [ASIMTBlockDimX(BX), ASIMTBlockDimX(BX, [0,7]), ASIMTBlockDimX(BX, [8,12]), ASIMTBlockDimX(BX, [0,9])];
    #Top idft rec rule produces 3 subrules 
    rtp_idft_rec_x := tag_rec(tag_idft_rec_x, Replicate(3, Ignore));
    
    #Top dft rec rule presents 4 parallelizable objects
    tag_dft_rec_x := [ASIMTBlockDimX(BX, [0,9]), ASIMTBlockDimX(BX), ASIMTBlockDimX(BX, [0,6]), ASIMTBlockDimX(BX, [7,12])];
    #Top dft rec rule produces 3 subrules 
    rtp_dft_rec_x := tag_rec(tag_dft_rec_x, Replicate(3, Ignore));
    
    #Top IOPrunedConv rule presents no parallelizable objects
    tag_conv_x := Ignore;
    #Top IOPrunedConv rule produces 2 subrules 
    rtp_conv_x := tag_rec(tag_conv_x, [rtp_idft_rec_x, rtp_dft_rec_x]);
    
    ## Pruned IDFT YZ vvvvv
    
    BX := 32;
    #Z-dim pridft rec rule presents 3 parallelizable objects
    tag_pridft_rec_z := [ASIMTBlockDimX(BX, [0,1]), ASIMTBlockDimX(BX, [0,0], [0,31]), ASIMTBlockDimX(BX)];
    #Z-dim pridft rec rule produces 3 subrules 
    rtp_pridft_rec_z := tag_rec(tag_pridft_rec_z, Replicate(3, Ignore));
    
    #Y-dim idft rec rule presents 4 parallelizable objects
    tag_idft_rec_y := [ASIMTBlockDimX(BX), ASIMTBlockDimX(BX, [0,7]), ASIMTBlockDimX(BX, [8,12]), ASIMTBlockDimX(BX, [0,9])];
    #Y-dim idft rec rule produces 3 subrules 
    rtp_idft_rec_y := tag_rec(tag_idft_rec_y, Replicate(3, Ignore));
    
    ## Pruned IDFT YZ ^^^^^
    
    TS_stg1 := 11;
    TS_stg2 := 11;
    TS_stg543 := 4;
    
    # Define TagTree
    trstg5to1 := tag_rec( 
                        # Taggable operators (Tensors, Sums, etc.) as they appear in Collect( IOPrunedMDRConv.apply(...), @([Tag. ops]) )
                        [ ASIMTLoopDim(), ASIMTKernelFlag( ASIMTGridDimX(nd3) ), ASIMTBlockDimY(6), ASIMTBlockDimY(6), # Stages 7-6: 4 tag operators
                            ASIMTLoopDim(), ASIMTLoopDim(), ASIMTTileFlag(
                                                            ASIMTListDim(
                                                                ASIMTKernelFlag(ASIMTGridDimX(nfreq*n2/TS_stg543)),
                                                                ASIMTBlockDimY(TS_stg543)
                                                            ), TS_stg543), ASIMTLoopDim(), ASIMTLoopDim(), # Stages 5-4-3: 5 tag operators
                            ASIMTKernelFlag(ASIMTGridDimX(ns3)), ASIMTTileFlag(
                                                                ASIMTListDim(
                                                                    ASIMTGridDimY(nfreq/TS_stg2),
                                                                    ASIMTBlockDimX(11) # << 11 OK because perfectly nests Inner loop mapped to Y (so block sync is used at compose boundaries)
                                                                ), TS_stg2), # Stage 2: 2 tag operators
                            ASIMTKernelFlag( ASIMTGridDimX(ns2) ), 
                                    ASIMTTileFlag(
                                            ASIMTListDim(
                                                ASIMTGridDimY(ns3/TS_stg1),
                                                ASIMTFissionFlag( ASIMTListDim(ASIMTBlockDimY(32), ASIMTBlockDimX(16)) )
                                            ), TS_stg1), ASIMTLoopDim() ], # Stage 1: 3 tag operators
                        # Children as they appear in IOPrunedMDRConv.children(...)
                        [ rtp_prdft_rec_z, rtp_dft_rec_y, rtp_conv_x, rtp_idft_rec_y, rtp_pridft_rec_z, Ignore ]);
    
    tmp_tag_rt(rtb, trstg5to1, opts); # RuleTree + TagTree = TaggedRuleTree
    
    sb := SumsRuleTree(rtb, opts);
    cb := CodeSums(sb, opts);
    cb.transform := t;

    return cb;
end;


Class(HockneyMlcCUDADeviceOpts, TitanVDefaults, rec(
    operations := rec(Print := s -> Print("<Hockney MLC CUDA Device options record>")),
    tagIt := (self, t) >> t.withTags(self.tags)
));


hockneyMlcCUDADeviceOpts := function(arg) 
    local opts;
    
    opts := Copy(HockneyMlcCUDADeviceOpts);
    opts.includes := [];

    opts.breakdownRules.PRDFT := List([ PRDFT1_Base2, CopyFields(PRDFT_PD, rec(maxSize := 7)), PRDFT_PD_loop], _noT); 
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT_PD_loop, CopyFields(IPRDFT_PD, rec(maxSize := 7))], _noT); 
    opts.breakdownRules.IPRDFT2 := List([ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT], _noT); 
    opts.breakdownRules.PRDFT3 := [ ];
    opts.breakdownRules.URDFT := List([ URDFT1_Base1, URDFT1_Base2, URDFT1_Base4, URDFT1_CT ], _noT); 
    opts.breakdownRules.DFT := List([ DFT_Base, CopyFields(DFT_PD, rec(maxSize := 7)),  DFT_PD_loop], _noT); 
    opts.breakdownRules.PrunedPRDFT := List([ CopyFields(PrunedPRDFT_base, rec(maxSize := 5)), PrunedPRDFT_CT_rec_block ], _noT); 
    opts.breakdownRules.PrunedIPRDFT := List([ CopyFields(PrunedIPRDFT_base, rec(maxSize := 5)), PrunedIPRDFT_CT_rec_block ], _noT); 
    opts.breakdownRules.PrunedDFT := List([ PrunedDFT_base, PrunedDFT_CT_rec_block ], _noT); 
    opts.breakdownRules.IOPrunedMDRConv := List([ IOPrunedMDRConv_3D_2trip_zyx_freqdata ], _noT); 
    #opts.breakdownRules.MDRConv := [ MDRConv_3D_2trip_zyx_freqdata ]; # [MDRConv_Base]
    
    wrap_rule_apply(opts.breakdownRules.PRDFT, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.IPRDFT, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.IPRDFT2, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.URDFT, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.DFT, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.PrunedPRDFT, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.PrunedIPRDFT, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.PrunedDFT, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.PrunedIDFT, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.TTensorI{[2..3]}, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.IOPrunedConv, wrap_apply);
    wrap_rule_apply(opts.breakdownRules.IOPrunedMDRConv, wrap_apply);

    opts.codegen._Formula := Copy(opts.codegen.Formula);
    opts.codegen.Formula := meth ( self, o, y, x, opts )
        local prog, prog1, s;
        s := Copy(o);
        s := fixUpSigmaSPL(s, opts);
    
        opts.("postCompileStrategy") := function(icode, opts)
            local icode1;
            icode1 := skip();
            # This reduces the amount of constant matrices but might prevent to store 
            # some of them in constant mem. E.g., two identical matrices but only one has broadcast-type access.
            # while icode1 <> icode do 
            #     icode1 := icode;
            #     icode := fixReplicatedData(icode, opts);
            # od;
            icode := scalarizeAccumulator(icode, opts);
            return icode;
        end;
    
        prog := opts.codegen._Formula(s, y, x, opts);
    
        return prog;
    end;


    opts.formulaStrategies.postProcess := opts.formulaStrategies.postProcess 
        :: opts.formulaStrategies.sigmaSpl :: [MergedRuleSet(HockneyMLC_SIMTRules, RulesSums)];
        
    opts.fftxGen := (self, t) >> HockneyMLC_fftxGen(t, self);
    opts.tags := [];
    
    opts.preProcess := (self, t) >> let(t1 := RulesFFTXPromoteNT(Copy(t)), RulesFFTXPromoteNT_Cleanup(t1));
    opts.prettyPrint := meth(self, c)
        local name;
        name := "fftx_generated";
        if (IsBound(c.transform) and IsBound(c.transform.params[2].fname)) then
            name := c.transform.params[2].fname;
        fi;
        PrintCode(name, c, self);
    end;
    
    return opts;
end;


confHockneyMlcCUDADevice := rec(
    defaultName := "confHockneyMlcCUDADevice",
    defaultOpts := (arg) >> rec(useHockneyMlcCUDADevice := true),
    confHandler := hockneyMlcCUDADeviceOpts 
);

fftx.FFTXGlobals.registerConf(confHockneyMlcCUDADevice);



