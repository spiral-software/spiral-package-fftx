
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Class(RulesSIMTFission, RuleSet);

RewriteRules(RulesSIMTFission, rec(
    FissionSIMTISum := Rule([@(1, SIMTISum), @(2, Compose, e->ForAll(e.children(), c->ObjId(c) = SIMTISum))],
        e-> let(gath := Gath(fTensor(fBase(@(1).val.var), fId(Cols(@(2).val.child(1))))),
                scat := Scat(fTensor(fBase(@(1).val.var), fId(Cols(@(2).val.child(1))))),
                c1 := @(2).val.child(1) * gath, 
                cn := scat * Last(@(2).val.children()),
                cms := List(DropLast(Drop(@(2).val.children(), 1), 1), c->scat * c *gath),
                Compose(List([c1]::cms::[cn], c -> SIMTISum(@(1).val.simt_dim, @(1).val.var, @(1).val.domain, c))))),
));

RewriteRules(RulesSums, rec(
    OO_Gath := Rule([@(1, Compose), OO, Gath], 
        e -> ApplyFunc(OO, @(1).val.dims())),
    Gath_OO := Rule([@(1, Compose), Scat, OO], 
        e -> ApplyFunc(OO, @(1).val.dims())),
    Scat_OO_Gath := Rule([@(1, Compose), Scat, OO, Gath], 
        e -> ApplyFunc(OO, @(1).val.dims())),
));

SIMT_NextLoop := (s, simtidx) -> 
    SubstBottomUp(s, [@(1, SIMTISum), @(2, Compose, e->ForAll(e.children(), c->ObjId(c) in [ISum, SUM]))], 
        e->let(sx1c := @(2).val.children(),
            doms := List(sx1c, c->When(ObjId(c) = ISum, c.domain, Length(c.children()))),
            mdom := Maximum(doms),
            ranges := List(List(sx1c, c->[0, When(ObjId(c) = ISum, c.domain, Length(c.children()))-1])),
            newc := List([1..Length(sx1c)], 
                i-> When(ObjId(sx1c[i]) = ISum,
                        SIMTISum(simtidx(mdom, ranges[i]), sx1c[i].var, sx1c[i].domain, sx1c[i].child(1)),
                        SIMTSUM(simtidx(mdom, ranges[i]), sx1c[i].children()))),
            SIMTISum(@(1).val.simt_dim, @(1).val.var, @(1).val.domain, Compose(newc))
        )
    );


_fBaseVar := function(s, i, v)
    local cands;
    
    cands := Collect(s, @(0, fBase, e->e.params[1]=i.range and IsInt(e.params[2]) and e.params[2] = v));
    if cands = [] then return s; fi;
    
    s := SubstTopDown(s, @(0, fBase, e->e.params[1]=i.range and IsInt(e.params[2]) and e.params[2] = v), 
        e->fBase(i));
    return s;
end;

FixUpCUDASigmaSPL := function(s, opts)
    local srt, kernels, i, _s, newv;
    
    if IsBound(s.ruletree) then srt := s.ruletree; fi;
    
    s := SIMT_NextLoop(s, ASIMTBlockDimY);
    s := SIMT_NextLoop(s, ASIMTBlockDimX);
    
    s:= SubstTopDown(s, @(1, SIMTSUM),
        e->let(
            its := @(1).val.simt_dim.params[1],
            ii := Ind(its),
            ch := @(1).val.children(),
            nch := Length(ch),
            SIMTISum(@(1).val.simt_dim, ii, nch, SUM(List([0..nch-1], 
               i->COND(eq(ii, V(i)), _fBaseVar(ch[i+1], ii, i), ApplyFunc(OO, ch[i+1].dims()))))))
    );

    
    if Length(Collect(s,  @(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX))) > 0 then
        s := SubstTopDown(s, [@@(1, SIMTISum, (e,cx)->ObjId(e.simt_dim) = ASIMTBlockDimY), @(2, [BB, SUM])],
            (e,cx)->let(xdimcds := Collect(Filtered(cx.SIMTISum, s->ObjId(s.simt_dim) = ASIMTKernelFlag), @(0, SIMTISum, (e)->ObjId(e.simt_dim) = ASIMTBlockDimX)),
                xdim := When(xdimcds <> [], xdimcds[1].domain, @@(1).val.domain),
                SIMTISum(@@(1).val.simt_dim, @@(1).val.var, @@(1).val.domain, 
                    SIMTISum(ASIMTBlockDimX(xdim,[0,0]), Ind(1), 1, @@(1).val.child(1))))
        );
        
        s := ApplyStrategy(s, [RulesSIMTFission, RulesSums], BUA, opts);
#        s := RulesSums(RulesSIMTFission(s));
        
        s := SubstTopDown(s, @@(1, SIMTSUM, (e,cx)->ObjId(e.simt_dim) = ASIMTBlockDimY and ForAll(e.children(), c->not ObjId(c) in [SIMTISum, SIMTSUM])),
            (e,cx)->let(xdimcds := Collect(Filtered(cx.SIMTISum, s->ObjId(s.simt_dim) = ASIMTBlockDimZ), @(0, SIMTISum, (e)->ObjId(e.simt_dim) = ASIMTBlockDimX)),
                xdim := When(xdimcds <> [], xdimcds[1].domain, When(IsBound(@@(1).val.domain), @@(1).val.domain, 1)),
                SIMTSUM(@@(1).val.simt_dim,  
                    List(@@(1).val.children(), c->SIMTISum(ASIMTBlockDimX(xdim, [0,0]), Ind(1), 1, c))))
        );
        
        s := SubstTopDown(s, [@(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX and Set(Flat(e.simt_dim.simt_all_ranges())) = [0]), @(2, ISum)],
            e -> SIMTISum(ASIMTBlockDimX(@(2).val.domain), @(2).val.var, @(2).val.domain, @(2).val.child(1))
        );

        if ObjId(s) = Compose then         
            kernels := s.children();
    
            for i in [1..Length(kernels)] do
                _s := kernels[i];
                if Length(Collect(_s,  @(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX))) > 0 then
                    newv := Maximum(List(List(Collect(_s, @(1, [SIMTSUM, SIMTISum], e->ObjId(e.simt_dim) = ASIMTBlockDimX)), e-> e.simt_dim), i->i.params[1]));
                    _s := SubstTopDown(_s, @(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX),
                        e->SIMTISum(ApplyFunc(ASIMTBlockDimX, [newv]::Drop(@(1).val.simt_dim.params, 1)), @(1).val.var, @(1).val.domain, @(1).val.child(1)) 
                    );
                
                    _s := SubstTopDown(_s, @(1, SIMTSUM, e->ObjId(e.simt_dim) = ASIMTBlockDimX),
                        e->SIMTSUM(ApplyFunc(ASIMTBlockDimX, [newv]::Drop(@(1).val.simt_dim.params, 1)), @(1).val.children()) 
                    );
                fi;    
                kernels[i] := _s;
            od;
            s := Compose(kernels);
        fi;
    
        s := SubstTopDown(s, [@(1, SIMTSUM, e->ObjId(e.simt_dim) = ASIMTBlockDimX), @(2,BB), @(3, ISum)], 
            e->SIMTSUM(ApplyFunc(ObjId(@(1).val.simt_dim),  [@(1).val.simt_dim.params[1], [0, @(3).val.domain-1], [@(3).val.domain,@(3).val.domain]]), 
                SIMTISum(ApplyFunc(ObjId(@(1).val.simt_dim),  [@(1).val.simt_dim.params[1], [0, @(3).val.domain-1]]), @(3).val.var, @(3).val.domain, @(3).val.child(1)),
                @(2).val)
        );
    fi;
    # somehow SIMTSUM by itself does not create the necessary loop dimension (?)
    s := SubstBottomUp(s, [@(1,SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimZ), [BB, @(2,SUM)], ...],
        e->let(ch := @(2).val.children(), nch := Length(ch),
            ii := Ind(nch),
            SIMTISum(@(1).val.simt_dim, @(1).val.var, @(1).val.domain,
                SIMTISum(ASIMTBlockDimY(nch), ii, nch, SUM(List([0..nch-1], i->COND(eq(ii, V(i)), BB(_fBaseVar(ch[i+1], ii, i)), ApplyFunc(OO, ch[i+1].dims())))))))
    );

    s := SubstTopDown(s, @(1, Grp), e->e.child(1));
    
    s := SubstBottomUp(s, @(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimZ),
        e->let(SIMTISum(ASIMTGridDimZ(@(1).val.simt_dim.params[1]), @(1).val.var, @(1).val.domain, @(1).val.child(1)))
    );
    
    if IsBound(opts.fixUpZ) and opts.fixUpZ then
        s := SubstBottomUp(s, @(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTGridDimZ),
            e -> let(
                orig_zdim := @(1).val.domain,
                xdim := Maximum([1]::List(Collect(@(1).val.child(1), @(2, [SIMTSUM, SIMTISum], e->ObjId(e.simt_dim) = ASIMTBlockDimX)), i->i.simt_dim.params[1])),
                ydim := Maximum([1]::List(Collect(@(1).val.child(1), @(2, [SIMTSUM, SIMTISum], e->ObjId(e.simt_dim) = ASIMTBlockDimY)), i->i.simt_dim.params[1])),
                zdim_inner := Maximum([1]::Filtered(DivisorsInt(orig_zdim), i -> i * xdim * ydim <= (opts.max_threads))),
                zdim_outer := orig_zdim / zdim_inner,
                io := Ind(zdim_outer),
                sdo := ASIMTGridDimZ(zdim_outer),
                ii := Ind(zdim_inner),
                sdi := ASIMTBlockDimZ(zdim_inner),
                krn := SubstVars(Copy(@(1).val.child(1)), rec((@(1).val.var.id) := (io * zdim_inner + ii))),
                SIMTISum(sdo, io, zdim_outer,
                    SIMTISum(sdi, ii, zdim_inner, krn))
            )
        );
    fi;

    if IsBound(s.ruletree) then s.ruletree := srt; fi;

    return s;
end;


#FixUpCUDASigmaSPL_3Stage := (ss, opts) -> SubstTopDown(ss, @(1, Grp), e->e.child(1));
FixUpCUDASigmaSPL_3Stage := function(ss, opts)
    local kernels, _s, newv;

    # drop grp
    ss := SubstTopDown(ss, @(1, Grp), e->e.child(1));
   
    # parallelize and flatten loop
    ss:= let(simtidx := ASIMTBlockDimX, 
        SubstBottomUp(ss, [@(1, SIMTISum), @(2, Compose, e->ForAll(e.children(), c->ObjId(c) = ISum))], 
            e->let( #Error(), 
                    simtloop := @(1).val,
                    sx1c := @(2).val.children(),
                    doms := List(sx1c, c->c.domain),
                    i := @(1).val.var,
                    mdom := Maximum(doms),
                    ranges := List(List(sx1c, c->[0, c.domain-1])),
                    nch := [sx1c[1] * Gath(fTensor(fBase(i), fId(Cols(sx1c[1]))))] :: 
                        List(sx1c{[2..Length(sx1c)-1]}, c -> Scat(fTensor(fBase(i), fId(Rows(c)))) * c * Gath(fTensor(fBase(i), fId(Cols(c))))) :: 
                        [ Scat(fTensor(fBase(i), fId(Rows(Last(sx1c))))) * Last(sx1c)],
                    newc := List([1..Length(nch)], 
                        j-> SIMTISum(simtloop.simt_dim, simtloop.var, simtloop.domain, 
                                SIMTISum(simtidx(mdom, ranges[j]), sx1c[j].var, sx1c[j].domain, ApplyStrategy(nch[j], opts.formulaStrategies.sigmaSpl, BUA, opts).child(1)))),
                ApplyFunc(Compose, newc)
            ))
    );
    ss := ApplyStrategy(ss, opts.formulaStrategies.sigmaSpl, BUA, opts);

#    ss:= let(simtidx := ASIMTBlockDimX, 
#        SubstBottomUp(ss, [@(1, SIMTISum), @(2, Compose, e->ForAll(e.children(), c->ObjId(c) = ISum))], 
#            e->let(sx1c := @(2).val.children(),
#                    doms := List(sx1c, c->c.domain),
#                    i := @(1).val.var,
#                    mdom := Maximum(doms),
#                    ranges := List(List(sx1c, c->[0, c.domain-1])),
#                    newc := List([1..Length(sx1c)], 
#                        j-> SIMTISum(@(1).val.simt_dim, @(1).val.var, @(1).val.domain, SIMTISum(simtidx(mdom, ranges[j]), sx1c[j].var, sx1c[j].domain, sx1c[j].child(1)))),
#                ApplyFunc(Compose, newc)
#            ))
#    );

    # loop distribution Y(X*X)
    ss := SubstBottomUp(ss, [@(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimY), Compose], 
        e -> let(ch := @(1).val.child(1).children(), i := @(1).val.var, 
            nch := [ch[1] * Gath(fTensor(fBase(i), fId(Cols(ch[1]))))] :: 
                List(ch{[2..Length(ch)-1]}, c -> Scat(fTensor(fBase(i), fId(Rows(c)))) * c * Gath(fTensor(fBase(i), fId(Cols(c))))) :: 
                [ Scat(fTensor(fBase(i), fId(Rows(Last(ch))))) * Last(ch)],
            Compose(List(nch, c -> SIMTISum(@(1).val.simt_dim, @(1).val.var, @(1).val.var.range, c)))
            ));
    ss := ApplyStrategy(ss, opts.formulaStrategies.sigmaSpl, BUA, opts);

    # privatize temporaries -- that should be a no-op now
    ss := SubstTopDown(ss, [@(1, SIMTISum), @(2, Compose, e->ForAll(e.children(), c->ObjId(c) = ISum))],
        e-> let(ch := @(1).val.child(1).children(), 
            i := @(1).val.var, 
            nch := [ch[1] * Gath(fTensor(fBase(i), fId(Cols(ch[1]))))] :: 
                List(ch{[2..Length(ch)-1]}, c -> Scat(fTensor(fBase(i), fId(Rows(c)))) * c * Gath(fTensor(fBase(i), fId(Cols(c))))) :: 
                [ Scat(fTensor(fBase(i), fId(Rows(Last(ch))))) * Last(ch)],
            SIMTISum(@(1).val.simt_dim, @(1).val.var, @(1).val.domain, Compose(List(nch, c->ApplyStrategy(c, opts.formulaStrategies.sigmaSpl, BUA, opts))))
        )
    );

    # flatten X/X -> X loops
    ss := SubstTopDown(ss, 
        [@(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX), [@(2, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX), BB]],
        e->let(s1 := @(1).val,
            i1 := s1.var,
            i2 := s1.child(1).var,
            rng := i1.range * i2.range,
            ii := Ind(rng),
            sr := rec(
                (i1.id) := idiv(ii, i2.range),
                (i2.id) := imod(ii, i2.range)
            ),
            sdim := ASIMTBlockDimX(rng),
            SIMTISum(sdim, ii, ii.range, SubstVars(s1.child(1).child(1), sr))
        )
    );

    # flatten Y/X -> X loops
    ss := SubstTopDown(ss, 
        [@(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimY), [@(2, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX), BB]],
        e->let(s1 := @(1).val,
            i1 := s1.var,
            i2 := s1.child(1).var,
            rng := i1.range * i2.range,
            ii := Ind(rng),
            sr := rec(
                (i1.id) := idiv(ii, i2.range),
                (i2.id) := imod(ii, i2.range)
            ),
            sdim := ASIMTBlockDimX(rng),
            SIMTISum(sdim, ii, ii.range, SubstVars(s1.child(1).child(1), sr))
        )
    );
    
    # fix loop iterations
    if ObjId(ss) = Compose then         
        kernels := ss.children();
    
        for i in [1..Length(kernels)] do
            _s := kernels[i];
            if Length(Collect(_s,  @(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX))) > 0 then
                newv := Maximum(List(List(Collect(_s, @(1, [SIMTSUM, SIMTISum], e->ObjId(e.simt_dim) = ASIMTBlockDimX)), e-> e.simt_dim), i->i.params[1]));
                _s := SubstTopDown(_s, @(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX),
                    e->SIMTISum(ApplyFunc(ASIMTBlockDimX, [newv]::Drop(@(1).val.simt_dim.params, 1)), @(1).val.var, @(1).val.domain, @(1).val.child(1)) 
                );
            
                _s := SubstTopDown(_s, @(1, SIMTSUM, e->ObjId(e.simt_dim) = ASIMTBlockDimX),
                    e->SIMTSUM(ApplyFunc(ASIMTBlockDimX, [newv]::Drop(@(1).val.simt_dim.params, 1)), @(1).val.children()) 
                );
            fi;    
            kernels[i] := _s;
        od;
        ss := Compose(kernels);
    fi;
    
    return ss;
end;


PingPong_3Stages := function(c, opts)
    local cands, candvars, outvars, invars, in_loops, substvars, link_loops, linkvars, link_func, subst_list, substrec;
    
    cands := Collect(c, @(1, specifiers_func, e -> Collect(e, @(2, decl, f -> Length(f.vars) = 2 and ForAll(f.vars, k -> ObjId(k.t) = TArray))) <> []));
    candvars := chain(cands).free()::[X, Y];

    if Length(cands) > 0 then
        outvars := Filtered([Y]::Flat(List(Collect(c, @@(1, decl,
            (e, cx) -> (not IsBound(cx.specifiers_func) or cx.specifiers_func = []) and
                       (not IsBound(cx.func) or cx.func = []))), x->x.vars)), p -> p in candvars);

        invars := Filtered([X]::Flat(List(Collect(c, @@(1, decl,
            (e, cx) -> (not IsBound(cx.specifiers_func) or cx.specifiers_func = []) and
                       (not IsBound(cx.func) or cx.func = []))), x->x.vars)), p -> p in candvars);


        in_loops := Collect(cands, @(1, simt_loop, e -> Collect(e.cmd, @(2,simt_loop)) = [] and
            Collect(e.cmd, [assign, @(3), @(4, nth, f->f.loc in invars), ...]) <> []));

        substvars := Set(List(Collect(in_loops, [assign, @(1, nth, e -> ObjId(e.loc.t) = TArray), ...]), f -> f.loc.loc));

        link_loops := Collect(cands, @(1, simt_loop, e -> Collect(e.cmd, @(2,simt_loop)) = [] and
            Collect(e.cmd, [assign, @(3), @(4, nth, f->f.loc in substvars), ...]) <> []));

        linkvars := Set(List(Collect(link_loops, [assign, @(1, nth, e -> ObjId(e.loc.t) = TArray), ...]), f -> f.loc.loc));

        link_func := v -> Set(Collect(Filtered(Collect(c, @(1, specifiers_func)),
            e -> Collect(e.cmd, [assign, @(3), @(4, nth, f->f.loc = v), ...]) <> []), @(1, var, e-> e in outvars)))[1];

        subst_list := List(substvars, vv -> [vv, link_func(vv)]);

        substrec := FoldR(subst_list, (a,b) -> CopyFields(a, rec((b[1].id) := b[2])), rec());

        c := SubstBottomUp(c, @(1, decl), e-> decl(Filtered(@(1).val.vars, v -> not v in substvars), @(1).val.cmd));
        c := SubstVars(c, substrec);
    fi;

    return c;
end;


#FixUpCUDASigmaSPL_3Stage := (ss, opts) -> _FixUpCUDASigmaSPL_3Stage(_FixUpCUDASigmaSPL_3Stage(ss, opts), opts);


FixUpTeslaV_Code := function (c, opts)
    local kernels, kernel_inits, globals, var_decls, var_dels, cx, v, heap_init, dptr; 

    if IsBound(opts.fixUpTeslaV_Code) and opts.fixUpTeslaV_Code then
        kernels := List(Collect(c, specifiers_func), k->k.id);

        dptr := var.fresh_t("hp", TPtr(TReal));

#        kernel_inits := List(kernels, k-> call(rec(id := "INIT_KERNEL"), k));
#        #define INIT_KERNEL(k) cudaFuncSetCacheConfig(k, cudaFuncCachePreferEqual)
        kernel_inits := List(kernels, k-> call(rec(id := "cudaFuncSetCacheConfig"), k, "cudaFuncCachePreferEqual"));

#       cudaDeviceSetLimit(cudaLimitMallocHeapSize, (DEV_MIN_HEAP_SIZE));        
        heap_init := call(rec(id := "cudaDeviceSetLimit"), "cudaLimitMallocHeapSize", opts.max_heap);

        globals := Flat(List(Collect(c, @@(1, decl, (e, cx) -> (not IsBound(cx.specifiers_func) or cx.specifiers_func = []) and
                               (not IsBound(cx.func) or cx.func = []))), x->x.vars));

#        var_decls := List(globals, v -> call(rec(id := "DECLARE_DEVICE_ARRAY"), v.id, sizeof(v.t.t) * v.t.size));
#        double *hp;\
#        cudaMalloc((void**)&hp,(sz));\
#        cudaMemcpyToSymbol(p,&hp,sizeof(double*));\

        var_decls := chain(Flat(List(globals, v -> [ 
                call(rec(id := "cudaMalloc"), tcast(TPtr(TPtr(TVoid)), addrof(dptr)), sizeof(v.t.t) * v.t.size), 
                call(rec(id := "cudaMemcpyToSymbol"), v, addrof(dptr), sizeof(dptr.t)) 
            ])));
        
#        var_dels := List(globals, v -> call(rec(id := "DELETE_DEVICE_ARRAY"), v.id));
#        double *hp;\
#        cudaMemcpyFromSymbol(&hp,p,sizeof(double*));\
#        cudaFree((void**)&hp);\

        var_dels := decl(dptr, chain(Flat(List(globals, v -> [ 
                call(rec(id := "cudaMemcpyFromSymbol"), addrof(dptr), v, sizeof(dptr.t)), 
                call(rec(id := "cudaFree"), dptr)
            ]))));

        cx := chain([heap_init] :: kernel_inits :: [var_decls]);
        for v in globals do
            v.t := TPtr(v.t.t, ["__device__"]);
        od;

        c := SubstBottomUp(c, @(1, func, f -> f.id = "init"),
            e -> CopyFields(@(1).val, rec(cmd := decl(dptr, chain(cx, @(1).val.cmd))))
        );
        c := SubstBottomUp(c, @(1, func, f -> f.id = "destroy"),
            e -> CopyFields(@(1).val, rec(cmd := chain(var_dels, @(1).val.cmd)))
        );
        c := SubstBottomUp(c, @(1, chain, e -> ForAny(e.cmds, e -> ObjId(e) = func and e.id = "init")),
            e -> chain(Filtered(@(1).val.cmds, e-> ObjId(e) <> func or e.id <> "init") :: Filtered(@(1).val.cmds, e -> ObjId(e) = func and e.id = "init"))
        );

        
    fi;

    return c;
end;
