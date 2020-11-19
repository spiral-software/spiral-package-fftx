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
