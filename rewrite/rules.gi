
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

RewriteRules(RulesRC, rec(
    RC_ISumAcc := Rule([RC, @(1, ISumAcc)], e -> ISumAcc(@(1).val.var, @(1).val.domain, RC(@(1).val.child(1)))),
    RC_COND := Rule([RC, @(1, COND)], e -> ApplyFunc(COND, [ @(1).val.cond ]::List(@(1).val._children, i->RC(i)))),
));

RewriteRules(RulesFuncSimp, rec(
    H_fTensor_fBase_fId := ARule(fCompose, [@(1, H, e->e.domain()+1 = e.range() and e.params[4] = 1), 
        [@(0, fTensor), @(2, fBase), @(3, fId)]],
        e-> [fAdd(@(1).val.range(), @(3).val.domain(), @(1).val.params[3] + @(0).val.at(0))]
    ),
    fTensor_fBase_fId_H := Rule(@(1, fTensor, e->(
            ForAll(e.children(), c->ObjId(c) in [fBase, fId, H] and When(ObjId(c) = H, c.domain() = 1, true)) 
            and ForAny(e.children(), c->ObjId(c) = H))), 
        e->fTensor(List(@(1).val.children(), c->When(ObjId(c) = H, fBase(c.params[1], c.params[3]), c)))
    )
));

RewriteRules(RulesSums, rec(
    Scat_ISumAcc := ARule(Compose,
       [ @(1, Scat), @(2, ISumAcc) ],
        e -> [ CopyFields(@(2).val, 
        rec(_children :=  List(@(2).val._children, c -> ScatAcc(@(1).val.func) * c), dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

    PropagateBB := Rule([@(1,BB), @(2,Compose, e->ForAny(e.children(), i->ObjId(i)=COND))],
        e -> Compose(List(@(2).val.children(), i->Cond(ObjId(i) = COND, ApplyFunc(COND, [i.cond]::List(i.children(), j->BB(j))), BB(i))))),

    MergeBB := ARule(Compose, [@(1, BB), @(2, BB)], e-> [ BB(@(1).val.child(1) * @(2).val.child(1)) ]),

    COND_true := Rule(@(1, COND, e->e.cond = V(true)), e -> @(1).val.child(1)),

    eq_false := Rule([@(1, eq), [add, @(3, var, e->e.t=TInt), @(2, Value, e->e.v >0) ], @(0, Value, e->e.v=0)], e-> V(false)),

    eq_false2 := Rule([@(1, eq), @(2, add, e->ForAny(e.args, j->IsValue(j) and j.v > 0)), @(0, Value, e->e.v=0)], e-> V(false)),

    COND_false := Rule(@(1, COND, e->e.cond = V(false)), e -> @(1).val.child(2)),

    DiagISumAccLeft := ARule( Compose, [ @(1, ISumAcc, canReorder), @(2, RightPull) ],
        e -> [ ISumAcc(@1.val.var, @1.val.domain, @1.val.child(1) * @2.val).attrs(@(1).val) ]),

    COND_Diag :=  ARule( Compose, [ @(1, COND), @(2, [Diag, RCDiag]) ],
        e -> [ CopyFields(@(1).val, rec(_children := List(@(1).val._children, c -> c * @(2).val),
                dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ])
));


RewriteRules(RulesStrengthReduce, rec(
    mul_cond := Rule([@(1, mul), [cond, @(2), @(3, Value), @(4, Value)], @(5)],
        e->cond(@(2).val, @(5).val*@(3).val, @(5).val*@(4).val)),

    eq_add := Rule([eq, [add, @(1,Value), @(2)], @(3,Value)], e->eq(@(2).val, @(3).val-@(1).val)),

    eq_false := Rule([@(1, eq), [add, @(3, var, e->e.t=TInt), @(2, Value, e->e.v >0) ], @(0, Value, e->e.v=0)], e-> V(false)),

    eq_false2 := Rule([@(1, eq), @(2, add, e->ForAny(e.args, j->IsValue(j) and j.v > 0)), @(0, Value, e->e.v=0)], e-> V(false)),

    fix_cond := Rule([eq, [mul, @(0,Value, e->e.t=TInt and e.v>0), @(1,var,IsLoopIndex)], @(2, Value, e->e.t=TInt and e.v<0)], e->V(false)),
));

fixISumAccPDLoop := function(s, opts)
    local ss, vr, scat, dim, krnl, lvar, it0, itn, sct, isum, its, sx, scand, scats;
 
    while Length(Collect(s, ISumAcc)) >0  do
        scand := Collect(s, ISumAcc);
        ss := scand[1];
        vr := ss.var;
        scats := Collect(ss, ScatAcc);
        if Length(scats) > 1 then Error("There should be only one ScatAcc in an ISumAcc..."); fi;
        scat := scats[1];
        dim := Cols(scat);

        krnl := ss.child(1);
        lvar := Ind(vr.range-1);

        it0 := RulesSums(SubstTopDown(SubstVars(Copy(krnl), rec((vr.id) := V(0))), ScatAcc, e-> Scat(fId(dim))));
        itn := RulesSums(RulesSums(SubstTopDown(SubstVars(Copy(krnl), rec((vr.id) := lvar+1)), ScatAcc, e-> ScatAcc(fAdd(dim, dim, 0)))));
        sct := Scat(Collect(krnl, ScatAcc)[1].func);
        isum := ISum(lvar, itn);
        isum.doNotMarkBB := true;
        its := BB(sct) * SUM(it0, isum);

        s := SubstTopDown(s, ss, e->its);
        s := MergedRuleSet(RulesSums, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, RulesRC)(s);
    od;

    return s;
end;


fixScatter := function(s, opts)
    local tags, scts, sct, f, fl, vars, fll, rng, it, itspace, vals, vby2, vby2d, vby2d2, accf2, flnew, sctnew, srec;
    
    tags := rec();
    if IsBound(s.ruletree) then tags.ruletree  := s.ruletree; fi;
    
    # very brittle and danger of infinite loop
    scts := Collect(s, @(1, Scat, e->Cols(e) = 4 and Length(Collect(e, BH)) > 0 and ForAny(e.free(), i->i.t <> TInt))); 
    while Length(scts) > 0  do
        sct := scts[1];
    
        f := sct.func;
        fl := f.lambda();
        vars := Filtered(fl.free(), i->i.t = TInt);
        fll := fl.tolist();
        rng := List(vars, i->i.range);
        itspace := Cartesian(List(rng, i->[1..i]));
    
        vals := [];
        for it in itspace do
            srec := rec();
            for i in [1..Length(vars)] do
               srec := CopyFields(srec, rec((vars[i].id):=V(it[i]-1)));
            od;
            vals := vals :: List(Copy(fll), e->EvalScalar(SubstVars(e, srec)));
        od;
       
        vby2 := List([1..Length(vals)/2], i->vals[2*i-1]/2);
        vby2d := fTensor(FData(List(vby2, i->V(i))), fId(2));
    
        vby2d2 := FData(List(vby2, i->V(i)));
        accf2 := fTensor(List(vars, fBase) :: [fId(fl.vars[1].range/2)]);
        flnew := fTensor(fCompose(vby2d2, accf2).lambda().setRange(Rows(sct)/2), fId(2));
    
        sctnew := Scat(flnew);
        s := SubstTopDownNR(s, sct, e->sctnew);
        # very brittle and danger of infinite loop
        scts := Collect(s, @(1, Scat, e->Cols(e) = 4 and Length(Collect(e, BH)) > 0 and ForAny(e.free(), i->i.t <> TInt))); 
    od;
    s := CopyFields(tags, s);

    return s;
end;

fixGather := function(s, opts)
    local tags, gaths, gath, f, fl, vars, fll, rng, it, itspace, vals, vby2, vby2d, vby2d2, accf2, flnew, gathnew, srec;

    tags := rec();
    if IsBound(s.ruletree) then tags.ruletree  := s.ruletree; fi;

    # very brittle and danger of infinite loop
    gaths := Collect(s, @(1, Gath, e->Rows(e) = 4 and Length(Collect(e, BH)) > 0 and ForAny(e.free(), i->i.t <> TInt)));
    while Length(gaths) > 0  do
        gath := gaths[1];

        f := gath.func;
        fl := f.lambda();
        vars := Filtered(fl.free(), i->i.t = TInt);
        fll := fl.tolist();
        rng := List(vars, i->i.range);
        itspace := Cartesian(List(rng, i->[1..i]));

        vals := [];
        for it in itspace do
            srec := rec();
            for i in [1..Length(vars)] do
               srec := CopyFields(srec, rec((vars[i].id):=V(it[i]-1)));
            od;
            vals := vals :: List(Copy(fll), e->EvalScalar(SubstVars(e, srec)));
        od;

        vby2 := List([1..Length(vals)/2], i->vals[2*i-1]/2);
        vby2d := fTensor(FData(List(vby2, i->V(i))), fId(2));

        vby2d2 := FData(List(vby2, i->V(i)));
        accf2 := fTensor(List(vars, fBase) :: [fId(fl.vars[1].range/2)]);
        flnew := fTensor(fCompose(vby2d2, accf2).lambda().setRange(Cols(gath)/2), fId(2));

        gathnew := Gath(flnew);
        s := SubstTopDownNR(s, gath, e->gathnew);
        # very brittle and danger of infinite loop
        gaths := Collect(s, @(1, Gath, e->Rows(e) = 4 and Length(Collect(e, BH)) > 0 and ForAny(e.free(), i->i.t <> TInt)));
    od;
    s := CopyFields(tags, s);
    return s;
end;

fixInitFunc := function(c, opts)
    local cis, ci, ci2, fc, freei, offending, myd, tags;
   
    tags := rec();
    if IsBound(c.dimensions) then tags.dimensions  := c.dimensions; fi;
    if IsBound(c.ruletree) then tags.ruletree  := c.ruletree; fi;
    
    cis := Collect(c, @(1, func, e->e.id="init"));

    for ci in cis do
        fc := c.free();
        freei := ci.free();
        offending := Filtered(fc, i->i in freei);
        for myd in offending do
           ci2 := func(ci.ret, ci.id, ci.params,
               data(myd, myd.value, ci.cmd));
           c := SubstTopDown(c, @(1, func, e->e = ci), e->ci2);    
        od;
    od;    
    
    c := CopyFields(tags, c);
    return c;    
end;

fixUpSigmaSPL := function(s, opts)
    s := fixISumAccPDLoop(s, opts);
    s := fixScatter(s, opts);
    s := fixGather(s, opts);

    s := SubstTopDown(s, [@(1,RCDiag, e->e.dims()=[2,2]), 
           @(2, FDataOfs, e->IsBound(e.var.value) and e.var.value = V(Flat(List([1..e.var.value.t.size / 2], i->[V(1.0), V(0.0)])))), @(3, I)], e->I(2));
           
    s := SubstTopDown(s, [@(1,RCDiag, e->e.dims()=[4,4]), @(2, FDataOfs,
            e->IsBound(e.var.value) and ForAll([1..e.var.value.t.size/4], i->e.var.value.v[4*i-3] = V(1.0)) and ForAll([1..e.var.value.t.size/4],
            i->e.var.value.v[4*i- 2] = V(0.0))), @(3, I)],
        e->let(vals := V(Flat(List([1..@(2).val.var.value.t.size/4], i->[@(2).val.var.value.v[4*i-1], @(2).val.var.value.v[4*i]]))),
            dt := Dat(vals.t).setValue(vals),
            fdo := FDataOfs(dt, 2, V(2) * @(2).val.ofs.free()[1]),
            rcdnew  := RCDiag(fdo, I(2)),
            SumsSPL(DirectSum(I(2), rcdnew), opts))
    );
    return s;
end;

fixReplicatedData := function(c, opts)
    local datas, d, replicas, cc, r, tags;

    tags := rec();
    if IsBound(c.dimensions) then tags.dimensions  := c.dimensions; fi;
    if IsBound(c.ruletree) then tags.ruletree  := c.ruletree; fi;
    
    c  := SubstBottomUp(c, @(1, data, e->IsArray(e.value.t) and ObjId(e.cmd)=data and IsArray(e.cmd.value.t) and e.value.t.size < e.cmd.value.t.size),
        e->data(@(1).val.cmd.var, @(1).val.cmd.value, data(@(1).val.var, @(1).val.value, @(1).val.cmd.cmd)));

    datas := Collect(c, data);
    for d in datas do
        replicas := Collect(d.cmd, @(1, data, 
            e->IsBound(e.value.t.size) and e.value.t.size <= d.value.t.size and e.value.v = d.value.v{[1..e.value.t.size]}));
        for r in replicas do
            cc := r.cmd;
            cc := SubstVars(cc, rec((r.var.id) := d.var));
            c := SubstTopDown(c, @(1, data, e->e.var = r.var), e->cc);
        od;    
    od;
    c := CopyFields(tags, c);
    return c;
end;


scalarizeAccumulator := function(c, opts)
    local scand, svr, svars, tags;

    tags := rec();
    if IsBound(c.dimensions) then tags.dimensions  := c.dimensions; fi;
    if IsBound(c.ruletree) then tags.ruletree  := c.ruletree; fi;

    scand := Filtered(Set(Flat(List(Collect(c, decl), i->i.vars))), e->IsArray(e.t) and ForAll(Collect(c, @(1, nth, f-> f.loc = e)), g->IsValue(g.idx)));
    for svr in scand do
        svars := List([1..svr.t.size], e->var.fresh_t("q", svr.t.t));
        c := SubstTopDown(c, @(1, decl, e->svr in e.vars), e-> decl(svars :: Filtered(@(1).val.vars, i-> i <>svr), @(1).val.cmd));
        c := SubstTopDown(c, @(1, nth, e->e.loc = svr), e->svars[e.idx.v+1]);
    od;
    c := CopyFields(tags, c);
    return c;
end;
