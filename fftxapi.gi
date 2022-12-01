
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

_orderedUniquify := l-> Flat([l[1]]::List([2..Length(l)], i->When(l[i] in l{[1..i-1]}, [], [l[i]])));

Class(FFTXGenMixin, rec(
    search := (self, t) >> When(IsBound(self.useDP) and self.useDP, DP(t, When(IsBound(self.dpopts), self.dpopts, rec()), self)[1].ruletree, RuleTreeMid(t, self)),
    preProcess := (self, t) >> let(t1 := RulesFFTXPromoteNT(Copy(t)), RulesFFTXPromoteNT_Cleanup(t1)),
    
    codeSumsCPU := meth(self, ss) 
                        local opts2, ss2, c;
                        
                        opts2 := Copy(SpiralDefaults);
                        opts2.useDeref := false;
#                        opts2.codegen := VecRecCodegen;
#                        opts2.unparser := SSEUnparser;
                        
                        ss2 := Copy(ss);
                        
                        ss2 := SubstBottomUp(ss2, @(1, SIMTISum),
                            e -> ISum(@(1).val.var, @(1).val.domain, @(1).val.child(1)));
                            
                        self.debug.ss_cpu := ss2;    
                            
                        c := CodeSums(ss2, opts2);
                        c := fixReplicatedData(c, opts2);

                        self.cpu_opts := opts2;
                        c.dimensions := ss.dims();
                        
                        return c;
                    end,

    prettyPrintCPU := meth(self, c)
                       local name;
                       name := "fftx_generated";
                       if (IsBound(c.ruletree) and IsBound(c.ruletree.node.params[2].fname)) then
                           name := c.ruletree.node.params[2].fname;
                       fi;
                       PrintCode(name, c, self.cpu_opts);
                   end,

    fftxGenCPU := meth(self, t) 
                   local tt, rt, c, s, r;
				   if Length(t.params) > 1 then
				       r := t.params[2];
					   if IsRec(r) and IsBound(r.fname) then
					      self.cudasubName := r.fname;
					   fi;
				   fi;

                   self.debug := rec();  
                   tt := self.preProcess(t);
                   self.debug.tt := tt;
                   rt := self.search(tt);
                   self.debug.rt := rt;
                   s := self.sumsRuleTree(rt);
                   self.debug.ss := s;
                   
                   c := self.codeSumsCPU(s);
                   c := SubstBottomUp(c, @(1,decl), e->let(fr := @(1).val.cmd.free(), decl(Filtered(e.vars, v->v in fr), @(1).val.cmd)));
                   
                   self.debug.c := c;
                   c.ruletree := rt;
                   c.dimensions := s.dims();
                   
                   return c;
               end,    
               
    cmeasureCPU := (self, c) >> CMeasure(c, self.cpu_opts),           
    
    codeSums := meth(self, ss) 
                        local c, cc, Xptr, Yptr, plist, plist1, tags;

                        if IsBound(self.Xptr) then
                            Xptr := self.Xptr;
                        else
                            Xptr := X;
                        fi;
                        if IsBound(self.Yptr) then
                            Yptr := self.Yptr;
                        else
                            Yptr := Y;
                        fi;
                       
                        c := CodeSums(ss, self);
                        tags := rec();
                        if IsBound(c.dimensions) then tags.dimensions  := c.dimensions; fi;
                        if IsBound(c.ruletree) then tags.ruletree  := c.ruletree; fi;

                        c := SubstVars(c, rec(X := Xptr, Y := Yptr));
                        cc := fixReplicatedData(c, self);
                        cc := Rewrite(cc, RulesDropSymDecl, self);
                        
                        plist := Collect(cc, @(1,func, e->e.id="transform"))[1].params;
                        plist1 := _orderedUniquify(plist);
                        if Length(plist1) = 2 then 
                            plist1 := plist1 :: self.symbol; 
                        fi;
                        cc := SubstTopDown(cc, @(1).cond(e->IsList(e) and e = plist), e->plist1);
                       
                        cc := fixReplicatedData(cc, self);
                        cc := SubstTopDown(cc, @(1, Value, e->e.t = TReal and IsDouble(e.v) and let(delta := AbsFloat(e.v - IntDouble(e.v)), delta >0 and delta  < 0.00000000001 )),
                            e->Value(TReal, IntDouble(@(1).val.v))
                        );

                        if IsBound(self.postProcessCode) then
                            cc := self.postProcessCode(cc, self);
                        fi;

                        cc := CopyFields(tags, cc);
                        return cc;
                    end,
                    
    sumsRuleTree := meth(self, rt) 
                        local  s, s2, ss, Xptr, Yptr, tags;

                        self.symbol := When(IsBound(rt.node.params[2].params), rt.node.params[2].params, []);

                        if IsBound(rt.node.params[2].XType) then
                            Xptr := var("Xptr", rt.node.params[2].XType);
                        else
                            Xptr := X;
                        fi;
                        if IsBound(rt.node.params[2].YType) then
                            Yptr := var("Yptr", rt.node.params[2].YType);
                        else
                            Yptr := Y;
                        fi;
                       
                        self.Xptr := Xptr;
                        self.Yptr := Yptr;
                         
                        s := SumsRuleTree(rt, self);
                        tags := rec(ruletree := rec());
                        if IsBound(s.ruletree) then tags.ruletree  := s.ruletree; fi;

                        s2 := fixUpSigmaSPL(s, self);
                       
                        ss := SubstVars(s2, rec(X := Xptr, Y := Yptr));
                        ss := SubstTopDown(ss, @(1, [NoDiagPullinLeft, NoDiagPullinRight, Grp]), e->e.child(1));
                        if IsBound(self.postProcessSums) then ss := self.postProcessSums(ss, self); fi;
                        ss.ruletree := tags.ruletree;
                        
                        return ss;
                    end,

    prettyPrint := meth(self, c)
                       local name;
                       name := "fftx_generated";
                       if (IsBound(c.ruletree) and IsBound(c.ruletree.node.params[2].fname)) then
                           name := c.ruletree.node.params[2].fname;
                       fi;
                       PrintCode(name, c, self);
                   end,
                   
    fftxGen := meth(self, t) 
                   local tt, rt, c, s, r;
				   if Length(t.params) > 1 then
				       r := t.params[2];
					   if IsRec(r) and IsBound(r.fname) then
					      self.cudasubName := r.fname;
					   fi;
				   fi;
                   tt := self.preProcess(t);
                   rt := self.search(tt);
                   s := self.sumsRuleTree(rt);
                   c := self.codeSums(s);
                   return c;
               end    
));


_tofAdd := (n, l) -> When(IsList(l), let(mn := Minimum(l), mx := Maximum(l), Checked(ForAll([mn..mx], i->i in l), fAdd(n, mx-mn+1, mn))), 
                          When(n <> l, fAdd(n, l, 0), fId(n)));
_toBox := (ns, nps)-> When(IsList(ns), fTensor(List(Zip2(ns, nps), i->ApplyFunc(_tofAdd, i))), _tofAdd(ns, nps));

ZeroEmbedBox := (ns, nps) -> Scat(_toBox(ns, nps));
ExtractBox := (ns, nps) -> Gath(_toBox(ns, nps));
BoxND := (l, stype) -> Cond(IsInt(l), TArray(stype, l), 
                             IsList(l) and Length(l) = 1, TArray(stype, l[1]), 
                             TArray(BoxND(Drop(l, 1), stype), l[1]));

BoxNDcmaj := (l, stype) -> TColMaj(BoxND(l, stype));
                             
fBox := l -> fTensor(List(l, fId));
fBoxInBox := (l2, l1, l3) ->fTensor(List([1..Length(l1)], _l -> fAdd(l1[_l], l2[_l], let(shft := l3[_l], When(IsInt(shft), shft, IntDouble(shft))))));

#_cplx := (r,i) -> r + E(4)*i;
#_imag := i -> E(4) * i;
_real := r -> r;
M_PI := d_acos(-1);

_cplx := (r,i) -> cxpack(r,i);
_imag := i -> cxpack(0, i);

TMapPar := (krn, dims) -> When(Length(dims) = 0, krn, let(tti := When(IsVar(dims[1]), TTensorInd, TTensorI), ApplyFunc(tti, [TMapPar(krn, Drop(dims, 1)), dims[1], APar, APar])));
TMap := (krn, dims, al, ar) -> TCompose(
    When(al = AVec, [TL(Product(List(dims, i->i.range)) * Rows(krn), Rows(krn), 1, 1)], []) ::
    [ TMapPar(krn, dims)] ::
    When(ar = AVec, [TL(Product(List(dims, i->i.range)) * Cols(krn), Product(List(dims, i->i.range)), 1, 1)], []));

lin_idx := arg -> fTensor(List(arg, fBase)).at(0);

TIterator1D := (A, n, l,  r) -> When(n = 1, A, TTensorI(A, n, l, r));

_gflops := (n, b, t) -> (b*5*n*LogInt(n, 2))/(t*1000000);

_idiv := (a, b) -> When(b = 2, bin_shr(a, LogInt(b, 2)), idiv(a,b));
_imod := (a, b) -> When(b = 2, bin_and(a, b-1), imod(a, b));
_fdiv := (a, b) -> cond(leq(abs(b), var("DBL_EPSILON")), V(0.0), fdiv(a, b));

StepPhase_Pointwise := (domain, symvar, c) -> let(
    i := Ind(domain),
    x := var.fresh_t("x", TPtr(TReal)),
    pw_op := (cval, a) -> a * cxpack(
        _fdiv(re(cval), sqrt(re(cval)*re(cval) +im(cval)*im(cval))), 
        _fdiv(im(cval), sqrt(re(cval)*re(cval) +im(cval)*im(cval))) 
    ),
    cx_nth := (xr, i) -> cxpack(nth(xr, _idiv(i, 2) * 2), nth(xr, (_idiv(i, 2) * 2) + 1)),
    extract := (cval, i) -> cond(eq(_imod(i, 2), V(0)), re(cval), im(cval)),
    ampli := FDataOfs(symvar, domain, 0),
    Pointwise(Lambda(i, Lambda(x, c * cond(leq(i, V(1)), nth(x, i), extract(pw_op(cx_nth(x, i), ampli.at(_idiv(i, 2))), i)))))
);

