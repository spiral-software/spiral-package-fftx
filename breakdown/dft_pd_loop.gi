
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

NewRulesFor(DFT, rec(

    DFT_PD_loop := rec(
        forTransposition := false,
        minSize := 7,
        maxSize := 13,
    
        TabPerm := r -> let(i:=Ind(r.domain()), f := FData(r.tolist()).at(i), Lambda(i, f).setRange(r.range())),
    
        applicable     := (self, nt) >> nt.params[1] > 2 and nt.params[1] in [self.minSize..self.maxSize] and IsPrime(nt.params[1]) and not nt.hasTags(),
    
        apply := (self, nt, C, cnt) >> let(
            N := nt.params[1], 
            k := nt.params[2], 
            root := PrimitiveRootMod(N),
            M:=(N-1)/2,
            i := Ind(M),
            j := Ind(M),
            k1 := Ind(M+1),
            m := MatSPL(DFT_PD.core(N, k, root, false) * DFT_PD.A(N)),
            m1 := Map(m{[2..(N+1)/2]}, r -> r{[2..(N+1)/2]}),
            m2 := Map(m{[(N+3)/2..N]}, r -> r{[(N+3)/2..N]}),
            d := Flat(m1)::Map(Flat(m2), c->im(ComplexAny(c)).v),
            fd := FData(d),
            
            s := Scat(self.TabPerm(RR(N, 1, root))),
            g := Gath(self.TabPerm(RR(N, 1, 1/root mod N))),

            gg := Gath(fCompose(fAdd(N, N-1, 1), fTensor(fId(2), fBase(M, k1-1)))),
            f2 := F(2),
            u := Ind(2),
            uf := Lambda(u, fd.at(u*M*M + M*i+(k1-1))),
            bb := DirectSum(Blk1(uf.at(0)), Scale(ImaginaryUnit(), Blk1(uf.at(1)))),
            
            krn0 := Mat([[1],[0]]) * Gath(fAdd(N, 1, 0)),
            krnn := bb * f2 * gg,
            krn := ScatAcc(fId(2)) * f2 * COND(eq(k1,0), krn0, krnn),
            
            q1 := L(2*M, 2) * IterVStack(i, ISumAcc(k1, krn)),
            #q2 := RowVec(fConst(13,V(1.0))),
            _i := Ind(N),
            q2 := ISumAcc(_i, ScatAcc(fAdd(1,1,0)) * Blk([[1]]) * COND(eq(_i, 0), Gath(fAdd(13,1,0)), Gath(fAdd(13,1,_i)))), 
            q3 := VStack(q2, q1 * g),
            qq := s * q3,            
            qq
        )
    )
));

RulesFor(PRDFT, rec(
   PRDFT_PD_loop := rec(
	forTransposition := false,
    minSize := 7,
	maxSize          := 13,
	isApplicable     := (self, P) >> P[1] > 2 and P[1] in [self.minSize..self.maxSize] and IsPrime(P[1]),
	
	rule := (self,P,C) >> let(N:=P[1], n:=N-1, k:=P[2], root:=PrimitiveRootMod(N),
        M:=n/2,
        i := Ind(M),
        j := Ind(M),
        k1 := Ind(M+1),
        m := MatSPL(DFT_PD.core(N, k, root, false) * DFT_PD.A(N)),
        m1 := Map(m{[2..(N+1)/2]}, r -> r{[2..(N+1)/2]}),
        m2 := Map(m{[(N+3)/2..N]}, r -> r{[(N+3)/2..N]}),
        d := Flat(m1)::Map(Flat(m2), c->im(ComplexAny(c)).v),
        fd := FData(d),
        #lfd1 := Lambda(j, fd.at(M*i+j)),
        #lfd1 := Lambda(k1, cond(eq(k1, V(0)), V(1.0), fd.at(M*i+(k1-V(1))))),
        #lfd2 := Lambda(j, fd.at(M*M+M*i+j)),
        
        gf := DFT_PD_loop.TabPerm(RR(N, 1, root)),
        g := Gath(gf),
        
        #kk1 := DirectSum(RowVec(lfd1), RowVec(lfd2)) * DirectSum(I(1), Tensor(F(2), I(M)) * OS(n, -1)),
        #q1 :=  IterVStack(i, BB(kk1)),
        
        #lbd := OS(12, -1).lambda(),
        _j := Ind(n),
        lbd := Lambda(_j, imod(V(n)-_j, V(n))),
        gg := Gath(fCompose(fAdd(N, N-1, 1), fCompose(lbd, fTensor(fId(2), fBase(M, k1-1))))),

        f2 := F(2),
        u := Ind(2),
        uf := Lambda(u, fd.at(u*M*M + M*i+(k1-1))),
        bb := DirectSum(Blk1(uf.at(0)), Blk1(uf.at(1))),

        krn0 := Mat([[1],[0]]) * Gath(fAdd(N, 1, 0)),
        krnn := bb * f2 * gg,
        krn := Grp(ScatAcc(fId(2))) * COND(eq(k1,0), krn0, krnn),

        q1 := IterVStack(i, ISumAcc(k1, krn)),
        q2a := RowVec(fConst(13,V(1.0))),
#        _i := Ind(N),
#        q2a := ISumAcc(_i, ScatAcc(fAdd(1,1,0)) * Blk([[1]]) * COND(eq(_i, 0), Gath(fAdd(13,1,0)), Gath(fAdd(13,1,_i)))), 
        q2b := RowVec(fConst(13,V(0.0))),
        q3 := VStack(BB(VStack(q2a, q2b)), q1),
        sf := DFT_PD_loop.TabPerm(Refl((N+1)/2, N, (N+1)/2, RR(N,1,root))),
        s := Scat(fTensor(sf, fId(2))),
        
        u1 := Ind(N+1),
        df := Lambda(u1, cond(logic_and(eq(V(1), bin_and(u1, 1)), geq(gf.at(idiv(u1, 2)), (N+1)/2)), V(-1), V(1))),
        dfl := df.tolist(),
        m1s := Filtered([0..N], _i->dfl[_i+1] = V(-1)),
        m1eq := List(m1s, _i->eq(u1, V(_i))),
        cnd := ApplyFunc(logic_or, m1eq),
        lbdcnd := Lambda(u1, cond(cnd, V(-1), V(1))),
        diag := Diag(lbdcnd),
        sct := s * diag,
        qq := sct * q3 * g,
        qq
    )
)));


RulesFor(IPRDFT, rec(
   IPRDFT_PD_loop := rec(
	forTransposition := false,
    minSize := 7,
	maxSize          := 13,
	isApplicable     := (self, P) >> P[1] > 2 and P[1] in [self.minSize..self.maxSize] and IsPrime(P[1]),
	
	rule := (self,P,C) >> let(N:=P[1], n:=N-1, k:=P[2], root:=PrimitiveRootMod(N), M:=n/2,
        # loop variables
        ii := Ind(M),
        k1 := Ind(M+1),
        u := Ind(2),
        # scatter
        fstr := fDirsum(fId(1), L(n, n/2)),
        fos := fDirsum(fId(2), J(N-2)),
        frr := RR(N, 1, root),
        fsct := fCompose(frr, fos, fstr),
        fl := fsct.tolist(){[2..N]},
        flst := FList(N-1, fl),
        fst :=  DFT_PD_loop.TabPerm(flst),
        i := Ind(N),
        flbd := Lambda(i, cond(eq(i, V(0)), V(0), fst.at(i-V(1)))).setRange(N),
        sct := Scat(flbd),
        # diagonal sign flip and block matrix
        rr := RealRR_Out(N, root).transpose(),
        dd := rr.child(1),
        mc := TransposedSPL(DFT_PD.core(N, k, root, false) * DFT_PD.A(N)) * DirectSum(Mat([[1,0]]), 2*I(n)),
        dd2 := DirectSum(I(2), L(n, 2)) * dd * DirectSum(I(2), L(n, n/2)),
        m := MatSPL(mc * dd2),
        m1 := Map(m{[2..(N+1)/2]}, r -> r{[3..(N+1)/2+1]}),
        m2 := Map(m{[(N+3)/2..N]}, r -> Map(r{[(N+3)/2+1..N+1]}, c->-im(ComplexAny(c)).v)),
        d := Flat(m1)::Flat(m2),
        fd := FData(d),
        # gather
        gth := rr.child(2),
        gf := gth.func,
        gfl := gf.tolist(){2*[1..(N-1)/2]+1},
        gst := FList((N-1)/2, gfl),
        gftb := DFT_PD_loop.TabPerm(gst).setRange(N+1),
        glbd := Lambda(u, gftb.at(k1-1) + u).setRange(N+1),
        gath := Gath(glbd),
        # kernel
        uf := Lambda(u, fd.at(u*M*M + M*ii+(k1-1))),
        bb := DirectSum(Blk1(uf.at(0)), Blk1(uf.at(1))),
        f2 := F(2),
        krn0 := f2 * Mat([[1],[0]]) * Gath(fAdd(N+1, 1, 0)),
        krnn := f2 * bb * gath,
        krn := Grp(ScatAcc(fId(2))) * COND(eq(k1,0), krn0, krnn),
        # stack
        q1 := IterVStack(ii, ISumAcc(k1, krn)),
        r1 := RowVec(diagDirsum(fConst(1,V(1.0)), fConst(n/2,V(2.0)))) * Gath(fTensor(fId((N+1)/2), fBase(2, 0))),
        qq := VStack(r1, q1),
        qq1 := sct * qq,
        qq1
    ))
));


