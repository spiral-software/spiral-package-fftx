
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Class(CConv, TaggedNonTerminal, rec(
    abbrevs := [
        (n,h) -> Checked(IsPosIntSym(n), 
            [_unwrap(n), h, false]),
        (n,h,isFreqData) -> Checked(IsPosIntSym(n),
            [_unwrap(n), h, isFreqData])
        ],
    dims      := self >> [self.params[1], self.params[1]],
    terminate := self >> let(size := self.params[1],
        taps := When(self.params[7], MatSPL(DFT(self.params[1], -1)) * When(IsList(self.params[2]), self.params[2], self.params[2].list), self.params[2]),
        spiral.transforms.filtering.Circulant(size, taps, -size).terminate()),
    isReal    := self >> false,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(5 * n * d_log(n) / d_log(2))),
    TType := TComplex,
    hashAs := self >> ApplyFunc(ObjId(self), [self.params[1], fUnk(self.params[2].range(), self.params[2].domain())]::Drop(self.params, 2))
));

NewRulesFor(CConv, rec(
    CConv_DFT_IDFT := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> not nt.hasTags(),
       children := nt -> [[ DFT(nt.params[1], 1),
                            DFT(nt.params[1], -1) ]],
       apply := (nt, C, cnt) -> Cond(ObjId(nt.params[2]) = var, # a variable is frequency domain data -> FData
                                    C[1] * Diag(FData(nt.params[2])) * C[2], # if data is provided it needs to be frequency domain
                                    ObjId(nt.params[2]) = Lambda,  # data is inside a larger array
                                    C[1] * Diag(nt.params[2]) * C[2],
                                    nt.params[7],   # if params[7] is true then data is frequency domain data
                                    C[1] * Diag(When(IsList(nt.params[2]), FList(TComplex, nt.params[2]), nt.params[2])) * C[2],
                                    let(cxfftdiag := 1/nt.params[1]*ComplexFFT(List(nt.params[2].tolist(), i->ComplexAny(_unwrap(i)))),
                                        C[1] * Diag(FData(cxfftdiag)) * C[2]))
    )
));


NewRulesFor(MDRConv, rec(
    MDPRConv_Base := rec(
        info := "MDRConv -> IMDPRDFT * RCDiag(sym) * MDPRDFT",
        applicable     := nt -> nt.params[3] and IsVar(nt.params[2]), # need symbol to be in frequency domain

        children       := nt -> [[ MDPRDFT(nt.params[1], -1).withTags(nt.getTags()),
                                   IMDPRDFT(nt.params[1], 1).withTags(nt.getTags())
                                 ]],

        # nonterminal, children, children non-terminals
        apply          := (nt, C, Nonterms) -> C[2] * 
                            RCDiag(FDataOfs(nt.params[2], C[1].dims()[1], 0)) *
                            C[1]
    ),

    MDPRConv_Lambda := rec(
        info := "MDRConv -> IMDPRDFT * Pointwise(Lambda) * MDPRDFT",
        applicable     := nt -> nt.params[3] and IsLambda(nt.params[2]), # need symbol to be in frequency domain

        children       := nt -> [[ MDPRDFT(nt.params[1], -1).withTags(nt.getTags()),
                                   IMDPRDFT(nt.params[1], 1).withTags(nt.getTags())
                                 ]],

        # nonterminal, children, children non-terminals
        apply          := (nt, C, Nonterms) -> C[2] * 
                            Pointwise(nt.params[2]) *
                            C[1]
    ),

    MDRConv_3D_2trip_zyx_freqdata := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> not nt.hasTags() and Length(nt.params[1]) = 3 and nt.params[3],
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            nfreq := nlist[1]/2+1,
                            i := Ind(nfreq*nlist[2]),
                            hfunc := Cond(ObjId(diag) = Lambda,
                                let(j := Ind(nlist[3]),
                                    # Lambda(j, cxpack(diag.at(2*(j + i*nlist[3])), diag.at(2*(j + i*nlist[3])+1)))
                                    pos := i +j*nfreq*nlist[2],
                                    Lambda(j, cxpack(diag.at(2*pos), diag.at(2*pos+1)))
                                ),
                                IsVar(diag),
                                let(j := Ind(nlist[3]),
                                    pos := i +j*nfreq*nlist[2],
                                    Lambda(j, cxpack(nth(diag, 2*pos), nth(diag, 2*pos+1))).setRange(TComplex)
                                ),
                                ObjId(diag) = fUnk,
                                fUnk(TComplex, nlist[3]),
                                let(list := nt.params[2].list,  # here we assume FList(TReal, [...])
                                    clist := List([1..Length(list)/2], i->Cplx(list[2*i-1], list[2*i])),
                                    fc := FList(TComplex, clist),
                                    gf := fTensor(fBase(i), fId(nlist[3])),
                                    fCompose(fc, gf)
                                )
                            ),
                            [[ PRDFT(nlist[1], -1),  # stage 1: PRDFT z
                               DFT(nlist[2], -1),    # stage 2: DFT y
                               CConv(nlist[3], hfunc), # stage 3+4+5: complex conv in x
                               DFT(nlist[2], 1), # stage 6: iDFT in y
                               IPRDFT(nlist[1], 1),   # stage 7: iPRDFT in z
                               InfoNt(i)
                            ]]),

       apply := (nt, C, cnt) -> let(prdft1d := C[1],
                                    pdft1d := C[2],
                                    iopconv := C[3],
                                    ipdft1d := C[4],
                                    iprdft1d := C[5],
                                    i := cnt[6].params[1],
                                    nlist := nt.params[1],
                                    n1 := nlist[1],
                                    nfreq := nlist[1]/2+1,
                                    n2 := nlist[2],
                                    n3 := nlist[3],
                                    stage1 := L(2*nfreq*n3*n2, n3) * Tensor(I(n2), Tensor(L(2*nfreq, 2) * prdft1d, I(n3))) * Tensor(L(n2*n1, n2), I(n3)),
                                    stage2 := Tensor(I(n3), Tensor(RC(pdft1d), I(nfreq))),
                                    pp := Tensor(L(n3*n2*nfreq, n2*nfreq), I(2)) * Tensor(I(n3), L(2*nfreq*n2, nfreq)),
                                    ppi := Tensor(I(n3), L(2*nfreq*n2, 2*n2)) * Tensor(L(n3*n2*nfreq, n3), I(2)),
                                    stage543 := Grp(ppi * IDirSum(i, RC(iopconv)) * pp),
                                    stage76 := Tensor(L(n2*n1, n1), I(n3)) * Grp(Tensor((Tensor(I(n2), iprdft1d * L(2*nfreq, nfreq)) * 
                                       Tensor(RC(ipdft1d), I(nfreq))), I(n3)) * L(2*nfreq*n3*n2, 2*nfreq*n2)),
                                    conv3dr := stage76 * stage543 * stage2 * stage1,
                                    conv3dr
                            )
    )
));


NewRulesFor(MDRConvR, rec(
    MDPRConvR_Base := rec(
        info := "MDRConv -> IMDPRDFT * Diag(sym) * MDPRDFT",
        applicable     := nt -> nt.params[3], # need symbol to be in frequency domain

        children       := nt -> [[ MDPRDFT(nt.params[1], -1).withTags(nt.getTags()),
                                   IMDPRDFT(nt.params[1], 1).withTags(nt.getTags())
                                 ]],

        # nonterminal, children, children non-terminals
        apply          := (nt, C, Nonterms) -> C[2] * 
                            Diag(diagTensor(FDataOfs(nt.params[2], Product(DropLast(nt.params[1], 1))* (Last(nt.params[1])/2+1), 0), fConst(TReal, 2, 1))) *
                            C[1]
    )
));
