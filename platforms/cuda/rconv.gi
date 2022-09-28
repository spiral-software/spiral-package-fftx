NewRulesFor(PrunedMDPRDFT, rec(
    PrunedMDPRDFT_tSPL_Base := rec(
        applicable     := nt -> Length(nt.params[1]) = 1,

        children       := nt -> [[ PrunedPRDFT(nt.params[1][1], nt.params[2]).withTags(nt.getTags()) ]],

        apply          := (nt, C, Nonterms) -> C[1]
    ),

    PrunedMDPRDFT_tSPL_RowCol1 := rec(
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children  := nt -> [[ PrunedMDDFT(DropLast(nt.params[1], 1), nt.params[3], 1, DropLast(nt.params[2], 1)),
                              PrunedPRDFT(Last(nt.params[1]), nt.params[3], 1, Last(nt.params[2])) ]],

        apply := (nt, C, cnt) ->  RC(Tensor(C[1], I(C[2].dims()[1]/2))) * Tensor(I(Product(List(cnt[1].params[4], i->Length(i)))), C[2])
    )
));


NewRulesFor(PrunedIMDPRDFT, rec(
    PrunedIMDPRDFT_tSPL_Base := rec(
        applicable     := nt -> Length(nt.params[1]) = 1,

        children       := nt -> [[ PrunedIPRDFT(nt.params[1][1], nt.params[2]).withTags(nt.getTags()) ]],

        apply          := (nt, C, Nonterms) -> C[1]
    ),

    PrunedIMDPRDFT_tSPL_RowCol1 := rec(
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children  := nt -> [[ PrunedIPRDFT(Last(nt.params[1]), nt.params[3], 1, Last(nt.params[2])),
                              PrunedIMDDFT(DropLast(nt.params[1], 1), nt.params[3], 1, DropLast(nt.params[2], 1)) ]],

        apply := (nt, C, cnt) -> Tensor(I(Product(List(cnt[2].params[4], i->Length(i)))), C[1]) * RC(Tensor(C[2], I(C[1].dims()[2]/2)))
    )
));


NewRulesFor(PrunedMDDFT, rec(
    PrunedMDDFT_tSPL_Base := rec(
        info := "PrunedMDDFT -> PrunedDFT",
        applicable     := nt -> Length(nt.params[1])=1,
        children       := nt -> let(P := nt.params, tags := nt.getTags(), [[ PrunedDFT(P[1][1], P[2], P[3], P[4][1]).withTags(tags) ]]),
        apply          := (nt, C, Nonterms) -> C[1]
    ),
    PrunedMDDFT_tSPL_RowCol := rec (
        info := "PrunedMDDFT_n -> PrunedMDDFT_n/d, PrunedMDDFT_d",
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children := nt -> let(
            dims := nt.params[1],
            len := Length(dims),
            pats := nt.params[4],
            List([1..len-1],
            i -> [ PrunedMDDFT(dims{[1..i]}, nt.params[2], nt.params[3], pats{[1..i]}), 
                   PrunedMDDFT(dims{[i+1..len]}, nt.params[2], nt.params[3], pats{[i+1..len]}) ])),

        apply := (nt, C, Nonterms) -> let(
            n1 := Cols(Nonterms[1]),
            n2 := Rows(Nonterms[2]),
            Tensor(C[1], I(n2)) *
            Tensor(I(n1), C[2])
        )
    )
));

NewRulesFor(PrunedIMDDFT, rec(
    PrunedIMDDFT_tSPL_Base := rec(
        info := "PrunedIMDDFT -> PrunedIDFT",
        applicable     := nt -> Length(nt.params[1])=1,
        children       := nt -> let(P := nt.params, tags := nt.getTags(), [[ PrunedIDFT(P[1][1], P[2], P[3], P[4][1]).withTags(tags) ]]),
        apply          := (nt, C, Nonterms) -> C[1]
    ),
    PrunedIMDDFT_tSPL_RowCol := rec (
        info := "PrunedIMDDFT_n -> PrunedIMDDFT_n/d, PrunedIMDDFT_d",
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children := nt -> let(
            dims := nt.params[1],
            len := Length(dims),
            pats := nt.params[4],
            List([1..len-1],
            i -> [ PrunedIMDDFT(dims{[1..i]}, nt.params[2], nt.params[3], pats{[1..i]}), 
                   PrunedIMDDFT(dims{[i+1..len]}, nt.params[2], nt.params[3], pats{[i+1..len]}) ])),

        apply := (nt, C, Nonterms) -> let(
            n1 := Cols(Nonterms[1]),
            n2 := Rows(Nonterms[2]),
            Tensor(C[1], I(n2)) *
            Tensor(I(n1), C[2])
        )
    )
));

NewRulesFor(IOPrunedMDRConv, rec(
    IOPrunedMDRConv_tSPL_InvDiagFwd := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> nt.hasTags() and Length(nt.params[1]) = 3 and IsFunc(nt.params[7]) and nt.params[7]()
                                    and nt.params[3] = 1 and nt.params[5] = 1, 
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nfreq := nlist[1]/2+1,
                            [[ TCompose([ 
                                PrunedIMDPRDFT(nt.params[1], nt.params[4], 1),
                                RCDiag(FDataOfs(nt.params[2].expr.loc, nt.params[2].vars[1].range, 0)),
                                PrunedMDPRDFT(nt.params[1], nt.params[6], -1)]).withTags(nt.getTags())
                            ]]),

       apply := (nt, C, cnt) -> C[1]
    ),

    IOPrunedMDRConv_tSPL_5stage := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> nt.hasTags() and Length(nt.params[1]) = 3 and IsFunc(nt.params[7]) and nt.params[7]()
                                    and nt.params[3] = 1 and nt.params[5] = 1, 

       children  := nt -> let(a_lengths := nt.params[1],
                               a_exp := nt.params[3],
                               tags := nt.getTags(),
                               iprdft := IPRDFT1(Last(a_lengths), a_exp),
                               prdft := PRDFT1(Last(a_lengths), a_exp),
                               rcdim := Rows(prdft),
                               rdim := Rows(iprdft),
                               cdim := Cols(iprdft),
                               #Error(),
                               [ [ TCompose([ TGrp(TCompose([
                                             TTensorI(PrunedIPRDFT(Last(a_lengths), a_exp, 1, Last(nt.params[2])), 
                                                Product(List(DropLast(nt.params[2], 1), Length)), APar, APar),
                                             TL(cdim * Product(List(DropLast(nt.params[2], 1), Length)) / 2, Product(List(DropLast(nt.params[2], 1), Length)), 1, 2), 
                                       ])) ] ::
                                       Reversed(List([1..Length(nt.params[1])-1], j->let(i := nt.params[1][j], 
                                           DropLast(a_lengths, 1), TRC(TTensorI(PrunedIDFT(i, a_exp,1, nt.params[2][j]), 
                                                cdim * Product(nt.params[1]{[j+1..Length(nt.params[2])-1]}) * Product(List(nt.params[2]{[1..j]}, Length))/(i), 
                                           APar, AVec)))))
                                      ::
                                      TDiag(nt.params[2])
                                      ::
                                       List([1..Length(nt.params[1])-1], j->
                                        let(i := nt.params[1][j], TRC(TTensorI(PrunedDFT(i, a_exp, 1, nt.params[2][j]), 
                                            rcdim * Product(nt.params[1]{[j+1..Length(nt.params[2])-1]}) * Product(List(nt.params[2]{[1..j]}, Length))/(i), 
                                            AVec, APar)))) ::
                                               [ TGrp(TCompose([TL(rcdim * Product(List(DropLast(nt.params[2], 1), Length)) / 2, rcdim / 2, 1, 2), 
                                                 TTensorI(PrunedPRDFT(Last(a_lengths), a_exp, 1, Last(nt.params[2])), 
                                                    Product(List(DropLast(nt.params[2], 1), Length)), APar, APar)
                                                 ])) ]).withTags(tags) ]] ),

        apply := (nt, C, cnt) -> C[1]
    )
    
));
#
#
#    ## GPU/TITAN V Hockney algotithm variant
#    ## 2-trip, 5-step, ZYX ====================================================
#    IOPrunedMDRConv_3D_2trip_zyx_freqdata := rec(
#       forTransposition := false,
#       applicable :=  (self, nt) >> not nt.hasTags() and Length(nt.params[1]) = 3 and nt.params[7],
#       children := nt -> let(nlist := nt.params[1],
#                            diag := nt.params[2],
#                            oblk := nt.params[3],
#                            opats := nt.params[4],
#                            iblk := nt.params[5],
#                            ipats := nt.params[6],
#                            nfreq := nlist[1]/2+1,
#                            i := Ind(nfreq*nlist[2]),
#                            hfunc := Cond(ObjId(diag) = Lambda,
#                                let(j := Ind(nlist[3]),
#                                    # Lambda(j, cxpack(diag.at(2*(j + i*nlist[3])), diag.at(2*(j + i*nlist[3])+1)))
#                                    pos := i +j*nfreq*nlist[2],
#                                    Lambda(j, cxpack(diag.at(2*pos), diag.at(2*pos+1)))
#                                ),
#                                ObjId(diag) = fUnk,
#                                fUnk(TComplex, nlist[3]),
#                                let(list := nt.params[2].list,  # here we assume FList(TReal, [...])
#                                    clist := List([1..Length(list)/2], i->Cplx(list[2*i-1], list[2*i])),
#                                    fc := FList(TComplex, clist),
#                                    gf := fTensor(fBase(i), fId(nlist[3])),
#                                    fCompose(fc, gf)
#                                )
#                            ),
#                            [[ PrunedPRDFT(nlist[1], -1, iblk, ipats[1]),  # stage 1: PRDFT z
#                                PrunedDFT(nlist[2], -1, iblk, ipats[2]),    # stage 2: DFT y
#                                IOPrunedConv(nlist[3], hfunc, oblk, opats[3], iblk, ipats[3], true), # stage 3+4+5: complex conv in x
#                                PrunedIDFT(nlist[2], 1, oblk, opats[2]), # stage 6: iDFT in y
#                                PrunedIPRDFT(nlist[1], 1, oblk, opats[1]),   # stage 7: iPRDFT in z
#                                InfoNt(i)
#                            ]]),
#
#       apply := (nt, C, cnt) -> let(prdft1d := C[1],
#                                    pdft1d := C[2],
#                                    iopconv := C[3],
#                                    ipdft1d := C[4],
#                                    iprdft1d := C[5],
#                                    i := cnt[6].params[1],
#                                    nlist := nt.params[1],
#                                    n1 := nlist[1],
#                                    nfreq := nlist[1]/2+1,
#                                    n2 := nlist[2],
#                                    n3 := nlist[3],
#                                    oblk := nt.params[3],
#                                    opats := nt.params[4],
#                                    iblk := nt.params[5],
#                                    ipats := nt.params[6],
#                                    ns1 := iblk * Length(ipats[1]),
#                                    ns2 := iblk * Length(ipats[2]),
#                                    ns3 := iblk * Length(ipats[3]),
#                                    nd1 := oblk * Length(opats[1]),
#                                    nd2 := oblk * Length(opats[2]),
#                                    nd3 := oblk * Length(opats[3]),
#                                    stage1 := L(2*nfreq*ns3*ns2, ns3) * Tensor(I(ns2), Tensor(L(2*nfreq, 2) * prdft1d, I(ns3))) * Tensor(L(ns2*ns1, ns2), I(ns3)),
#                                    stage2 := Tensor(I(ns3), Tensor(RC(pdft1d), I(nfreq))),
#                                    pp := Tensor(L(ns3*n2*nfreq, n2*nfreq), I(2)) * Tensor(I(ns3), L(2*nfreq*n2, nfreq)),
#                                    ppi := Tensor(I(nd3), L(2*nfreq*n2, 2*n2)) * Tensor(L(nd3*n2*nfreq, nd3), I(2)),
#                                    stage543 := ppi * IDirSum(i, RC(iopconv)) * pp,
#                                    stage76 := Tensor(L(nd2*nd1, nd1), I(nd3)) * Grp(Tensor((Tensor(I(nd2), iprdft1d * L(2*nfreq, nfreq)) *
#                                        Tensor(RC(ipdft1d), I(nfreq))), I(nd3)) * L(2*nfreq*nd3*n2, 2*nfreq*n2)),
#                                    conv3dr := stage76 * stage543 * stage2 * stage1,
#                                    conv3dr
#                            ),
#    ),
