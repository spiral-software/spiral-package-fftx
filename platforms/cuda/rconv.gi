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
                                TDiag(nt.params[2]),
                                PrunedMDPRDFT(nt.params[1], nt.params[6], -1)]).withTags(nt.getTags())
                            ]]),

       apply := (nt, C, cnt) -> C[1]
    ),
));


