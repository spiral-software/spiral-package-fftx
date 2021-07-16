

NewRulesFor(PrunedMDPRDFT, rec(
    PrunedMDPRDFT_Base := rec(
        applicable     := nt -> Length(nt.params[1]) = 1,

        children       := nt -> [[ PrunedPRDFT(nt.params[1][1], nt.params[2]).withTags(nt.getTags()) ]],

        apply          := (nt, C, Nonterms) -> C[1]
    ),

    PrunedMDPRDFT_RowCol1 := rec(
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children  := nt -> [[ PrunedMDDFT(DropLast(nt.params[1], 1), nt.params[3], 1, DropLast(nt.params[2], 1)),
                              PrunedPRDFT(Last(nt.params[1]), nt.params[3], 1, Last(nt.params[2])) ]],

        apply := (nt, C, cnt) ->  RC(Tensor(C[1], I(C[2].dims()[1]/2))) * Tensor(I(Product(List(cnt[1].params[4], i->Length(i)))), C[2])
    )
));


NewRulesFor(PrunedIMDPRDFT, rec(
    PrunedIMDPRDFT_Base := rec(
        applicable     := nt -> Length(nt.params[1]) = 1,

        children       := nt -> [[ PrunedIPRDFT(nt.params[1][1], nt.params[2]).withTags(nt.getTags()) ]],

        apply          := (nt, C, Nonterms) -> C[1]
    ),

    PrunedIMDPRDFT_RowCol1 := rec(
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children  := nt -> [[ PrunedIPRDFT(Last(nt.params[1]), nt.params[3], 1, Last(nt.params[2])),
                              PrunedIMDDFT(DropLast(nt.params[1], 1), nt.params[3], 1, DropLast(nt.params[2], 1)) ]],

        apply := (nt, C, cnt) -> Tensor(I(Product(List(cnt[2].params[4], i->Length(i)))), C[1]) * RC(Tensor(C[2], I(C[1].dims()[2]/2)))
    )
));



