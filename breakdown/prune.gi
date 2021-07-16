

NewRulesFor(PrunedMDPRDFT, rec(
    PrunedMDPRDFT_Base := rec(
        info := "MDPRDFT -> PRDFT1",
        applicable     := nt -> Length(nt.params[1]) = 1,

        # Generally, children is list of ordered lists of all possible children.
        children       := nt -> [[ PrunedPRDFT(nt.params[1][1], nt.params[2]).withTags(nt.getTags()) ]],

        # nonterminal, children, children non-terminals
        apply          := (nt, C, Nonterms) -> C[1]
    ),

    MDPRDFT_RowCol1 := rec(
        info :="MDDFT(n_1,n_2,...,n_t) = (RC(MDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x PRDFT1(n_t))",
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children  := nt -> [[ PrunedMDDFT(DropLast(nt.params[1], 1), nt.params[3], 1, DropLast(nt.params[2], 1)),
                              PrunedPRDFT(Last(nt.params[1]), nt.params[3], 1, Last(nt.params[2])) ]],

        apply := (nt, C, cnt) ->  RC(Tensor(C[1], I(C[2].dims()[1]/2))) * Tensor(I(Product(List(cnt[1].params[4], i->Length(i)))), C[2])
    )
));

