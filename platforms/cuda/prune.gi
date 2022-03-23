NewRulesFor(PrunedMDDFT, rec(
    MDDFT_tSPL_RowCol_SIMT := rec(
        info :="MDDFT(n_1,n_2,...,n_t) = (RC(MDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x PRDFT1(n_t))",
        applicable := nt->nt.hasTags() and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.params[1]) > 1,
        children  := nt -> let(a_lengths := nt.params[1],
                               a_exp := nt.params[2],
                               tags := nt.getTags(),
                               [[ TTensorI(MDDFT(DropLast(a_lengths, 1), a_exp), Last(a_lengths), AVec, AVec).withTags(tags),
                                  FoldR(DropLast(a_lengths, 1), (a,b)->TTensorI(a, b, APar, APar), DFT(Last(a_lengths), a_exp)).withTags(tags) ]]),
        apply := (nt, C, cnt) -> C[1] * C[2]
    )
));


NewRulesFor(PrunedMDPRDFT, rec(
    PrunedMDPRDFT_tSPL_RowCol_SIMT := rec(
        info :="MDDFT(n_1,n_2,...,n_t) = (RC(MDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x PRDFT1(n_t))",
        applicable := nt->nt.hasTags() and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.params[1]) > 1,
        children  := nt -> let(a_lengths := nt.params[1],
                               a_exp := nt.params[2],
                               tags := nt.getTags(),
                               mddft := MDDFT(DropLast(a_lengths, 1), a_exp),
                               prdft := PRDFT1(Last(a_lengths), a_exp),
                               [[ TTensorI(mddft, prdft.dims()[1]/2, AVec, AVec).withTags(tags),
                                  FoldR(DropLast(a_lengths, 1), (a,b)->TTensorI(a, b, APar, APar), prdft).withTags(tags) ]]),
        apply := (nt, C, cnt) -> RC(C[1]) * C[2]
    ),
));


NewRulesFor(PrunedIMDPRDFT, rec(
    PrunedIMDPRDFT_tSPL_RowCol_SIMT := rec(
        info :="IMDDFT(n_1,n_2,...,n_t) = (RC(IMDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x PRDFT1(-n_t))",
        applicable := nt->nt.hasTags() and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.params[1]) > 1,
        children  := nt -> let(a_lengths := nt.params[1],
                               a_exp := nt.params[2],
                               tags := nt.getTags(),
                               mddft := MDDFT(DropLast(a_lengths, 1), a_exp),
                               iprdft := IPRDFT1(Last(a_lengths), a_exp),
                               [[ FoldR(DropLast(a_lengths, 1), (a,b)->TTensorI(a, b, APar, APar), iprdft).withTags(tags),
                                   TTensorI(mddft, iprdft.dims()[2]/2, AVec, AVec).withTags(tags) ]]),
        apply := (nt, C, cnt) -> C[1] * RC(C[2])
    ),
));
