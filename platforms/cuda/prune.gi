NewRulesFor(PrunedMDPRDFT, rec(
    PrunedMDPRDFT_tSPL_Pease_SIMT := rec(
        applicable := nt->nt.hasTags() and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.params[1]) > 1,
        children  := nt -> let(a_lengths := nt.params[1],
                               a_exp := nt.params[3],
                               tags := nt.getTags(),
                               prdft := PRDFT1(Last(a_lengths), a_exp),
                               rcdim := Rows(prdft),
                               [ [ TCompose(List([1..Length(nt.params[1])-1], j->
                                let(i := nt.params[1][j], TRC(TTensorI(PrunedDFT(i, a_exp, 1, nt.params[2][j]), rcdim * Product(DropLast(nt.params[1], 1))/(2*i), AVec, APar))))::
                                           [ TGrp(TCompose([TL(rcdim * Product(DropLast(nt.params[1], 1)) / 2, rcdim / 2, 1, 2), 
                                             TTensorI(PrunedPRDFT(Last(a_lengths), a_exp, 1, Last(nt.params[2])), Product(DropLast(nt.params[1], 1)), APar, APar)
                                             ])) ]).withTags(tags) ]] ),
        apply := (nt, C, cnt) -> C[1]
    )
));


NewRulesFor(PrunedIMDPRDFT, rec(
    PrunedIMDPRDFT_tSPL_Pease_SIMT := rec(
        applicable := nt->nt.hasTags() and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.params[1]) > 1,
        children  := nt -> let(a_lengths := nt.params[1],
                               a_exp := nt.params[3],
                               tags := nt.getTags(),
                               iprdft := IPRDFT1(Last(a_lengths), a_exp),
                               rdim := Rows(iprdft),
                               cdim := Cols(iprdft),
                               [ [ TCompose([ TGrp(TCompose([
                                             TTensorI(IPRDFT1(Last(a_lengths), a_exp), Product(DropLast(a_lengths, 1)), APar, APar),
                                             TL(cdim * Product(DropLast(a_lengths, 1)) / 2, Product(DropLast(a_lengths, 1)), 1, 2), 
                                       ])) ] ::
                                       Reversed(List(DropLast(a_lengths, 1), i->TRC(TTensorI(DFT(i, a_exp), cdim * Product(DropLast(a_lengths, 1))/(2*i), APar, AVec))))
                                    ).withTags(tags) ]] ),
        apply := (nt, C, cnt) -> C[1]
    )
));

