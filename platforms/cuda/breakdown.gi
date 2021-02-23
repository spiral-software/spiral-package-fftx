
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Import(simt);

_isSIMTTag := tag -> IsBound(tag.isSIMTTag) and tag.isSIMTTag;
_toSIMTDim := (tags, n) -> Cond(  
    ObjId(tags[1]) = ASIMTLoopDim, tags[1],
    IsBound(tags[1].params[1]) and not IsInt(tags[1].params[1]), ApplyFunc(ObjId(tags[1]), [tags[1].params[1](n)]), 
    tags[1](n));

NewRulesFor(TIterHStack, rec(
    TIterHStack_SIMT := rec(
        applicable := nt->nt.hasTags() and _isSIMTTag(nt.firstTag()), 
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(Drop(nt.getTags(), 1)), InfoNt(nt.params[2]) ]],
        apply := (nt, c, cnt) -> SIMTIterHStack1(_toSIMTDim(nt.getTags(), cnt[2].params[1].range), cnt[2].params[1], cnt[2].params[1].range, c[1])
    )
));

NewRulesFor(TIterVStack, rec(
    TIterVStack_SIMT := rec(
        applicable := nt->nt.hasTags() and _isSIMTTag(nt.firstTag()), 
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(Drop(nt.getTags(), 1)), InfoNt(nt.params[2]) ]],
        apply := (nt, c, cnt) -> SIMTIterVStack(_toSIMTDim(nt.getTags(), cnt[2].params[1].range), cnt[2].params[1], cnt[2].params[1].range, c[1])
    )
));


NewRulesFor(MDDFT, rec(
    MDDFT_RowCol_3D_SIMT := rec(
        info :="MDDFT(n_1,n_2,...,n_t) = (RC(MDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x PRDFT1(n_t))",
        applicable := nt->nt.hasTags() and Length(nt.getTags()) = 2 and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.params[1]) = 3,
        children  := nt -> let(a_lengths := nt.params[1],
                               a_exp := nt.params[2],
                               tags := nt.getTags(),
                               taglen := Length(tags),
                               rtags1 := tags{[2]},
                               [[ MDDFT(DropLast(a_lengths, 1), a_exp).withTags(rtags1),
                                  DFT(Last(a_lengths), a_exp) ]]),

        apply := (nt, C, cnt) -> let(
            a := nt.params[1],
            n1 := a[1],
            n2 := a[2],
            n3 := a[3],
            SIMTTensor(_toSIMTDim(nt.getTags(){[1]}, n3), C[1], I(n3)) *
                                     SIMTTensor(_toSIMTDim(nt.getTags(){[1]}, n1), I(n1), 
                                         SIMTTensor(_toSIMTDim(nt.getTags(){[2]}, n2), I(n2),
                                             C[2])))
    ),
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



NewRulesFor(MDPRDFT, rec(
    MDPRDFT_3D_SIMT := rec(
        info :="MDDFT(n_1,n_2,...,n_t) = (RC(MDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x PRDFT1(n_t))",
        applicable := nt->nt.hasTags() and Length(nt.getTags()) = 2 and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.a_lengths()) = 3,
        # Generally, children is list of ordered lists of all possible children.
        # 2 children, both with same parameter a_exp():
        # [1] MDDFT on all but last dimension;
        # [2] PRDFT1 on last dimension, the fastest varying dimension.
        children  := nt -> let(a_lengths := nt.a_lengths(),
                               a_exp := nt.a_exp(),
                               tags := nt.getTags(),
                               taglen := Length(tags),
                               rtags1 := tags{[2]},
                               [[ MDDFT(DropLast(a_lengths, 1), a_exp).withTags(rtags1),
                                  PRDFT1(Last(a_lengths), a_exp) ]]),

        # nonterminal, children, children non-terminals
        # NOT YET CORRECT
        apply := (nt, C, cnt) -> let(a_lengths := nt.a_lengths(),
                                     Clast := C[2],
                                     Crest := C[1],
                                     Nlastcomplex := Clast.dims()[1]/2,
                                     Nrest := DropLast(a_lengths, 1),
                                     Nallbutlast := Product(Nrest),
                                     # Apply Clast on last dimension,
                                     # then Crest on all dimensions but last.
                                     RC(SIMTTensor(_toSIMTDim(nt.getTags(){[1]}, Nlastcomplex), Crest, I(Nlastcomplex))) *
                                     SIMTTensor(_toSIMTDim(nt.getTags(){[1]}, Nrest[1]), I(Nrest[1]), 
                                         SIMTTensor(_toSIMTDim(nt.getTags(){[2]}, Nrest[2]), I(Nrest[2]),
                                             Clast)))
    ),
    MDPRDFT_tSPL_RowCol_SIMT := rec(
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
    )
));


NewRulesFor(IMDPRDFT, rec(
    IMDPRDFT_3D_SIMT := rec(
        info :="IMDDFT(n_1,n_2,...,n_t) = (RC(IMDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x PRDFT1(-n_t))",
        applicable := nt->nt.hasTags() and Length(nt.getTags()) = 2 and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.a_lengths()) = 3,
        # Generally, children is list of ordered lists of all possible children.
        # 2 children, both with same parameter a_exp():
        # [1] IMDDFT on all but last dimension;
        # [2] IPRDFT1 on last dimension, the fastest varying dimension.
        children  := nt -> let(a_lengths := nt.a_lengths(),
                               a_exp := nt.a_exp(),
                               tags := nt.getTags(),
                               taglen := Length(tags),
                               rtags1 := tags{[2]},
                               [[ MDDFT(DropLast(a_lengths, 1), a_exp).withTags(rtags1),
                                  IPRDFT1(Last(a_lengths), a_exp) ]]),

        # nonterminal, children, children non-terminals
        # NOT YET CORRECT
        apply := (nt, C, cnt) -> let(a_lengths := nt.a_lengths(),
                                     Clast := C[2],
                                     Crest := C[1],
                                     Nlastcomplex := Clast.dims()[2]/2,
                                     Nrest := DropLast(a_lengths, 1),
                                     Nallbutlast := Product(Nrest),
                                     # Apply Crest on all dimensions but last,
                                     # then Clast on last dimension.
                                     SIMTTensor(_toSIMTDim(nt.getTags(){[1]}, Nrest[1]), I(Nrest[1]), 
                                         SIMTTensor(_toSIMTDim(nt.getTags(){[2]}, Nrest[2]), I(Nrest[2]),                                      
                                             Clast)) *
                                     RC(SIMTTensor(_toSIMTDim(nt.getTags(){[1]}, Nlastcomplex), Crest, I(Nlastcomplex))))
    ),
    IMDPRDFT_tSPL_RowCol_SIMT := rec(
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
    )
));


NewRulesFor(TRC, rec(
    TRC_SIMT := rec(
        forTransposition := false,
        applicable := nt->nt.hasTags() and _isSIMTTag(nt.firstTag()), 
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> RC(c[1])
    )
));


NewRulesFor(TL, rec(
#   TL(N,n,l,r,[]) -> I_l x L(N,n) x I_r
    L_SIMT := rec(
        forTransposition := false,
        applicable := nt->nt.hasTags() and _isSIMTTag(nt.firstTag()), 
        apply := (nt, c, cnt) -> let(
            c1 := When(nt.params[3]=1, [], [I(nt.params[3])]),
            c2 := When(nt.params[4]=1, [], [I(nt.params[4])]),
            Tensor(Concat(c1, [ L(nt.params[1], nt.params[2]) ], c2))
        )
    )
));


NewRulesFor(TTensorInd, rec(
#   base cases
#   I x A
    TTensorInd_SIMT := rec(
        info := "IxA base",
        forTransposition := false,
        applicable := nt -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsParPar(nt.params),
        children := nt -> [[ nt.params[1].withTags(Drop(nt.getTags(), 1)), InfoNt(nt.params[2]) ]],
        apply := (nt, c, cnt) -> SIMTIDirSum(_toSIMTDim(nt.getTags(), cnt[2].params[1].range), cnt[2].params[1], cnt[2].params[1].range, c[1])
    )
));


NewRulesFor(TTensorI, rec(
#   base cases
#   I x A
    IxA_SIMT := rec(
        info := "IxA base",
        forTransposition := false,
        applicable := nt -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and  IsParPar(nt.params) and nt.params[2] > 1,
        children := nt -> [[ nt.params[1].withTags(Drop(nt.getTags(), 1)) ]],
        apply := (nt, c, cnt) -> #When(nt.params[2] > 1,
            SIMTTensor(_toSIMTDim(nt.getTags(), nt.params[2]), I(nt.params[2]), c[1])#,
            #c[1])
    ),
#   A x I
    AxI_SIMT := rec(
        info := "AxI base",
        forTransposition := false,
        applicable := nt -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsVecVec(nt.params) and nt.params[2] > 1,
        children := nt -> [[ nt.params[1].withTags(Drop(nt.getTags(), 1)) ]],
        apply := (nt, c, cnt) -> #When(nt.params[2] > 1,
            SIMTTensor(_toSIMTDim(nt.getTags(), nt.params[2]), c[1], I(nt.params[2]))#,
            #c[1])
    ),
#   A x I
    trivialLoop := rec(
        info := "AxI base",
        forTransposition := false,
        applicable := nt -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and nt.params[2] = 1,
        children := nt -> [[ nt.params[1].withTags(DropLast(nt.getTags(), 1)) ]],
        apply := (nt, c, cnt) -> c[1]
    )
  
));


