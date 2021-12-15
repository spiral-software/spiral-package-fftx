
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
    ),
    MDDFT_tSPL_Pease_SIMT := rec(
        applicable := nt->nt.hasTags() and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.params[1]) > 1,
        children  := nt -> let(a_lengths := nt.params[1],
                               a_exp := nt.params[2],
                               tags := nt.getTags(),
                               [ [TCompose(List(nt.params[1], i->TTensorI(DFT(i, a_exp), Product(nt.params[1])/i, AVec, APar))).withTags(tags) ]]),
        apply := (nt, C, cnt) -> C[1]
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
    ),
    MDPRDFT_tSPL_Pease_SIMT := rec(
        applicable := nt->nt.hasTags() and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.params[1]) > 1,
        children  := nt -> let(a_lengths := nt.params[1],
                               a_exp := nt.params[2],
                               tags := nt.getTags(),
                               prdft := PRDFT1(Last(a_lengths), a_exp),
                               rcdim := Rows(prdft),
                               [ [ TCompose(List(DropLast(nt.params[1], 1), i->TRC(TTensorI(DFT(i, a_exp), rcdim * Product(DropLast(nt.params[1], 1))/(2*i), AVec, APar)))::
                                           [ TGrp(TCompose([TL(rcdim * Product(DropLast(nt.params[1], 1)) / 2, rcdim / 2, 1, 2), 
                                             TTensorI(PRDFT1(Last(a_lengths), a_exp), Product(DropLast(nt.params[1], 1)), APar, APar)])) ]).withTags(tags) ]] ),
        apply := (nt, C, cnt) -> C[1]
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
    ),
    IMDPRDFT_tSPL_Pease_SIMT := rec(
        applicable := nt->nt.hasTags() and ForAll(nt.getTags(), _isSIMTTag) and Length(nt.params[1]) > 1,
        children  := nt -> let(a_lengths := nt.params[1],
                               a_exp := nt.params[2],
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
    ),
#   L (I x A)
    L_IxA_SIMT := rec(
        forTransposition := false,
        
        # these config parameters need to be moved into the opts...
        mem := 1024*96,
        mem_per_pt := 2*8*2,
        max_threads := 2048,
        max_kernel := 18 * 18,
        _peelof := (self,n,m) >> Maximum(Filtered(self.mem_per_pt * Filtered(n*DivisorsInt(m), e-> e<self.max_threads), 
            f -> f < When(n >= self.max_kernel, self.mem/2, self.mem)))/(self.mem_per_pt*n),
        
        applicable := nt -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsVecPar(nt.params) and nt.params[2] > 1,
        children := (self, nt) >> let(n := Rows(nt.params[1]), m:= nt.params[2], peelof := self._peelof(n,m), remainder := m/peelof,
            [[ TCompose([TL(Rows(nt)/peelof, n, 1, peelof), 
                TTensorI(
                    TCompose([TL(Rows(nt.params[1]) * peelof, Rows(nt.params[1])), TTensorI(nt.params[1], peelof, APar, APar)]),
                    remainder, APar, APar)]).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1]
    ),
#   (I x A) L
    IxA_L_SIMT := rec(
        forTransposition := false,
        
        # these config parameters need to be moved into the opts...
        mem := 1024*96,
        mem_per_pt := 2*8*2,
        max_threads := 2048,
        max_kernel := 18 * 18,
        _peelof := (self,n,m) >> Maximum(Filtered(self.mem_per_pt * Filtered(n*DivisorsInt(m), e-> e<self.max_threads), 
            f -> f < When(n >= self.max_kernel, self.mem/2, self.mem)))/(self.mem_per_pt*n),
        
        applicable := nt -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsParVec(nt.params) and nt.params[2] > 1,
        children := (self, nt) >> let(n := Rows(nt.params[1]), m:= nt.params[2], peelof := self._peelof(n,m), remainder := m/peelof,
            [[ TCompose([
                TTensorI(
                    TCompose([TTensorI(nt.params[1], peelof, APar, APar), TL(Rows(nt.params[1]) * peelof, peelof)]), remainder, APar, APar),
                TL(Rows(nt)/peelof, (Rows(nt)/peelof)/n, 1, peelof), 
                ]).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1]
    )
  
));




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
       applicable :=  (self, nt) >> false, #not nt.hasTags() and Length(nt.params[1]) = 3 and nt.params[7], Breaks!
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nfreq := nlist[1]/2+1,
                            i := Ind(nfreq*nlist[2]),
                            hfunc := Cond(ObjId(diag) = Lambda,
                                let(j := Ind(nlist[3]),
                                    Lambda(j, cxpack(diag.at(2*(j + i*nlist[3])), diag.at(2*(j + i*nlist[3])+1)))
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
                            [[ PrunedPRDFT(nlist[2], -1, iblk, ipats[2]),  # stage 1: PRDFT y
                                PrunedDFT(nlist[1], -1, iblk, ipats[1]),    # stage 2: DFT z
                                IOPrunedConv(nlist[3], hfunc, oblk, opats[3], iblk, ipats[3], true), # stage 3+4+5: complex conv in x
                                PrunedIDFT(nlist[1], 1, oblk, opats[1]), # stage 6: iDFT in z
                                PrunedIPRDFT(nlist[2], 1, oblk, opats[2]),   # stage 7: iPRDFT in y
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
                                    nfreq := nlist[2]/2+1,
                                    n2 := nlist[2],
                                    n3 := nlist[3],
                                    oblk := nt.params[3],
                                    opats := nt.params[4],
                                    iblk := nt.params[5],
                                    ipats := nt.params[6],
                                    ns1 := iblk * Length(ipats[1]),
                                    ns2 := iblk * Length(ipats[2]),
                                    ns3 := iblk * Length(ipats[3]),
                                    nd1 := oblk * Length(opats[1]),
                                    nd2 := oblk * Length(opats[2]),
                                    nd3 := oblk * Length(opats[3]),
                                    stage1 := Tensor(I(nfreq*ns2), L(2*ns3, ns3)) * Tensor(I(ns2), prdft1d, I(ns3)),
                                    stage2 := RC(Tensor(pdft1d, I(nfreq*ns3))),
                                    stage543c := IDirSum(i, iopconv),
                                    stage543 := RC(stage543c),
                                    stage6 := RC(Tensor(ipdft1d, I(nfreq*nd3))),
                                    stage7 := Tensor(I(nd2), iprdft1d, I(nd3)) * Tensor(I(nfreq*nd2), L(2*nd3, 2)),
                                    conv3dr := stage7 * stage6 * stage543 * stage2 * stage1,
                                    conv3dr
                            ),
    ),
));


