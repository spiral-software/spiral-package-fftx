
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
    ),
    
    TTensorInd_SIMT_peelof := rec(
        forTransposition := false,
        
        # these config parameters need to be moved into the opts...
        mem := 1024*96,
        mem_per_pt := 2*8*2,
#        max_threads := 2048,
#        max_threads := 1024,
        max_threads := 512,
        max_kernel := 18 * 18,
        _peelof := (self,n,m) >> Maximum([1]::Filtered(self.mem_per_pt * Filtered(n*DivisorsInt(m), e-> e<self.max_threads), 
            f -> f < When(n >= self.max_kernel, self.mem/2, self.mem)))/(self.mem_per_pt*n),
        
        applicable := (self, nt) >> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsParPar(nt.params) and 
            nt.params[2].range > 1 and self._peelof(Rows(nt.params[1]), nt.params[2].range) > 1 and 
            ((nt.params[2].range / self._peelof(Rows(nt.params[1]), nt.params[2].range)) > 1 or not IsPrime(nt.params[2].range)),
        children := (self, nt) >> let(n := Rows(nt.params[1]), m:= nt.params[2].range, peelof := self._peelof(n,m), remainder := m/peelof, 
            dp := DivisorPairs(nt.params[2].range), kj := dp[When(IsEvenInt(Length(dp)), Length(dp)/2, (Length(dp)+1)/2)],
            k:= Ind(When(remainder = 1, kj[1], peelof)), j := Ind(When(remainder = 1, kj[2], remainder)),
            [[ TTensorInd(TTensorInd(SubstVars(Copy(nt.params[1]), rec((nt.params[2].id):=j * k.range + k)), k, APar, APar), j, APar, APar).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1]
    ),
    
    
#   (I x A) 
    TTensorInd_SIMT_peelof2 := rec(
        forTransposition := false,
        
        # these config parameters need to be moved into the opts...
        mem := 1024*96,
        mem_per_pt := 2*8*2,
#        max_threads := 2048,
#        max_threads := 1024,
        max_threads := 512,
        max_kernel := 18 * 18,
        _peelof := (self,n,m) >> Maximum([1]::Filtered(self.mem_per_pt * Filtered(n*DivisorsInt(m), e-> e<self.max_threads), 
            f -> f < When(n >= self.max_kernel, self.mem/2, self.mem)))/(self.mem_per_pt*n),
        
        applicable := (self, nt) >> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsParPar(nt.params) and 
            nt.params[2] > 1 and self._peelof(Cols(nt.params[1]), nt.params[2].range) > 1 and 
            ((nt.params[2].range / self._peelof(Rows(nt.params[1]), nt.params[2].range)) > 1 or not IsPrime(nt.params[2].range)),
        children := (self, nt) >> let(n := Cols(nt.params[1]), m:= nt.params[2].range, peelof := self._peelof(n,m), remainder := m/peelof,
            dp := DivisorPairs(nt.params[2].range), kj := dp[When(IsEvenInt(Length(dp)), Length(dp)/2, (Length(dp)+1)/2)],
            k:= Ind(When(remainder = 1, kj[1], peelof)), j := Ind(When(remainder = 1, kj[2], remainder)),
            [[  TTensorInd(TTensorInd(SubstVars(Copy(nt.params[1]), rec((nt.params[2].id):=j * k.range + k)), k, APar, APar), j, APar, APar).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1]
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
#        max_threads := 1024,
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
#        max_threads := 1024,
        max_kernel := 18 * 18,
        _peelof := (self,n,m) >> Maximum(Filtered(self.mem_per_pt * Filtered(n*DivisorsInt(m), e-> e<self.max_threads), 
            f -> f < When(n >= self.max_kernel, self.mem/2, self.mem)))/(self.mem_per_pt*n),
        
        applicable := nt -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsParVec(nt.params) and nt.params[2] > 1,
        children := (self, nt) >> let(n := Cols(nt.params[1]), m:= nt.params[2], peelof := self._peelof(n,m), remainder := m/peelof,
            [[ TCompose([
                TTensorI(
                    TCompose([TTensorI(nt.params[1], peelof, APar, APar), TL(Cols(nt.params[1]) * peelof, peelof)]), remainder, APar, APar),
                TL(Cols(nt)/peelof, (Cols(nt)/peelof)/n, 1, peelof), 
                ]).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1]
    ),
#   (I x A) 
    IxA_SIMT_peelof := rec(
        forTransposition := false,
        
        # these config parameters need to be moved into the opts...
        mem := 1024*96,
        mem_per_pt := 2*8*2,
#        max_threads := 2048,
        max_threads := 1024,
        max_kernel := 18 * 18,
        _peelof := (self,n,m) >> Maximum([1]::Filtered(self.mem_per_pt * Filtered(n*DivisorsInt(m), e-> e<self.max_threads), 
            f -> f < When(n >= self.max_kernel, self.mem/2, self.mem)))/(self.mem_per_pt*n),
        
        applicable := (self, nt) >> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsParPar(nt.params) and nt.params[2] > 1 and self._peelof(Rows(nt.params[1]), nt.params[2]) > 1 and 
            (nt.params[2] / self._peelof(Rows(nt.params[1]), nt.params[2])) > 1,
        children := (self, nt) >> let(n := Rows(nt.params[1]), m:= nt.params[2], peelof := self._peelof(n,m), remainder := m/peelof, 
            [[  TTensorI(TTensorI(nt.params[1], peelof, APar, APar), remainder, APar, APar).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1]
    ),
#   (I x A) 
    IxA_SIMT_peelof2 := rec(
        forTransposition := false,
        
        # these config parameters need to be moved into the opts...
        mem := 1024*96,
        mem_per_pt := 2*8*2,
#        max_threads := 2048,
        max_threads := 1024,
        max_kernel := 18 * 18,
        _peelof := (self,n,m) >> Maximum([1]::Filtered(self.mem_per_pt * Filtered(n*DivisorsInt(m), e-> e<self.max_threads), 
            f -> f < When(n >= self.max_kernel, self.mem/2, self.mem)))/(self.mem_per_pt*n),
        
        applicable := (self, nt) >> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsParPar(nt.params) and nt.params[2] > 1 and self._peelof(Cols(nt.params[1]), nt.params[2]) > 1 and 
            (nt.params[2] / self._peelof(Cols(nt.params[1]), nt.params[2])) > 1,
        children := (self, nt) >> let(n := Cols(nt.params[1]), m:= nt.params[2], peelof := self._peelof(n,m), remainder := m/peelof, 
            [[  TTensorI(TTensorI(nt.params[1], peelof, APar, APar), remainder, APar, APar).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1]
    )
    
  
));




