NewRulesFor(MDPRDFT, rec(
    #F Rule MDPRDFT_Base:  MDPRDFT -> PRDFT1
    #F
    #F Apply this rule when Length(a_lengths()) = 1.
    #F QUESTION: What do the tags mean?
    #F Note MDPRDFT.hasTags() = false = IMDPRDFT.hasTags().
    #F
    MDPRDFT_Base := rec(
        info := "MDPRDFT -> PRDFT1",
        applicable     := nt -> Length(nt.a_lengths())=1,

        # Generally, children is list of ordered lists of all possible children.
        children       := nt -> let(tags := nt.getTags(),
                                    a_lengths := nt.a_lengths(),
                                    a_exp := nt.a_exp(),
                                    [[ PRDFT1(a_lengths[1], a_exp).withTags(tags) ]]),

        # nonterminal, children, children non-terminals
        apply          := (nt, C, Nonterms) -> C[1]
    ),

    #F  MDPRDFT_RowCol1: MDPRDFT(n_1,n_2,...,n_t) = (RC(MDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x PRDFT1(n_t))
    #F
    #F Apply this rule when Length(a_lengths()) > 1 and no tags.
    #F QUESTION: What do the tags mean?
    #F I find MDPRDFT.hasTags() = IMDPRDFT.hasTags() = false.
    #F
    MDPRDFT_RowCol1 := rec(
        info :="MDDFT(n_1,n_2,...,n_t) = (RC(MDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x PRDFT1(n_t))",
        applicable := nt -> Length(nt.a_lengths())>1 and not nt.hasTags(),
        # Generally, children is list of ordered lists of all possible children.
        # 2 children, both with same parameter a_exp():
        # [1] MDDFT on all but last dimension;
        # [2] PRDFT1 on last dimension, the fastest varying dimension.
        children  := nt -> let(a_lengths := nt.a_lengths(),
                               a_exp := nt.a_exp(),
                               [[ MDDFT(DropLast(a_lengths, 1), a_exp),
                                  PRDFT1(Last(a_lengths), a_exp) ]]),

        # nonterminal, children, children non-terminals
        apply := (nt, C, cnt) -> let(a_lengths := nt.a_lengths(),
                                     Clast := C[2],
                                     Crest := C[1],
                                     Nlastcomplex := Clast.dims()[1]/2,
                                     Nrest := DropLast(a_lengths, 1),
                                     Nallbutlast := Product(Nrest),
                                     # Apply Clast on last dimension,
                                     # then Crest on all dimensions but last.
                                     RC(Tensor(Crest, I(Nlastcomplex))) *
                                     Tensor(I(Nallbutlast), Clast)
                                    )
    )
));


NewRulesFor(IMDPRDFT, rec(
    #F Rule IMDPRDFT_Base:  IMDPRDFT -> IPRDFT1
    #F
    #F Apply this rule when Length(a_lengths()) = 1.
    #F QUESTION: What do the tags mean?
    #F Note MDPRDFT.hasTags() = false = IMDPRDFT.hasTags().
    #F
    IMDPRDFT_Base := rec(
        info := "IMDPRDFT -> IPRDFT1",
        applicable     := nt -> Length(nt.a_lengths())=1,
        # Generally, children is list of ordered lists of all possible children.
        children       := nt -> let(tags := nt.getTags(),
                                    a_lengths := nt.a_lengths(),
                                    a_exp := nt.a_exp(),
                                    [[ IPRDFT1(a_lengths[1], a_exp).withTags(tags) ]]),
        # nonterminal, children, children non-terminals
        apply          := (nt, C, Nonterms) -> C[1]
    ),

    #F  IMDPRDFT_RowCol1: IMDPRDFT(n_1,n_2,...,n_t) = (RC(IMDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x IPRDFT1(-n_t))
    #F
    #F Apply this rule when Length(a_lengths()) > 1 and no tags.
    #F QUESTION: What do the tags mean?
    #F I find MDPRDFT.hasTags() = IMDPRDFT.hasTags() = false.
    #F
    #F FIXME: This is not done yet.
    #F
    IMDPRDFT_RowCol1 := rec(
        info :="IMDDFT(n_1,n_2,...,n_t) = (RC(IMDDFT(n_1,n_2,...,n_{t-1}) x I(.))) (I(n_1*n_2*...*n_{t-1}) x PRDFT1(-n_t))",
        applicable := nt -> Length(nt.a_lengths())>1 and not nt.hasTags(),
        # Generally, children is list of ordered lists of all possible children.
        # 2 children, both with same parameter a_exp():
        # [1] IMDDFT on all but last dimension;
        # [2] IPRDFT1 on last dimension, the fastest varying dimension.
        children  := nt -> let(a_lengths := nt.a_lengths(),
                               a_exp := nt.a_exp(),
                               [[ MDDFT(DropLast(a_lengths, 1), a_exp),
                                  IPRDFT1(Last(a_lengths), a_exp) ]]),

        # nonterminal, children, children non-terminals
        apply := (nt, C, cnt) -> let(a_lengths := nt.a_lengths(),
                                     Clast := C[2],
                                     Crest := C[1],
                                     Nlastcomplex := Clast.dims()[2]/2,
                                     Nrest := DropLast(a_lengths, 1),
                                     Nallbutlast := Product(Nrest),
                                     # Apply Crest on all dimensions but last,
                                     # then Clast on last dimension.
                                     Tensor(I(Nallbutlast), Clast) *
                                     RC(Tensor(Crest, I(Nlastcomplex)))
                                    )
    )
));
