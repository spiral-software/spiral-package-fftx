Import(dct_dst);

Class(MDDST1, TaggedNonTerminal, rec(
    abbrevs := [
    P     -> Checked(IsList(P), ForAll(P,IsPosInt), Product(P) > 1,
             [ RemoveOnes(P)])
    ],
    dims := self >> let(n := Product(self.params[1]), [n, n]),

    terminate := self >> Tensor(List(self.params[1], i -> DST1(i).terminate())),

    transpose := self >> Copy(self),

    isReal := True,

    normalizedArithCost :=  (self) >> let(n := Product(self.params[1]),
                                        IntDouble(2 * n * d_log(n) / d_log(2)) )
));


NewRulesFor(MDDST1, rec(
    MDDST1_DST1 := rec(
        applicable := nt -> Length(nt.params[1]) = 1 and not nt.hasTags(),
        children  := nt -> [[ DST1(nt.params[1][1]) ]],
        apply := (nt, C, cnt) -> C[1]
    ),

    MDDST1_RowCol := rec (
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children := nt -> let(
            dims := nt.params[1],
            len := Length(dims),
            List([1..len-1], i -> [ MDDST1(dims{[1..i]}), MDDST1(dims{[i+1..len]}) ])
        ),

        apply := (nt, C, Nonterms) -> let(
            a := Last(Nonterms[1].params[1]),
            n1 := Rows(Nonterms[1])/a,
            n2 := Rows(Nonterms[2]),
            Tensor(C[1], I(n2)) *
            Tensor(I(n1), Tensor(I(a), C[2]))
        )
    ),
    
    MDDST1_tSPL_RowCol := rec(
        applicable := (self, t) >> Length(t.params[1]) > 1 and t.hasTags(),

        children := nt -> let(
            dims := nt.params[1],
            len := Length(dims),
            List([1..len-1], i -> [ TTensor(MDDST1(dims{[1..i]}), MDDST1(dims{[i+1..len]})).withTags(nt.getTags()) ])
        ),

        apply := (t, C, Nonterms) -> C[1]
    )
));

