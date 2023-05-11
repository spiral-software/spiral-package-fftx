NewRulesFor(DFT, rec(
    DFT_CT_multstage := rec(
   
        maxSize       := false,
        maxRadix := 8,
        forcePrimeFactor := false,

        applicable := (self, nt) >> nt.params[1] > 2
            and not nt.hasTags()
            and (self.maxSize=false or nt.params[1] <= self.maxSize)
            and nt.params[1] > self.maxRadix
            and not IsPrime(nt.params[1])
            and When(self.forcePrimeFactor, not DFT_GoodThomas.applicable(nt), true),

        children  := (self, nt) >> Map2([Last(Filtered(DivisorPairs(nt.params[1]), e-> e[1] <= self.maxRadix))],
            (m,n) -> [ DFT(m, nt.params[2] mod m), DFT(n, nt.params[2] mod n) ]
        ),

        apply := (nt, C, cnt) -> let(mn := nt.params[1], m := Rows(C[1]), n := Rows(C[2]),
            Tensor(C[1], I(n)) *
            Diag(fPrecompute(Tw1(mn, n, nt.params[2]))) *
            Tensor(I(m), C[2]) *
            L(mn, m)
        )
    )
));

