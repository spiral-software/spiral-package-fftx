#F FDataNT(<datavar>, <nt>)
#F
Class(FDataNT, Function, rec(
    __call__ := (self, datavar, nt) >> WithBases(self, rec(
        var := datavar,
        operations := PrintOps,
        nt := nt,
        domain := self >> Rows(self.nt)
    )),

    rChildren := self >> [ self.var, self.nt, self._domain, self._range],
    rSetChild := rSetChildFields("var", "nt", "_domain", "_range"),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], rch[2]).setDomain(rch[4]).setRange(rch[5]),

    domain := self >> self.len,
    print := self >> Print(self.name,"(",self.var,", ",self.nt,")"),

# these are not yet right
    at := (self, n) >> When(IsInt(n) and IsValue(self.ofs) and IsBound(self.var.value),
        self.var.value.v[n + self.ofs.v + 1],
        nth(self.var, n + self.ofs)),

    tolist := self >> List([0..EvalScalar(self.len-1)], i -> nth(self.var, self.ofs+i)),
    lambda := self >> let(x := Ind(self.domain()), Lambda(x, nth(self.var, self.ofs+x))),
# up to here

    domain := self >> Rows(self.nt),
    range := self >> When(self._range=false, self.var.t.t, self._range),
    inline := true,
    free := self >> self.ofs.free()
));
