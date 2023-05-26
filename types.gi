Class(TArrayNDC, TArrayBase, rec(
    isArrayT := true,
    toPtrType := self >> TPtr(self.t),
    doHashValues := true,

     __call__ := (self, t, sizes) >>
        WithBases(self, rec(
        t    := Checked(IsType(t), t),
        sizes := sizes,
        operations := TypOps)),
    print := self >> Print(self.__name__, "(", self.t, ", ", self.sizes, ")"),
    
    rChildren := self >> [self.t, self.sizes],
    rSetChild := rSetChildFields("t", "sizes"),
    free := self >> []
));


Class(TArrayNDF, TArrayBase, rec(
    isArrayT := true,
    toPtrType := self >> TPtr(self.t),
    doHashValues := true,

     __call__ := (self, t, sizes) >>
        WithBases(self, rec(
        t    := Checked(IsType(t), t),
        sizes := sizes,
        operations := TypOps)),
    print := self >> Print(self.__name__, "(", self.t, ", ", self.sizes, ")"),
    
    rChildren := self >> [self.t, self.sizes],
    rSetChild := rSetChildFields("t", "sizes"),
    free := self >> [],
    ctype := self >> TPtr(self.t.realType())
));

