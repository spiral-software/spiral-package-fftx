BoxND := (l, stype) -> Cond(IsInt(l), TArray(stype, l), 
                             IsList(l) and Length(l) = 1, TArray(stype, l[1]), 
                             TArray(BoxND(Drop(l, 1), stype), l[1]));


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

Class(TArrayNDF_ConjEven, TArrayNDF);
