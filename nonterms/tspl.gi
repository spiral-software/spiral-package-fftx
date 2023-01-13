
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# from FFTX C++ unparser
Class(TFCall, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (nt, cconf) -> Checked(IsSPL(nt), [nt, cconf]) ],
    transpose := self >> ObjId(self)(self.params[1].transpose(), self.params[2]).withTags(self.getTags())
));

Class(TDeviceCall, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (nt, cconf) -> Checked(IsSPL(nt), [nt, cconf]) ],
    transpose := self >> ObjId(self)(self.params[1].transpose(), self.params[2]).withTags(self.getTags())
));


Class(TDecl, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (nt, vlist) -> Checked(IsSPL(nt), [nt, vlist]) ],
    transpose := self >> ObjId(self)(self.params[1].transpose(), self.params[2]).withTags(self.getTags())
));

Class(TDAGNode, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (nt, ylist, xlist) -> Checked(IsSPL(nt), [nt, When(IsList(ylist), ylist, [ylist]), When(IsList(xlist), xlist, [xlist])]) ],
    transpose := self >> ObjId(self)(self.params[1].transpose(), self.params[2], self.params[3]).withTags(self.getTags())
));

Class(TDAG, TCompose, rec(
    terminate := self >> Error("Not yet implemented."),
    
    from_rChildren := (self, rch) >> let(
        len := Length(rch),
        transposed := rch[len-1],
        tags := rch[len],
        t := ApplyFunc(ObjId(self), [rch{[1..len-2]}]),
        tt := When(transposed, t.transpose(), t),
        attrTakeA(tt.withTags(tags), self)
    ),

    rChildren := self >>
        Concatenation(self.params[1], [self.transposed, self.tags]),

    rSetChild := meth(self, n, newChild)
        local l;
        l := Length(self.params[1]);
        if n <= l then
            self.params[1][n] := newChild;
        elif n = l+1 then
            self.transposed := newChild;
        elif n = l+2 then
            self.tags := newChild;
        else Error("<n> must be in [1..", l+2, "]");
        fi;
        # self.canonizeParams(); ??
        self.dimensions := self.dims();
    end,
    
));


## the dims here are ordered as in the MDDFT case: e.g., for 3D: odims[1] = x, odims[2] = y, odims[3] = z
#Class(TResample, Tagged_tSPL_Container, rec(
#    abbrevs :=  [ (odims, idims) -> [ odims, idims, [] ],
#                  (odims, idims, shifts) -> [ odims, idims, shifts ],],
#    dims := self >> [Product(self.params[1]), Product(self.params[2])],
#    isReal := True,
#    terminate := self >> Error("Not yet implemented."),
#    transpose := self >> ObjId(self)(self.params[2], self.params[1], self.params[3]).withTags(self.getTags())
#));

Declare(TSparseMat);
Class(TSparseMat, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (dims, entries) -> [ dims, entries ] ],
    dims := self >> self.params[1],
    terminate := self >> Error("Not yet implemented."),
    transpose := self >> TSparseMat(Reversed(self.params[1]), [x->Error("Not yet done")])
));

Declare(TIterVStack);

Class(TIterHStack, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (nt, idx) -> Checked(IsSPL(nt), IsVar(idx),
	    [nt, idx]) ],

    dims := self >> self.params[1].dims(){[1]} :: self.params[1].dims(){[2]} * self.params[2].range,

    terminate := self >> IterHStack(self.params[2], self.params[2].range, self.params[1].terminate()),

    transpose := self >> TIterVStack(self.params[1].transpose(), self.params[2])
          .withTags(self.getTags()),

    isReal := self >> self.params[1].isReal(),

    normalizedArithCost := self >>
        self.params[1].normalizedArithCost() * self.params[2].range,

#    doNotMeasure := true,

    HashId := self >> let(
    	p := self.params,
    	h := When(IsBound(p[1].HashId), p[1].HashId(), p[1]),
            [h, p[2].range] :: When(IsBound(self.tags), self.tags, [])
    )
));

Class(TIterVStack, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (nt, idx) -> Checked(IsSPL(nt), IsVar(idx),
	    [nt, idx]) ],

    dims := self >> (self.params[1].dims(){[1]} * self.params[2].range) :: self.params[1].dims(){[2]},

    terminate := self >> IterVStack(self.params[2], self.params[2].range, self.params[1].terminate()),

    transpose := self >> TIterHStack(self.params[1].transpose(), self.params[2])
          .withTags(self.getTags()),

    isReal := self >> self.params[1].isReal(),

    normalizedArithCost := self >>
        self.params[1].normalizedArithCost() * self.params[2].range,

#    doNotMeasure := true,

    HashId := self >> let(
    	p := self.params,
    	h := When(IsBound(p[1].HashId), p[1].HashId(), p[1]),
            [h, p[2].range] :: When(IsBound(self.tags), self.tags, [])
    )
));

Class(TNoDiagPullin, Tagged_tSPL_Container, rec(
    abbrevs :=  [ s -> Checked(IsSPL(s), [s]) ],
    HashId := self >> let(
    	p := self.params[1],
    	h := When(IsBound(p.HashId), p.HashId(), p),
            [ self.__name__, h ] :: When(IsBound(self.tags), self.tags, [])
        )
));

Class(TNoDiagPullinRight, Tagged_tSPL_Container, rec(
    abbrevs :=  [ s -> Checked(IsSPL(s), [s]) ],
    HashId := self >> let(
    	p := self.params[1],
    	h := When(IsBound(p.HashId), p.HashId(), p),
            [ self.__name__, h ] :: When(IsBound(self.tags), self.tags, [])
        )
));

Class(TNoDiagPullinLeft, Tagged_tSPL_Container, rec(
    abbrevs :=  [ s -> Checked(IsSPL(s), [s]) ],
    HashId := self >> let(
    	p := self.params[1],
    	h := When(IsBound(p.HashId), p.HashId(), p),
            [ self.__name__, h ] :: When(IsBound(self.tags), self.tags, [])
        )
));

Class(TNoPullRight, Tagged_tSPL_Container, rec(
    abbrevs :=  [ s -> Checked(IsSPL(s), [s]) ],
    HashId := self >> let(
    	p := self.params[1],
    	h := When(IsBound(p.HashId), p.HashId(), p),
            [ self.__name__, h ] :: When(IsBound(self.tags), self.tags, [])
        )
));

Class(TNoPullLeft, Tagged_tSPL_Container, rec(
    abbrevs :=  [ s -> Checked(IsSPL(s), [s]) ],
    HashId := self >> let(
    	p := self.params[1],
    	h := When(IsBound(p.HashId), p.HashId(), p),
            [ self.__name__, h ] :: When(IsBound(self.tags), self.tags, [])
        )
));

flipDims := function(orig)
	local t;
	t := Copy(orig);
	t.params[1] := Reversed(t.params[1]);
	return t;
end;

#F   tSPL ColMajor -> RowMajor transformation
Class(TColMajor, Tagged_tSPL_Container, rec(
    _short_print := true,
    abbrevs :=  [ (A) -> Checked(IsNonTerminal(A) or IsSPL(A), [A]) ],
    dims := self >> 2*self.params[1].dims(),

    terminate := self >> Mat(MatSPL(flipDims(self.params[1]))),

    transpose := self >> ObjId(self)(
	self.params[1].conjTranspose()).withTags(self.getTags()),

    conjTranspose := self >> self.transpose(),

    isReal := self >> true,

    # Do not use doNotMeasure, this will prevent TRC_By_Def from ever being found!
    doNotMeasure := false,
    normalizedArithCost := self >> self.params[1].normalizedArithCost(),

    HashId := self >> let(
	h := [ When(IsBound(self.params[1].HashId), self.params[1].HashId(),
		    self.params[1]) ],
        When(IsBound(self.tags), Concatenation(h, self.tags), h))
));
