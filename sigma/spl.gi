
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Declare(ScatPtr);

Class(GathPtr, Gath, rec(
    rChildren := self >> [self.ptr, self.func],
    rSetChild := rSetChildFields("ptr", "func"),
    new := (self, ptr, func) >> SPL(WithBases(self, rec(
        ptr := ptr,
      	func := Checked(IsFunction(func) or IsFuncExp(func), func)))).setDims()
));

Class(ScatPtr, Scat, rec(
    rChildren := self >> [self.ptr, self.func],
    rSetChild := rSetChildFields("ptr", "func"),
    new := (self, ptr, func) >> SPL(WithBases(self, rec(
        ptr := ptr,
	    func := Checked(IsFunction(func) or IsFuncExp(func), func)))).setDims()
));

Class(OO, O);


Class(Pointwise, RCDiag, rec(isBlock := true));
# set isBlock so the BlockSums() pass in SumsRuleTree() allows the Pointwise in a larger 
# unrolled block before the Pointwise dims are fixed up

# FIXME
_transposeContext := c -> c;
Declare(FContainer);

Class(FContainer, Grp, rec(
    new := (self, spl, context) >> SPL(WithBases(self, rec(
        _children := [spl], context := context))).setDims(), 
    dims := self >> self.child(1).dims(),
    toAMat := (self) >> AMatMat(MatSPL(self.child(1))),
    isReal := self >> self.child(1).isReal(),
    #-----------------------------------------------------------------------
#    conjTranspose := self >> ObjId(self)(self.child(1).conjTranspose(), self.context.conjTtranspose),
    from_rChildren := (self, rch) >> FContainer(rch[1], self.context),
    rChildren := self >> self._children::[self.context],

    print := (self, i, is) >> Print(self.name, "(\n", Blanks(i+is), 
    	self.child(1).print(i+is,is), ",\n", Blanks(i+is), self.context, ")", 
    	self.printA()),
    unroll := self >> self,
    transpose := self >> FContainer(self.child(1).transpose(), _transposeContext(self.context)),
));
