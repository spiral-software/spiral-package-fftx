
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
