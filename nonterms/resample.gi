
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Class(TResample, Tagged_tSPL_Container, rec(
	     abbrevs :=  [ (odims, idims) -> [ odims, idims, [] ],
			   (odims, idims, shifts) -> [ odims, idims, shifts ],],
	     dims := self >> [Product(self.params[1]), Product(self.params[2])],
	     isReal := True,
	     terminate := self >> Error("Not yet implemented."),
	     transpose := self >> ObjId(self)(self.params[2], self.params[1], -1.0 * self.params[3]).withTags(self.getTags())
	     ));
