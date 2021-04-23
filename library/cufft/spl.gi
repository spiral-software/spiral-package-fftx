Declare(CUFFTCall);

Class(CUFFTCall, BaseMat, SumsBase, rec(
    dims := self >> self.L.dims(),
    isReal := self >> self.L.isReal(),
    #-----------------------------------------------------------------------
    rChildren := self >> [self.L, self.genInitCode, self.genCallCode, self.genGlobals],
    rSetChild := rSetChildFields("L", "genInitCode", "genCallCode", "genGlobals"),
    #-----------------------------------------------------------------------
    new := (self, L, codegen) >> SPL(WithBases(self,
        rec(L   := L,
            codegen   := codegen,
            dimensions     := L.dims())
    )),

    #-----------------------------------------------------------------------
    transpose := self >> CUFFTCall(self.L.transpose(), self.genInitCode, self.genCallCode, self.genGlobals),
    #-----------------------------------------------------------------------

    print := (self,i,is) >> Print(self.name, "(",
        self.L.print(i+is,is), ", <codegen>)"),
    #-----------------------------------------------------------------------
    toAMat := self >> self.L.toAMat(),
    #-----------------------------------------------------------------------
    isPermutation := False
));

