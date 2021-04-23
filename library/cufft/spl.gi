Declare(CUFFTCall);

Class(CUFFTCall, BaseMat, SumsBase, rec(
    dims := self >> self.L.dims(),
    isReal := self >> self.L.isReal(),
    #-----------------------------------------------------------------------
    rChildren := self >> [self.L, self.codegen],
    rSetChild := rSetChildFields("L", "codegen"),
    #-----------------------------------------------------------------------
    new := (self, L, codegen) >> SPL(WithBases(self,
        rec(L   := L,
            codegen   := CopyFields(codegen, rec(
                plan_var := var.fresh_t("plan", TSym("cufftHandle")),
                size_var := var.fresh_t("size", TInt),
                data := (self, c) >> data(self.size_var, V(self.N), c),
                init := self >> call(rec(id:="cufftPlanMany"), addrof(self.plan_var), V(1), addrof(self.size_var), addrof(self.size_var), V(1), V(self.M),
    	            addrof(self.size_var), V((self.N * self.K) / (self.p1 * self.p2)), V(1), "CUFFT_Z2Z", V((self.N * self.K) / (self.p1 * self.p2))),
            )),
            dimensions     := L.dims())
    )),

    #-----------------------------------------------------------------------
    transpose := self >> CUFFTCall(self.L.transpose(), self.codegen),
    #-----------------------------------------------------------------------

    print := (self,i,is) >> Print(self.name, "(",
        self.L.print(i+is,is), ", <codegen>)"),
    #-----------------------------------------------------------------------
    toAMat := self >> self.L.toAMat(),
    #-----------------------------------------------------------------------
    isPermutation := False
));

RewriteRules(RulesRC, rec(
    RC_CUFFTCall := Rule([RC, @(1, CUFFTCall)], e -> CUFFTCall(RC(@(1).val.L), @(1).val.codegen)),
));


CudaCodegen.CUFFTCall := (self, o, y, x, opts) >> simt_block(
    call(rec(id := "cufftExecZ2Z", codegen := o.codegen), o.codegen.plan_var,
        tcast(TPtr(TSym("cufftDoubleComplex")), x), tcast(TPtr(TSym("cufftDoubleComplex")), y), When(o.codegen.k = 1, "CUFFT_INVERSE", "CUFFT_FORWARD")));

