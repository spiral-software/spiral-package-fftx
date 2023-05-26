# NewRulesFor(DFT, rec(
#     DFT_tSPL_CT := rec(
#     info          := "tSPL DFT(mn,k) -> DFT(m, k%m), DFT(n, k%n)",
# 
#     maxSize       := false,
#     filter := e->true,
# 
#     applicable    := (self, nt) >> nt.params[1] > 2
#         and (self.maxSize = false or nt.params[1] <= self.maxSize)
#         and not IsPrime(nt.params[1])
#         and nt.hasTags(),
# 
#     children      := (self, nt) >> Map2(Filtered(DivisorPairs(nt.params[1]), self.filter), (m,n) -> [
#         TCompose([
#             TGrp(TCompose([
#                 TTensorI(DFT(m, nt.params[2] mod m), n, AVec, AVec),
#                 TTwiddle(m*n, n, nt.params[2])
#             ])),
#             TGrp(TTensorI(DFT(n, nt.params[2] mod n), m, APar, AVec))
#         ]).withTags(nt.getTags())
#     ]),
# 
#     apply := (nt, c, cnt) -> c[1],
#     switch := false
#     )
# ));

# 
# NewRulesFor(TTensorI, rec(
#     TTensorI_vecrec := rec(
#         forTransposition := false,
#         minSize := false,
#         numTags := false,
#         supportedNTs := [],
#         applicable := (self, nt) >>
#             ObjId(nt.params[1]) in self.supportedNTs and 
#             When(IsBound(self.minSize), ForAll(nt.params[1].dims(), i-> i >= self.minSize), true) and
#             nt.hasTags() and
#             When(IsBound(self.numTags), Length(nt.getTags()) = self.numTags, true),
# 
#         children := nt -> [[nt.params[1].withTags(Drop(nt.params[1].getTags(), 1))]],
# 
#         apply := (nt, c, cnt) -> Error(),
# 
#         switch := false
# 
#     )
# ));
