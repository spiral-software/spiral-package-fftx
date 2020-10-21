Declare(MDRConv);
# From IOPrunedMDRConv, signature (n,h,oblk,opat,iblk,ipat,isFreqData)
# but this one has oblk=1, iblk=1, opat=ipat=[[0..n(1)-1], ...]
# Left with n (was [1]), h, (was [2]), isFreqData (was [7]).
Class(MDRConv, TaggedNonTerminal, rec(
    a_n := self >> self.params[1],
    a_h := self >> self.params[2],
    a_isFreqData := self >> self.params[3],

    abbrevs := [
                (n, h) -> Checked(ForAll(n, IsPosIntSym),
                                  [_unwrap(n), h, false]),
                (n, h, isFreqData) -> Checked(ForAll(n, IsPosIntSym),
                                              [_unwrap(n), h, isFreqData])
               ],
    dims   := self >> let(nprod := Product(self.a_n()),
                          [nprod, nprod]),
    isReal := self >> true,
    normalizedArithCost := self >> let(n := self.a_n(),
                                       IntDouble(5 * n * d_log(n) / d_log(2))),
    TType := TReal,
    terminate := self >> When(self.a_isFreqData(),
               # case of isFreqData = true
               let(nlist := self.a_n(),
                   n1 := nlist[1],
                   nfreq := RClength(n1)/2, # WAS n1/2 + 1
                   nrest := Drop(nlist, 1),
                   idft := Tensor(IPRDFT(n1, -1), I(Product(nrest))) *
                                  Tensor(I(nfreq), L(2*Product(nrest), 2)) *
                                  Tensor(I(nfreq), RC(MDDFT(nrest, -1))),
                   tlist := MatSPL(idft) * self.a_h().list,
                   MDRConv(nlist,
                           FList(TReal, tlist),
                           false).terminate()
                   ),
               # case of isFreqData = false
               let(nlist := self.a_n(),
                   scat3d := Tensor(List(nlist, n->I(n))::[Mat([[1], [0]])]),
                   gath3d := Tensor(List(nlist, n->I(n))::[Mat([[1, 0]])]),
                   dft3dr := RC(MDDFT(nlist, -1)),
                   idft3dr := RC(MDDFT(nlist, 1)),
                   gfd := List((1/Product(nlist)) *
                               MatSPL(MDDFT(nlist, 1)) *
                               self.a_h().list,
                               ComplexAny),
                   gdiagr := RC(Diag(gfd)),
                   t := gath3d * idft3dr * gdiagr * dft3dr * scat3d,
                   t.terminate()
                  )
        ), # end of When

    hashAs := self >> ApplyFunc(ObjId(self),
                                [self.a_n(),
                                 fUnk(self.a_h().range(),
                                 self.a_h().domain())]::Drop(self.params, 2))
));
