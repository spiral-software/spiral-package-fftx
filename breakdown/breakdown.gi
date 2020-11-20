NewRulesFor(Circulant, rec(
    Circulant_PRDFT_FDataNT := rec(
    	forTransposition := false,
    	switch           := true,
    
    	applicable       := nt -> ObjId(nt.params[2]) = FDataNT and ObjId(nt.params[2].nt) = IPRDFT and 
                                  nt.params[2].nt.params = [nt.params[1], 1] and nt.params[1] = -nt.params[3],
    
    	children      := nt -> let(n := nt.params[1],
    	    [[ spiral.transforms.realdft.IPRDFT(n, 1), spiral.transforms.realdft.PRDFT(n, -1) ]]),
    
        apply := (nt, C, cnt) -> C[1] * RCDiag(FDataOfs(nt.params[2].var, Cols(nt.params[2].nt), 0)) * C[2]
    )
));

#_isFractionP := (n, p) -> IntDouble(p*n)/p = n;
#
#NewRulesFor(TResample, rec(
#    TResample_TGath := rec(
#    	forTransposition := false,
#    	switch           := true,
#    
#    	applicable       := nt -> ForAny(Zip2(nt.params[1], nt.params[2]), l-> l[1]<l[2]),
#    
#    	children      := nt -> [[TResample(nt.params[1], nt.params[1], nt.params[3]), TGath(_toBox(nt.params[2], nt.params[1]))]],
#    
#        apply := (nt, C, cnt) -> C[1] * C[2]
#    ),
#    
#    TResample_TScath := rec(
#    	forTransposition := false,
#    	switch           := true,
#    
#    	applicable       := nt -> ForAny(Zip2(nt.params[1], nt.params[2]), l-> l[1]>l[2]),
#    
#    	children      := nt -> [[TScat(_toBox(nt.params[1], nt.params[2])), TResample(nt.params[2], nt.params[2], nt.params[3])]],
#    
#        apply := (nt, C, cnt) -> C[1] * C[2]
#    ),
#    
#    TResample_RC := rec(
#    	forTransposition := false,
#    	switch           := true,
#    
#    	applicable       := nt -> Length(nt.params[1]) > 1 and Length(nt.params[2]) > 1 and Length(nt.params[3]) > 1 
#                                  and nt.params[1][1] = nt.params[2][1],
#    
#    	children      := nt -> [ List(Zip2(Zip2(nt.params[1], nt.params[2]), nt.params[3]), e-> ApplyFunc(TResample, List(Flat(e), i->[i]))) ],
#    
#        apply := (nt, C, cnt) -> let(_C:= Reversed(C), n := Length(nt.params), Compose(
#            List([1..n], i->let(lens := List(_C, j->j.dims()[1]), Tensor(I(Product(lens{[1..i-1]})), _C[i], I(Product(lens{[i+1..n]})))))))
#    ),         
#        
#    TResample_I := rec(
#    	forTransposition := false,
#    	switch           := true,
#    
#    	applicable       := nt -> Length(nt.params[1]) = 1 and Length(nt.params[2]) = 1 and Length(nt.params[3]) = 1 
#                                  and nt.params[1][1] = nt.params[2][1]  and nt.params[3][1] = 0,
#    
#    	children      := nt -> [[]],
#    
#        apply := (nt, C, cnt) -> I(nt.dims()[1])
#    ),
#
#    TResample_PRDFT_frac := rec(
#    	forTransposition := false,
#        denominator      := 2, 
#    	switch           := true,
#    
#    	applicable       := (self,nt) >> Length(nt.params[1]) = 1 and Length(nt.params[2]) = 1 and Length(nt.params[3]) = 1 
#                                  and nt.params[1][1] = nt.params[2][1] and nt.params[3][1] <> 0 and _isFractionP(nt.params[3][1], self.denominator),
#    
#    	children      := nt -> [[ IPRDFT1(nt.params[1][1], -1), PRDFT1(nt.params[1][1], 1) ]],
#    
#        apply := (self, nt, C, cnt) >> let(n := nt.params[1][1], nf := n/2+1, shft := IntDouble(nt.params[3][1] * self.denominator),
#            df := diagMul(fConst(nf, 1/n), fCompose(dOmega(self.denominator*n, shft), dLin(nf, 1, 0, TInt))),
#            dm :=RCDiag(RCData(df)),
#            C[1] *dm * C[2])
#    )
#));
#
