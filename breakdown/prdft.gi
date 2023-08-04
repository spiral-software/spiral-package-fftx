ImportAll(realdft);

NewRulesFor(PRDFT1, rec(
    #F PRDFT1_CT: projection of DFT_CT 
    PRDFT1_CT_ := rec(
	forcePrimeFactor := false,
	applicable := (self, nt) >> not IsPrime(nt.params[1]) and 
	    When(self.forcePrimeFactor, not PRDFT1_PF.isApplicable(self.params), true),

	children  := nt -> PRF12_CT_Children(nt.params[1], nt.params[2], PRDFT1, DFT1, PRDFT3, PRDFT1), 
	apply := (nt, C, cnt) -> let(N:=nt.params[1], k:=nt.params[2], m:=Cols(C[1]),
	    PRF12_CT_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,0,j+1)))))))

));

 
RulesFor(IPRDFT1, rec(
    IPRDFT1_CT_ := rec(
	applicable := (self, nt) >> not IsPrime(self.params[1]),
	children  := nt -> PRF12_CT_Children(nt.params[1], nt.params[2], IPRDFT1, DFT1, IPRDFT2, IPRDFT1), 
    apply := (nt, C, cnt) -> let(N:=nt.params[1], k:=nt.params[2], m:=Rows(C[1]),
	    IPRF12_CT_Rule(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,0,j+1)))))))

));
