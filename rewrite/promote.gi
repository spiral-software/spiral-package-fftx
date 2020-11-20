# the rule set to promote nonterminals
Class(RulesFFTXPromoteNT, RuleSet);

Class(RulesFFTXPromoteNT_Cleanup, RuleSet);

#RewriteRules(RulesFFTXPromoteNT, rec(
#    IPRDFT_RCDiag_PRDFT__Circulant_Rule := Rule([Compose, @(1,IPRDFT), [@(2,RCDiag), @(4, FDataOfs, e->e.ofs = 0), @(5,I)], @(3,PRDFT)],
#        e->let(n := @(1).val.params[1], fdata := @(2).val.element,
#            Circulant(n, FDataNT(fdata.var, @(1).val), -n))
#    )
#));

_toSymList := l -> [Minimum(l)..Maximum(l)];

RewriteRules(RulesFFTXPromoteNT, rec(
    Scat_Circulant_Gath__IOPrunedRConv := ARule(Compose, [[@(1, Gath), fAdd],  @(2,Circulant), [@(3, Scat), fAdd]], 
        e-> [IOPrunedRConv(@(2).val.params[1], 
            FDataOfs(@(2).val.params[2].var, 2*(@(2).val.params[1]/2+1), 0), 
            1, _toSymList(List(@(1).val.func.tolist(), _unwrap)), 
            1, _toSymList(List(@(3).val.func.tolist(), _unwrap)), true)]),
            
# This is one mega promotion rule for Hockney that needs to be broken apart after MDRConv and/or PrunedMDRDFT is introduced
    Hockney_hack := ARule(Compose, [[@(6, Gath), @(8,fTensor, e->ForAll(e.children(), i->ObjId(i)=fAdd))],
                                     @(1,IMDPRDFT, e -> e.params[2] = 1), [@(2,RCDiag), @(4, FDataOfs, e->e.ofs = 0), @(5,I)], 
                                     @(3,MDPRDFT, e -> e.params[2] = Product(e.params[1])-1),
                                     [@(7, Scat), @(9,fTensor, e->ForAll(e.children(), i->ObjId(i)=fAdd))]],
        e-> let(ii := Ind(Rows(@(2).val)),
            sym := @(2).val.element.var,
            symf := Lambda(ii, nth(sym,ii)),
            opat := List(@(8).val.children(), i-> _toSymList(List(i.tolist(), _unwrap))),
            ipat := List(@(9).val.children(), i-> _toSymList(List(i.tolist(), _unwrap))),
        [ IOPrunedMDRConv(@(1).val.params[1], symf, 1, opat, 1, ipat, true) ])),
        
# DAG to Compose rules
    DAG_collapse1 := Rule([@(1, TDAG), ..., @(2, TDAGNode, e->ForAny(Drop(@(1).val.params[1], 1), k->k.params[3] = e.params[2])), ...], 
        e->  let(nbr := Filtered(@(1).val.params[1], k->k.params[3] = @(2).val.params[2])[1], 
                 ch := Filtered(Drop(@(1).val.params[1],1) , i -> i<> nbr),
                 dn := TDAGNode(nbr.params[1] * @(2).val.params[1], nbr.params[2], @(2).val.params[3]),
                 TDAG([dn]::ch))),
    DAG_remove := Rule([@(1,TDAG), @(2, TDAGNode), @(3), @(4)],
        e-> @(2).val.params[1]),
        
    Drop_TDecl := Rule(@(1, TDecl, e->ForAll(e.params[2], i->not i in e.params[1].free())),
        e -> @(1).val.params[1]), 
));

RewriteRules(RulesFFTXPromoteNT_Cleanup, rec(
# FIXME: the promotion rule does not (yet) have the guard to ensure the value of k is correct
    IPRDFT_RCDiag_PRDFT__Circulant_ARule := ARule(Compose, [@(1,IPRDFT, e -> e.params[2] = 1), [@(2,RCDiag), 
            @(4, FDataOfs, e->e.ofs = 0), @(5,I)], @(3,PRDFT, e -> e.params[2] = e.params[1]-1)],
        e->let(n := @(1).val.params[1], fdata := @(2).val.element,
            [Circulant(n, FDataNT(fdata.var, @(1).val), -n) ])),
            
# FIXME: the promotion rule does not (yet) have the guard to ensure the value of k is correct
    IMDPRDFT_RCDiag_MDPRDFT__RConv_ARule := ARule(Compose, [@(1,IMDPRDFT, e -> e.params[2] = 1), [@(2,RCDiag), @(4, FDataOfs, e->e.ofs = 0), @(5,I)], 
            @(3,MDPRDFT, e -> e.params[2] = Product(e.params[1])-1)],
        e->let(n := @(1).val.params[1], fdata := @(2).val.element,
            [MDRConv(n, fdata.var, true) ])),
            
# MDDFT * Scat -> PrunedMDDFT
    MDDFT_Scat__PrunedMDDFT := ARule(Compose, [@(1,MDDFT), [@(2, Scat), @(3,fTensor, e->ForAll(e.children(), i->ObjId(i)=fAdd))]],
        e -> [ PrunedMDDFT(@(1).val.params[1], @(1).val.params[2], 1,  List(@(3).val.children(), i-> _toSymList(List(i.tolist(), _unwrap))))] ),
        
# Gath * MDDFT -> PrunedIMDDFT
    Gath_MDDFT__PrunedIMDDFT := ARule(Compose, [[@(1, Gath), @(2,fTensor, e->ForAll(e.children(), i->ObjId(i)=fAdd))], @(3, MDDFT)],
        e -> [PrunedIMDDFT(@(3).val.params[1], @(3).val.params[2], 1,  List(@(2).val.children(), i-> _toSymList(List(i.tolist(), _unwrap))))]),

# the n-dimensional non-terminal is still missing here
# PRDFT * Scat -> PrunedPRDFT
    PRDFT_Scat__PrunedPRDFT := ARule(Compose, [@(1,PRDFT), [@(2, Scat), @(3,fAdd)]],
        e -> [ PrunedPRDFT(@(1).val.params[1], @(1).val.params[2], 1,  _toSymList(List(@(3).val.tolist(), _unwrap)))] ),
        
# Gath * PRDFT -> PrunedIPRDFT
    Gath_PRDFT__PrunedIPRDFT := ARule(Compose, [[@(1, Gath), @(2,fAdd)], @(3, IPRDFT)],
        e -> [PrunedIPRDFT(@(3).val.params[1], @(3).val.params[2], 1,  _toSymList(List(@(2).val.tolist(), _unwrap)))])
));

        
# WarpX stuff
Class(RulesFFTXPromoteWarpX1, RuleSet);
RewriteRules(RulesFFTXPromoteWarpX1, rec(
    MultiX := Rule([@(0, TDAGNode), @(1), @(2), @(3).cond(e->IsList(e) and Length(e) = 1 and ObjId(e[1]) = nth and e[1].loc = X), ...],
        e -> TDAGNode(@(1).val * GathPtr(@(3).val[1], fId(Cols(@(1).val))), @(2).val, [ @(3).val[1].loc ]).withTags(@(0).val.getTags())),

    MultiY := Rule([@(0, TDAGNode), @(1), @(2).cond(e->IsList(e) and Length(e) = 1 and ObjId(e[1]) = nth and e[1].loc = Y), @(3), ...],
        e -> TDAGNode(ScatPtr(@(2).val[1], fId(Rows(@(1).val))) * @(1).val, [ @(2).val[1].loc ], @(3).val).withTags(@(0).val.getTags())),
));
        
Class(RulesFFTXPromoteWarpX2, RuleSet);
RewriteRules(RulesFFTXPromoteWarpX2, rec(
    TDAGNode_VStack1 := Rule([@(1, TDAG), ..., @(2, TDAGNode, 
            e-> IsList(e.params[2]) and Length(e.params[2]) = 1 and ObjId(e.params[2][1]) = nth and e.params[2][1].loc <> Y and _unwrap(e.params[2][1].idx) = 0
                and not ObjId(e.params[3][1]) = nth), 
            ...],
        e -> let(
            vr := @(2).val.params[2][1].loc,
            ch := Filtered(@(1).val.params[1], c->IsList(c.params[2]) and Length(c.params[2]) = 1 and ObjId(c.params[2][1]) = nth and c.params[2][1].loc = vr
                     and c.params[3] = @(2).val.params[3]),
            sl := Flat(SortRecordList(ch, c->_unwrap(c.params[2][1].idx))),
            spls := List(sl, i-> i.params[1]),
            vstk := VStack(spls),
            dn := TDAGNode(vstk, [vr], @(2).val.params[3]),
            nch := Filtered(@(1).val.params[1], c->(not IsList(c.params[2])) or (not Length(c.params[2]) = 1) or (not ObjId(c.params[2][1]) = nth) or (not c.params[2][1].loc = vr)
                     or (not c.params[3] = @(2).val.params[3])),
            TDAG([dn]::nch).withTags(@(1).val.tags)
        )),
        
    TDAGNode_VStack2 := Rule([@(1, TDAG), ..., @(2, TDAGNode, 
            e-> IsList(e.params[3]) and Length(e.params[3]) = 1 and ObjId(e.params[3][1]) = nth and e.params[3][1].loc <> X and _unwrap(e.params[3][1].idx) = 0
                and not ObjId(e.params[2][1]) = nth), 
            ...],
        e -> let(
            vr := @(2).val.params[3][1].loc,
            ch := Filtered(@(1).val.params[1], c->IsList(c.params[3]) and Length(c.params[3]) = 1 and ObjId(c.params[3][1]) = nth and c.params[3][1].loc = vr
                     and c.params[2] = @(2).val.params[2]),
            sl := Flat(SortRecordList(ch, c->_unwrap(c.params[3][1].idx))),
            spls := List(sl, i-> i.params[1]),
            hstk := HStack(spls),
            dn := TDAGNode(hstk,  @(2).val.params[2], [vr]),
            nch := Filtered(@(1).val.params[1], c->(not IsList(c.params[3])) or (not Length(c.params[3]) = 1) or (not ObjId(c.params[3][1]) = nth) or (not c.params[3][1].loc = vr)
                     or (not c.params[2] = @(2).val.params[2])),
            TDAG(nch::[dn]).withTags(@(1).val.tags)
        )),
        
   TResample_TGath := Rule(@(1, TResample, e->TResample_TGath.applicable(e)),
       e->let(l := @(1).val,
              c := TResample_TGath.children(l)[1],
              TResample_TGath.apply(l, c, c)
           )),
        
   TResample_TScat := Rule(@(1, TResample, e->TResample_TScat.applicable(e)),
       e->let(l := @(1).val,
              c := TResample_TScat.children(l)[1],
              TResample_TScat.apply(l, c, c)
           )),

   TResample_MD_nofrac := Rule(@(1, TResample, e->TResample_MD_nofrac.applicable(e)),
       e->let(l := @(1).val,
              c := TResample_MD_nofrac.children(l)[1],
              TResample_MD_nofrac.apply(l, c, c)
           )),
          
   TGath_Gath := Rule(@(1, TGath, e-> TGath_base.applicable(e)), e->ApplyRuleSPL(TGath_base, @(1).val)),

   TScat_Scat := Rule(@(1, TScat, e-> TScat_base.applicable(e)), e->ApplyRuleSPL(TScat_base, @(1).val)),
           
   Gath_GathPtr := ARule(Compose, [@(1, Gath, e-> ObjId(e.func) = fTensor and ForAll(e.func.children(), c->ObjId(c) = fId or (ObjId(c) = fAdd and c.params[3] = 0))), @(2, GathPtr)], 
       e->[ GathPtr(@(2).val.ptr, fId(Rows(@(1).val))) ]),
        
   ScatPtr_Scat := ARule(Compose, [@(1, ScatPtr), @(2, Scat, e-> ObjId(e.func) = fTensor and ForAll(e.func.children(), c->ObjId(c) = fId or (ObjId(c) = fAdd and c.params[3] = 0)))], 
       e->[ ScatPtr(@(1).val.ptr, fId(Cols(@(2).val))) ]), 
       
   Expand_TResample := Rule(@(1, TResample, e->TResample_MD_frac.applicable(e)),
       e->let(trs := @(1).val,
               c := TResample_MD_frac.children(trs)[1],
               TResample_MD_frac.apply(trs, c, c)
           )),
));
           
Class(RulesFFTXPromoteWarpX3, RuleSet);
RewriteRules(RulesFFTXPromoteWarpX3, rec(          
    Merge_TTensorI_VStack := ARule(Compose, [ @(1, TTensorI, e->e.params[3] = APar and e.params[4] = APar), 
        @(2, VStack, e->@(1).val.params[2] = Length(e.children()) and ForAll(e.children(), e->Rows(e) = Cols(@(1).val.params[1]))) ],
        e->[VStack(List(@(2).val.children(), c->@(1).val.params[1] * c))]),     

    Merge_HStack_TTensorI := ARule(Compose, [ @(1, HStack),
        @(2, TTensorI, e->e.params[3] = APar and e.params[4] = APar and
            e.params[2] = Length(@(1).val.children()) and ForAll(@(1).val.children(), j->Cols(j) = Rows(e.params[1]))) ],
        e->[HStack(List(@(1).val.children(), c->c * @(2).val.params[1]))]), 
       
# this rule is WRONG: the sign needs to be inverse of each other!       
    MDPRDFT_IMDPRDFT := ARule(Compose, [@(1, MDPRDFT), @(2, IMDPRDFT, e->Length(@(1).val.params[1]) = Length(e.params[1]) and Mod(@(1).val.params[2] + e.params[2], Product(e.params[1])) = 0 and 
        ForAll([1..Length(e.params[1])], j->e.params[1][j] = @(1).val.params[1][j]))], 
        e->[Diag(fConst(TReal, Rows(@(1).val), Product(@(1).val.params[1])))]),
        
    Diag_RCDiag := ARule(Compose, [@(1, Diag, e->ObjId(e.element) = fConst and e.element.range() = TReal), @(2, RCDiag)],
        e -> [RCDiag(diagMul(@(1).val.element, @(2).val.element))]),

    RCDiag_Diag := ARule(Compose, [@(1, RCDiag), @(2, Diag, e->ObjId(e.element) = fConst and e.element.range() = TReal)],
        e -> [RCDiag(diagMul(@(2).val.element, @(1).val.element))]),

        
 # these rules need a bit more guards to not misfire      
    Peel_HStack_RCDiag := Rule(@(1, HStack, e->ForAll(e.children(), c->let(cc := c.children(), Length(cc) = 3 and 
            ObjId(cc[1]) = ScatPtr and ObjId(cc[2]) = IMDPRDFT and ObjId(cc[3]) = RCDiag  and ObjId(cc[3].element) = RCData))),
        e -> let(cc := @(1).val.children(), i := Ind(Length(cc)), c1 := cc[1].children()[1], c2 := cc[1].children()[2],
#                Error(), 
                fts := Flat(List(cc, c->Collect(c, fTensor))),
#                rng := List(fts, f->List([1..Length(f.children())], ii->f.child(ii).range())),
                _rng := Flat(List(fts, f->List([1..Length(f.children())], ii->f.child(ii).range()))),
#                rngf := FData(rng),
                _rngf := FData(_rng),
                dom := fts[1].child(1).domain(),
#                fts2 := fTensor(List([0..Length(fts[1].children())-1],  ii->fAdd(nth(rngf.at(i), ii), dom, 0))),
                _fts2 := fTensor(List([0..Length(fts[1].children())-1],  ii->fAdd(_rngf.at(Length(fts[1].children()) * i + ii), dom, 0))),
            TNoDiagPullinRight(TIterHStack(TCompose([ScatPtr(nth(c1.ptr.loc, i), _fts2), c2]), i)) * 
                TNoPullLeft(RCDiag(RCData(diagDirsum(List(cc, c->Last(c.children()).element.func))))))),
            
    Peel_RCDiag_VStack := Rule(@(1, VStack, e->ForAll(e.children(), c->let(cc := Reversed(c.children()), Length(cc) in [2, 3] and 
            ObjId(cc[1]) = GathPtr and ObjId(cc[2]) = MDPRDFT and When(Length(cc) = 3, ObjId(cc[3]) = RCDiag and ObjId(cc[3].element) = RCData, true)))),
        e-> let(cc := @(1).val.children(), i := Ind(Length(cc)), rcc := Reversed(cc[1].children()),
                fts := Flat(List(cc, c->Collect(c, fTensor))),
#                rng := List(fts, f->List([1..Length(f.children())], ii->f.child(ii).range())),
                _rng := Flat(List(fts, f->List([1..Length(f.children())], ii->f.child(ii).range()))),
#                rngf := FData(rng),
                _rngf := FData(_rng),
                dom := fts[1].child(1).domain(),
#                fts2 := fTensor(List([0..Length(fts[1].children())-1],  ii->fAdd(nth(rngf.at(i), ii), dom, 0))),
                _fts2 := fTensor(List([0..Length(fts[1].children())-1],  ii->fAdd(_rngf.at(Length(fts[1].children()) * i + ii), dom, 0))),

            TNoPullRight(RCDiag(RCData(diagDirsum(List(cc, c->When(ObjId(c.children()[1]) = RCDiag, c.children()[1].element.func, 
                fConst(TComplex, Rows(rcc[2])/2, V(1)))))))) *
            TNoDiagPullinLeft(TIterVStack(TCompose([rcc[2], GathPtr(nth(rcc[1].ptr.loc, i), _fts2)]), i))
        )),
        
    Fuse_Diag := ARule(Compose, [ @(1, TNoPullLeft, e->ObjId(e.params[1]) = RCDiag), @(2, TRC), @(3, TNoPullRight, e->ObjId(e.params[1]) = RCDiag)], 
        e->[ TRC(TGrp(TCompose([Diag(@(1).val.params[1].element.func), @(2).val.params[1], Diag(@(3).val.params[1].element.func)])))]),
        
    Promote_Compose := Rule(@(1, Compose, e->ForAll(e.children(), c->ObjId(c) in [TRC, TNoDiagPullinLeft, TNoDiagPullinRight])),
        e->TCompose(@(1).val.children())),
        
    flatten_I := Rule(@(1, Tensor, e->ForAll(e.children(), c->ObjId(c)=I)), e->I(Rows(@(1).val))), 
  
));


RewriteRules(RulesSums, rec(
 PullInRightScatPtr := ARule( Compose,
       [ @(1, [ScatPtr, TNoPullLeft]),
         @(2, [RecursStep, Grp, BB, SUM, Buf, ISum, Data, COND, TNoDiagPullin, TNoDiagPullinLeft, TNoDiagPullinRight, NeedInterleavedComplex, SIMTISum]) ],
    e -> [ CopyFields(@(2).val, rec(
             _children :=  List(@(2).val._children, c -> @(1).val * c),
             dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

 PullInLeftGathPtr := ARule( Compose,
       [ @(1, [RecursStep, Grp, BB, SUM, SUMAcc, Buf, ISum, ISumAcc, Data, COND, TNoDiagPullin, TNoDiagPullinLeft, TNoDiagPullinRight, NeedInterleavedComplex, SIMTISum]),
         @(2, [GathPtr, TNoPullRight]) ],
    e -> [ CopyFields(@(1).val, rec(
                _children := List(@(1).val._children, c -> c * @(2).val),
                dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),
 ComposeGathGathPtr := ARule(Compose, [ @(1, Gath), @(2, GathPtr) ], # o 1-> 2->
     e -> [ GathPtr(@(2).val.ptr, fCompose(@(2).val.func, @(1).val.func)) ]),

 ComposeScatPtrScat := ARule(Compose, [ @(1, ScatPtr), @(2, Scat) ], # <-1 <-2 o
     e -> [ ScatPtr(@(1).val.ptr, fCompose(@(1).val.func, @(2).val.func)) ]),
                
));




# IOPrunedRConv(@(2).val.params[1], @(2).val.params[2], 1, _toSymList(List(@(1).val.func.tolist(), _unwrap)), 1, _toSymList(List(@(3).val.func.tolist(), _unwrap)));
# IOPrunedRConv(@(2).val.params[1], FDataOfs(@(2).val.params[2].var, 2*(@(2).val.params[1]/2+1), 0), 1, _toSymList(List(@(1).val.func.tolist(), _unwrap)), 1, _toSymList(List(@(3).val.func.tolist(), _unwrap)), true);


