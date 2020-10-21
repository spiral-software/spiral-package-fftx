RewriteRules(RulesStrengthReduce, rec(
    mul_rv_cxpack:= Rule([@(2, mul), @(1, Value, e->(e.t = TComplex and let(vv:= im(e.v).eval(), IsValue(vv) and vv.v = 0.0)) or e.t in [TReal, TInt] ), @(3, cxpack)],
        e->let(vv := Value(TReal, re(@(1).val).eval()), cxpack(vv * @(3).val.args[1], vv * @(3).val.args[2]))),
));

RewriteRules(RulesFuncSimp, rec(
    distribure_diagDirsum_fTensor_fId := Rule([@(1, fCompose), @(2, diagDirsum, e->ForAll(e.children(), c->c.domain() = e.children()[1].domain())), [@(3, fTensor), @(4, fId, e->e.range() = Length(@(2).val.children())),...]],
        e->diagDirsum(List([0..@(4).val.range()-1], j->fCompose(@(2).val.children()[j+1], fTensor(Drop(@(3).val.children(), 1)))))),
        
    distribure_diagDirsum_fTensor_fBase := Rule([@(1, fCompose), @(2, diagDirsum, e->ForAll(e.children(), c->c.domain() = e.children()[1].domain())), [@(3, fTensor), @(4, fBase, e->e.range() = Length(@(2).val.children())),...]],
        e->fCompose(@(2).val.child(@(4).val.params[2]+1), fTensor(Drop(@(3).val.children(), 1)))),
        
    flip_L_fbase_fId := ARule(fCompose, [ @(1,L), @(3, fTensor, e->let(ch := e.children(), ObjId(Last(ch)) = fId 
            and ForAll(DropLast(ch, 1), j->ObjId(j) = fBase) and @(1).val.params[1] / @(1).val.params[2] = Last(ch).size))],
        e-> [ fTensor([Last(@(3).val.children())]::DropLast(@(3).val.children(), 1))]),
        
    push_RCData :=ARule(fCompose, [@(1,RCData), @(3, fTensor, e->ObjId(Last(e.children())) = fId and Last(e.children()).size = 2)],
        e->[RCData(fCompose(@(1).val.func, fTensor(DropLast(@(3).val.children(), 1))))]),

    diagMul_fConst_RCData := Rule([@(1, diagMul), @(2, fConst), @(3, RCData)],
        e->RCData(diagMul(fConst(@(2).val.params[1], @(2).val.params[2]/2, @(2).val.params[3]), @(3).val.func))),
        
    diagMul_diagTensor := Rule([@(1, diagMul), @(2, fConst), @(3, diagTensor)],
        e->let(ch := @(3).val.children(),
               loc := PositionProperty(ch, c->ObjId(c) = Lambda and ObjId(c.expr) = cxpack),
               chn := ch[loc],
               pre := [1..loc-1],
               post := [loc+1..Length(ch)], 
               ldc := Filtered(ch, c->ObjId(c) = Lambda and IsValue(c.expr)),
               fct := Product(List(ldc), c->_unwrap(c.expr)),
               chl := List(ch{pre}, c->When(IsValue(c.expr), Lambda(c.vars, V(1)), c)),
               chr := List(ch{post}, c->When(IsValue(c.expr), Lambda(c.vars, V(1)), c)),
               diagTensor(chl::[Lambda(chn.vars, mul(V(fct * @(2).val.params[3]), chn.expr))]::chr))),
    
    fCompose_Lambda_fBase := Rule([@(1,fCompose), @(3,Lambda), @(2, fBase)], e->let(ii := Ind(1), Lambda(ii, @(3).val.at(@(2).val.params[2]))))
        
));
