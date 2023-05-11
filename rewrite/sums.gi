Add(spiral.sigma.RightPull, Pointwise);


_buildLambda := function(e, at1, at2)
    local pw, iv, iiv, xv, av, ii, dotsl, dotsr;
#    Error();
    dotsl := ....left;
    dotsr := ....right;
    pw := at2;
    iv := at2.element.vars[1];
    iiv := Ind(at1.func.domain());
    xv := at2.element.expr.vars[1];
    av := at2.free()[1];
    pw.element.vars := [iiv];
    pw := SubstTopDown(pw, @@(1, var, (e,cx)->e=iv and IsBound(cx.nth) and cx.nth <> [] and cx.nth[1].loc = xv), e->iiv);
    ii := RulesStrengthReduce(RulesFuncSimp(at1.func).at(iiv));
#    pw := SubstTopDown(pw, @@(1, var, (e,cx)->e=iv and IsBound(cx.nth) and cx.nth <> [] and cx.nth[1].loc = av), e->ii);
    pw := RulesStrengthReduce(SubstTopDown(pw, @@(1, var, (e,cx)->e=iv), e->ii));
#    Error();
    ....left := dotsl;
    ....right := dotsr;
    return pw;
end;   


Class(RulesDiagStandalonePointwise, RuleSet);

RewriteRules(RulesDiagStandalonePointwise, rec(
 # Gath * Pointwise
 CommuteGathDiag := ARule( Compose,
       [ @(11, Gath), @(12, Pointwise) ], # o 1-> 2->
       e->[_buildLambda(e, @(11).val, @(12).val), @(11).val]
    ),

 # Pointwise * Scat
# CommuteDiagScat := ARule( Compose,
#       [ @(1, Pointwise), @(2, Scat) ], # <-1 <-2 o
#  e -> [ Error(), @2.val, Pointwise(fCompose(@1.val.element, @2.val.func)).attrs(@(1).val) ]),
));


#ir := @(1).val.func.domain();
#iiv := Ind(ir);
#
#iv := @(2).val.element.vars[1];
#
#xv := @(2).val.element.expr.vars[1];
#
#av := @(2).val.free()[1];
#
#Collect(@(2).val, @(3, nth, e->e.loc = av));
#
#Collect(@(2).val, @(3, nth, e->e.loc = xv));
#
#a2v := SubstBottomUp(@(2).val, @@(1)
#
#Collect(@(2).val, @@(1, var, (e,cx)->e=iv and IsBound(cx.nth) and cx.nth <> [] and cx.nth[1].loc = av ));
#
#Collect(@(2).val, @@(1, var, (e,cx)->e=iv and IsBound(cx.nth) and cx.nth <> [] and cx.nth[1].loc = xv));
#
#
#lnew := Copy(@(2).val);
#lnew.element.vars := [iiv];
#lnew := SubstTopDown(lnew, @@(1, var, (e,cx)->e=iv and IsBound(cx.nth) and cx.nth <> [] and cx.nth[1].loc = xv), e->iiv);
#ii := @(1).val.func.at(iiv);
#lnew := SubstTopDown(lnew, @@(1, var, (e,cx)->e=iv and IsBound(cx.nth) and cx.nth <> [] and cx.nth[1].loc = av), e->ii);
