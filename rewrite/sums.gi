Add(spiral.sigma.RightPull, Pointwise);

RewriteRules(RulesDiagStandalone, rec(
 # Gath * Pointwise
 CommuteGathDiag := ARule( Compose,
       [ @(1, Gath), @(2, Pointwise) ], # o 1-> 2->
  e -> [ Pointwise(fCompose(@2.val.element, @1.val.func)).attrs(@(2).val), @1.val ]),

 # Pointwise * Scat
 CommuteDiagScat := ARule( Compose,
       [ @(1, Pointwise), @(2, Scat) ], # <-1 <-2 o
  e -> [ @2.val, Pointwise(fCompose(@1.val.element, @2.val.func)).attrs(@(1).val) ]),
));
