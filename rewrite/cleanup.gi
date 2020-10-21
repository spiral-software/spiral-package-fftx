Class(RulesDropSymDecl, RuleSet);
RewriteRules(RulesDropSymDecl, rec(
    drop_decl := Rule(@@(1, decl, (e,cx)->ForAny(cx.opts.symbol, i->i in e.vars)), 
        (e,cx)-> decl(Filtered(@@(1).val.vars, i->not i in cx.opts.symbol), @@(1).val.cmd))
));

