
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Class(MultiPtrCodegenMixin, rec(
    GathPtr := meth(self, o, y, x, opts)
        local i, func, rfunc, ix;
        i := Ind(); func := o.func.lambda();
        
        o.ptr.setType();
        if IsBound(o.func.rlambda) then
            rfunc := o.func.rlambda();
            ix := var.fresh_t("ix", TInt);
            return decl(ix, chain(
                    assign(ix, func.at(0)),
                    assign(nth(y, 0), nth(o.ptr, ix)),
                    loop(i, o.func.domain()-1,
                        chain(assign(ix, rfunc.at(ix)),
                            assign(nth(y, i+1), nth(o.ptr, ix))))));
        else
            return loop(i, o.func.domain(), assign(nth(y,i), nth(o.ptr, func.at(i))));
        fi;
    end,

    ScatPtr := meth(self, o, y, x, opts)
        local i, func, rfunc, ix;
        i := Ind(); func := o.func.lambda();
        o.ptr.setType();

        if IsBound(o.func.rlambda) then
            rfunc := o.func.rlambda();
            ix := var.fresh_t("ix", TInt);
            return decl(ix, chain(
                    assign(ix, func.at(0)),
                    assign(nth(o.ptr, ix), nth(x, 0)),
                    loop(i, o.func.domain()-1,
                        chain(assign(ix, rfunc.at(ix)),
                            assign(nth(o.ptr, ix), nth(x, i+1))))));
        else
            return loop(i, o.func.domain(), assign(nth(o.ptr, func.at(i)), nth(x, i)));
        fi;
    end
));

