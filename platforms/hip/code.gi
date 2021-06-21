
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

FixUpHIP_Code := function (c, opts)
    local kernels, kernel_inits, globals, var_decls, var_dels, cx, v, dptr; 

    if IsBound(opts.fixUpTeslaV_Code) and opts.fixUpTeslaV_Code then
        kernels := List(Collect(c, specifiers_func), k->k.id);

        dptr := var.fresh_t("hp", TPtr(TReal));

        kernel_inits := List(kernels, k-> call(rec(id := "hipFuncSetCacheConfig"), fcall("reinterpret_cast<const void*>", k), "hipFuncCachePreferL1"));

        globals := Flat(List(Collect(c, @@(1, decl, (e, cx) -> (not IsBound(cx.specifiers_func) or cx.specifiers_func = []) and
                               (not IsBound(cx.func) or cx.func = []))), x->x.vars));

        var_decls := chain(Flat(List(globals, v -> [ 
                call(rec(id := "hipMalloc"), tcast(TPtr(TPtr(TVoid)), addrof(dptr)), sizeof(v.t.t) * v.t.size), 
                call(rec(id := "hipMemcpyToSymbol"), fcall("HIP_SYMBOL", v), addrof(dptr), sizeof(dptr.t)) 
            ])));
        
        var_dels := decl(dptr, chain(Flat(List(globals, v -> [ 
                call(rec(id := "hipMemcpyFromSymbol"), addrof(dptr), fcall("HIP_SYMBOL", v), sizeof(dptr.t)), 
                call(rec(id := "hipFree"), dptr)
            ]))));

        cx := chain(kernel_inits :: [var_decls]);
        for v in globals do
            v.t := TPtr(v.t.t, ["__device__"]);
        od;

        c := SubstBottomUp(c, @(1, func, f -> f.id = "init"),
            e -> CopyFields(@(1).val, rec(cmd := decl(dptr, chain(cx, @(1).val.cmd))))
        );
        c := SubstBottomUp(c, @(1, func, f -> f.id = "destroy"),
            e -> CopyFields(@(1).val, rec(cmd := chain(var_dels, @(1).val.cmd)))
        );
        c := SubstBottomUp(c, @(1, chain, e -> ForAny(e.cmds, e -> ObjId(e) = func and e.id = "init")),
            e -> chain(Filtered(@(1).val.cmds, e-> ObjId(e) <> func or e.id <> "init") :: Filtered(@(1).val.cmds, e -> ObjId(e) = func and e.id = "init"))
        );

        
    fi;

    return c;
end;
