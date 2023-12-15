
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Class(OpenCLCodegen, CudaCodegen, rec(

    #changing __device__ and __constant__ to be global and constant for OpenCL
    ker_datas_to_device := meth(self, ker_datas, cuda_icode, opts)
        local d, li;

        for d in Reversed(ker_datas) do
            li := self._get_data_loopindex_deps(cuda_icode, d, opts);
            # If any warp-divergent access then allocate in dev mem otherwise constant.
            if ForAny(li, v -> IsBound(self.simtidx_rec.(v.id)) 
                                and ObjId(self.simtidx_rec.(v.id)) = simtThreadIdxX) then
                d.("decl_specs") := ["global"];
            else
                d.("decl_specs") := ["constant"];
            fi;
            cuda_icode := data(d, d.value, cuda_icode);
        od;

        return cuda_icode;
    end,

    #changing __device__ and __constant__ to be global and constant for OpenCL
    #OpenCL does not have a memcpyToSymbol so mempool objects have to go into kernel arguments,
    #cleanup for static arrays and sizes is FixUpOpenCL_Code function
    make_kernels := meth(self, full_kernel, device_data, full_args, opts)
        local kercx, ker_cnt, ker, ker_args, ker_datas, cuda_ker, cuda_desc, cuda_icode, cuda_sub, 
                dim_grid, dim_block, tbody, tas_grid, tas_block_shmem, cross_ker_tas, ta, check_args, v;
				
		cuda_sub := Cond(IsBound(opts.cudasubName), Concat("ker_",opts.cudasubName), "ker_code");

        cuda_desc := rec(grid_tarrays := [], cuda_kers := [] );

        [full_kernel, ker_datas] := self.extract_ker_datas(full_kernel, opts);
        
        # Mark cross kernel temporaries
        cross_ker_tas := Filtered(Flat(List(Collect(full_kernel, @@(1, decl, (d, ctx) -> not IsBound(ctx.simt_block) or ctx.simt_block = [] )),
                            d -> d.vars)), v -> IsArrayT(v.t));
        DoForAll(cross_ker_tas, v -> v.setAttrTo("crossker", true));
        #Extract all temp arrays that should map to dev mem 
        tas_grid := self.extract_tarrays(full_kernel, 
                                            When(opts.use_shmem,
                                                    v -> (IsBound(v.crossker) and v.crossker) or (IsBound(v.t.size) and (v.t.size >= opts.max_shmem_size/8/2)),
                                                    True
                                            ), opts);

        #Allocate in dev mem 
        [full_kernel, tas_grid] := self.to_grid_arrays(full_kernel, tas_grid, full_args, opts);
        Append(cuda_desc.grid_tarrays, tas_grid);

        kercx := Collect(full_kernel, simt_block);
        kercx := When(kercx = [], [ full_kernel ], kercx);
        ker_cnt := 0;
        for ker in kercx do
            
            tbody := When(ObjId(ker) = simt_block, chain(ker.cmds), ker);

            #SANIL CHANGE
            #<<<< old
            # ker_args := Intersection(device_data, tbody.free())::Intersection(full_args, tbody.free());
            #>>>>> new
            ker_args := Set(tbody.free());
            #----------
            #Extract remaining shared mem temp arrays if required 
            tas_block_shmem := self.extract_tarrays(tbody, v -> opts.use_shmem, opts);
            #Allocate in shared mem 
            [tbody, tas_block_shmem] := self.to_block_shmem_arrays(tbody, tas_block_shmem, ker_args, opts);

            dim_grid := var.fresh_t_obj("g", Dim3(), rec( x := self.get_dim_idx(tbody, simtBlockIdxX), 
                                                          y := self.get_dim_idx(tbody, simtBlockIdxY), 
                                                          z := self.get_dim_idx(tbody, simtBlockIdxZ)
                                                        ) );
            dim_block := var.fresh_t_obj("b", Dim3(), rec( x := self.get_dim_idx(tbody, simtThreadIdxX), 
                                                           y := self.get_dim_idx(tbody, simtThreadIdxY), 
                                                           z := self.get_dim_idx(tbody, simtThreadIdxZ)
                                                        ) );

            #SANIL CHANGE
            #<<<< old
            # cuda_ker := specifiers_func(["kernel"], TVoid, cuda_sub::StringInt(ker_cnt), Filtered(ker_args, d -> not IsBound(d.("decl_specs")) or not "__constant__" in d.("decl_specs")), tbody );
            #>>>>> new
            check_args := Filtered(ker_args, d -> not IsBound(d.value));
            for v in check_args do
                if ObjId(v.t) = TPtr and (IsBound(v.t.qualifiers) and v.t.qualifiers = []) then
                v.t := TPtr(v.t.t, ["global"]);
                fi;
            od;
            cuda_ker := specifiers_func(["kernel"], TVoid, cuda_sub::StringInt(ker_cnt), check_args, tbody );
            #----------
            Add(cuda_desc.cuda_kers,  rec(dim_grid := dim_grid, dim_block := dim_block, cuda_ker := cuda_ker));

            ker_cnt := ker_cnt + 1;
        od;

        cuda_icode := chain(List(cuda_desc.cuda_kers, ck -> ck.cuda_ker));
        cuda_icode := self.ker_datas_to_device(ker_datas, cuda_icode, opts);

        return [cuda_desc, cuda_icode];

    end,

    #changing __device__ and __constant__ to be global and constant for OpenCL
    to_grid_arrays := meth(self, ker, ta_list, ker_args, opts)
        local ta, idx_list, blk_ids, th_ids, idx, idx_cnt, idx_rng, derefs, nths, v;
        
        for ta in ta_list do
            ta.("decl_specs") := ["global"];
            idx_list := When(IsBound(ta.parctx), ta.parctx, []);
            #parctx format: [ [loopvar1, simtIdx1], ... ]
            blk_ids := Filtered(idx_list, idx -> IsSimtBlockIdx(idx[2])); Sort(blk_ids, (i1, i2) -> i1[2].name < i2[2].name);
            th_ids  := Filtered(idx_list, idx -> IsSimtThreadIdx(idx[2])); Sort(th_ids, (i1, i2) -> i1[2].name < i2[2].name);
            for idx in th_ids::blk_ids do
                idx_cnt := idx[2].count(); # Num of threads associated with index
                if idx_cnt > 1 then
                    idx_rng := idx[2].get_rng(); # Thread range: [<1st tid>, <last tid>]
                    derefs := Collect(ker, @(1, deref, v -> ta in v.free()));
                    nths   := Collect(ker, @(1,   nth, v -> ta in v.free()));

                    for v in derefs do
                        v.loc := add(v.loc, ta.t.size*(idx[2]-idx_rng[1]) );
                    od;
                    for v in nths do
                        v.idx := add(v.idx, ta.t.size*(idx[2]-idx_rng[1]));
                    od;
                    ta.t.size := ta.t.size*idx_cnt;
                fi;
            od;
        od;

        if opts.mempool then
            [ker, ta_list] := self.compact_arrays(ker, ta_list, ker_args, ["global"], opts);
        fi;
        return [ker, ta_list];
    end,

    #changing __device__ and __constant__ to be global and constant for OpenCL
    #local shared memory cannot be allocated in kernel it has to come from host side invocation,
    #have to move local memory arrays to kernel args as a pointer
    #size info cleanup in FixUpOpenCL_Code function
    to_block_shmem_arrays := meth(self, ker, ta_list, ker_args, opts)
        local ta, idx_list, blk_ids, th_ids, idx, idx_cnt, idx_rng, derefs, nths, size, padded, v;

        for ta in ta_list do
            padded := false;
            ta.("decl_specs") := ["local"];
            idx_list := When(IsBound(ta.parctx), ta.parctx, []);
            th_ids  := Filtered(idx_list, idx -> IsSimtThreadIdx(idx[2])); Sort(th_ids, (i1, i2) -> i1[2].name < i2[2].name);
            for idx in th_ids do
                idx_cnt := idx[2].count();
                if idx_cnt > 1 then
                    size := ta.t.size;
                    if not padded and Mod(size, 2) = 0 then
                        size := size+1;
                    fi;
                    padded := true;
                    idx_rng := idx[2].get_rng();
                    derefs := Collect(ker, @(1, deref, v -> ta in v.free()));
                    nths   := Collect(ker, @(1,   nth, v -> ta in v.free()));

                    for v in derefs do
                        v.loc := add(v.loc, size*(idx[2]-idx_rng[1]) );
                    od;
                    for v in nths do
                        v.idx := add(v.idx, size*(idx[2]-idx_rng[1]));
                    od;
                    ta.t.size := size*idx_cnt;
                fi;
            od;
            #SANIL CHANGE
            #>>>>>>>new
            Add(ker_args, ta);
            #---------
        od;

        #SANIL CHANGE
        #<<<<<<old
        # ker := decl(ta_list, ker);
        #>>>>>>new
        ker := decl([], ker);
        #---------

        return [ker, ta_list];
    end,
));

#using the Hipify base needed to perform same fix for shared memory variables as well as global memory variables
FixUpOpenCL_Code := function (c, opts)
    local kernels, kernel_inits, globals, locals, var_decls, var_dels, cx, v, dptr, size, params; 

    if IsBound(opts.fixUpTeslaV_Code) and opts.fixUpTeslaV_Code then

        #SANIL CHANGE
        opts.Xptr.t := TPtr(opts.Xptr.t.t, ["global"]);
        opts.Yptr.t := TPtr(opts.Yptr.t.t, ["global"]);

        kernels := List(Collect(c, specifiers_func), k->k.id);

        dptr := var.fresh_t("hp", TPtr(TReal));

        kernel_inits := List(kernels, k-> call(rec(id := "hipFuncSetCacheConfig"), fcall("reinterpret_cast<const void*>", k), "hipFuncCachePreferL1"));

        globals := Flat(List(Collect(c, @@(1, decl, (e, cx) -> (not IsBound(cx.specifiers_func) or cx.specifiers_func = []) and
                               (not IsBound(cx.func) or cx.func = []))), x->x.vars));

        #SANIL CHANGE
        locals := Set(Collect(c, @(1,var, e-> (IsBound(e.decl_specs) and e.decl_specs = ["local"]))));
       
        var_decls := chain(Flat(List(globals::locals, v -> [ 
                call(rec(id := "hipMalloc"), tcast(TPtr(TPtr(TVoid)), addrof(dptr)), sizeof(v.t.t) * v.t.size), 
                call(rec(id := "hipMemcpyToSymbol"), fcall("HIP_SYMBOL", v), addrof(dptr), sizeof(dptr.t)) 
            ])));
        
        var_dels := decl(dptr, chain(Flat(List(globals::locals, v -> [ 
                call(rec(id := "hipMemcpyFromSymbol"), addrof(dptr), fcall("HIP_SYMBOL", v), sizeof(dptr.t)), 
                call(rec(id := "hipFree"), dptr)
            ]))));

        cx := chain(kernel_inits :: [var_decls]);
        for v in globals do
            size := v.t.size;
            v.t := TPtr(v.t.t, ["global"]);
            v.t.size := size;
        od;

        #SANIL CHANGE
        for v in locals do
            size := v.t.size;
            v.t := TPtr(v.t.t, ["local"]);
            v.t.size := size;
        od;

        #SANIL CHANGE
        params := Collect(c, @(1, func, e-> e.id = "transform"))[1].params;
        params := Filtered(params, d -> not d in [X,Y]);
        for v in params do
            if IsBound(v.t.size) then
            size := v.t.size;
            fi;
            if ObjId(v.t) = TPtr then
            v.t := TPtr(v.t.t, ["global"]);
            else
            Error("none pointer based inputs not supported, please convert TFCall params or opts.symbol to pointers\n");
            fi;
            if IsBound(v.t.size) then
            v.t.size := size;
            fi;
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

        #Sanil CHANGE
        c.cmds[1].vars := [];
    fi;

    return c;
end;
