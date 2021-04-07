
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

sizes := [
     [ 100, 224, 224],
     [32, 32, 48],
     [32, 48, 32],
     [ 48, 32, 32],
     [32, 32, 32],
     [64, 64, 64],
     [48, 48, 48],
     [96, 96, 96],
     [128,128,128],
     [16, 16, 16],
     [ 320, 96, 96],
     [ 100, 100, 100],
     [ 100, 224, 224 ],
     [ 80, 80, 80 ],
];

i := 1;
szcube := sizes[i];
var.flush();
d := Length(szcube);

name := "mddft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

t := TFCall(TRC(MDDFT(szcube, 1)), 
    rec(fname := name, params := []));

opts := conf.getOpts(t);

Import(fftx.platforms.cuda);
Import(simt);

opts.breakdownRules.MDDFT := [fftx.platforms.cuda.MDDFT_tSPL_Pease_SIMT];
opts.breakdownRules.TTwiddle := [ TTwiddle_Tw1 ];
#opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimZ, ASIMTBlockDimY, ASIMTBlockDimX];
#opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimX];
opts.tags := [ASIMTKernelFlag(ASIMTGridDimX), ASIMTBlockDimY, ASIMTBlockDimX];


_thold := 16;
opts.globalUnrolling := 2*_thold + 1;

opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)), fftx.platforms.cuda.L_IxA_SIMT]::opts.breakdownRules.TTensorI;
#opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, filter := e-> ForAll(e, i -> i in [8..20])))]::opts.breakdownRules.DFT;
opts.breakdownRules.DFT := [CopyFields(DFT_tSPL_CT, rec(switch := true, 
   filter := e-> When(e[1]*e[2] <= _thold^2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold)))]::opts.breakdownRules.DFT;

opts.unparser.simt_synccluster := opts.unparser.simt_syncblock;

opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
       FixUpCUDASigmaSPL_3Stage(s1, opts)); 


tt := opts.tagIt(t);

_tt := opts.preProcess(tt);
rt := opts.search(_tt);

##xx := FindUnexpandableNonterminal(_tt, opts);
#
#spl := SPLRuleTree(rt);
ss := opts.sumsRuleTree(rt);

## drop grp
#ss := SubstTopDown(ss, @(1, Grp), e->e.child(1));
#
## parallelize and flatten loop
#ss:= let(simtidx := ASIMTBlockDimX, 
#    SubstBottomUp(ss, [@(1, SIMTISum), @(2, Compose, e->ForAll(e.children(), c->ObjId(c) = ISum))], 
#        e->let(sx1c := @(2).val.children(),
#                doms := List(sx1c, c->c.domain),
#                mdom := Maximum(doms),
#                ranges := List(List(sx1c, c->[0, c.domain-1])),
#                newc := List([1..Length(sx1c)], 
#                    i-> SIMTISum(@(1).val.simt_dim, @(1).val.var, @(1).val.domain, SIMTISum(simtidx(mdom, ranges[i]), sx1c[i].var, sx1c[i].domain, sx1c[i].child(1)))),
#            ApplyFunc(Compose, newc)
#        ))
#);
#
##ll := Collect(ss, [@(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX), [@(2, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX), BB]]);
##
##s1 := ll[1];
##i1 := s1.var;
##i2 := s1.child(1).var;
##
##rng := i1.range * i2.range;
##ii := Ind(rng);
##sr := rec(
##    (i1.id) := idiv(ii, i2.range),
##    (i2.id) := imod(ii, i2.range)
##);
##
##sdim := ASIMTBlockDimX(rng);
##
##ss2 := SIMTISum(sdim, ii, ii.range, SubstVars(s1.child(1).child(1), sr));
#
## normalize loop
#ss := SubstTopDown(ss, 
#    [@(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX), [@(2, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX), BB]],
#    e->let(s1 := @(1).val,
#        i1 := s1.var,
#        i2 := s1.child(1).var,
#        rng := i1.range * i2.range,
#        ii := Ind(rng),
#        sr := rec(
#            (i1.id) := idiv(ii, i2.range),
#            (i2.id) := imod(ii, i2.range)
#        ),
#        sdim := ASIMTBlockDimX(rng),
#        SIMTISum(sdim, ii, ii.range, SubstVars(s1.child(1).child(1), sr))
#    )
#);
#
## fix loop iterations
#if ObjId(ss) = Compose then         
#    kernels := ss.children();
#
#    for i in [1..Length(kernels)] do
#        _s := kernels[i];
#        if Length(Collect(_s,  @(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX))) > 0 then
#            newv := Maximum(List(List(Collect(_s, @(1, [SIMTSUM, SIMTISum], e->ObjId(e.simt_dim) = ASIMTBlockDimX)), e-> e.simt_dim), i->i.params[1]));
#            _s := SubstTopDown(_s, @(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX),
#                e->SIMTISum(ApplyFunc(ASIMTBlockDimX, [newv]::Drop(@(1).val.simt_dim.params, 1)), @(1).val.var, @(1).val.domain, @(1).val.child(1)) 
#            );
#        
#            _s := SubstTopDown(_s, @(1, SIMTSUM, e->ObjId(e.simt_dim) = ASIMTBlockDimX),
#                e->SIMTSUM(ApplyFunc(ASIMTBlockDimX, [newv]::Drop(@(1).val.simt_dim.params, 1)), @(1).val.children()) 
#            );
#        fi;    
#        kernels[i] := _s;
#    od;
#    ss := Compose(kernels);
#fi;
# 

c:= opts.codeSums(ss);
c.ruletree := rt;
opts.prettyPrint(c);
#
#
#
#tm := MatSPL(_tt);
#sm := MatSPL(spl);
#InfinityNormMat(tm-sm);
#

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));

PrintTo(name::".rt.g", c.ruletree);

ss := opts.sumsRuleTree(c.ruletree);
PrintTo(name::".ss.g", ss);

PrintTo(name::".spl.g", spl);


##-------------------
#224 x 224 x 100
#
#10 x 10 x 224 x 224
#
#kernel(10) * warp(10) x groupY(8) x  block(16 x 14 x 28) 
#10*10*8* 16*2*2 = 50 kB shared memory
#
#
#14 x 16 x 100 x 224
#
#kernel(14) x warp(16) x groupX(5) x block(20 x 14 x 16)
#14*5*16 * 2*16*2 = 70 kB
#
#kernel(16) x warp(14) x groupX(5) x block(20 x 14 x 16)
#
#
#DFT(100 x I(50176)
#
#n := 100;
#m := 50176;
#mem := 1024*96;
#mem_per_pt := 2*8*2*2;
#max_threads := 2048;
#
#_peelof := (n,m) -> Maximum(Filtered(mem_per_pt * Filtered(n*DivisorsInt(m), e-> e<max_threads), f -> f < mem))/(mem_per_pt*n);
#
#_peelof(100, 224*224);
#
#_peelof(224, 100*224);
#
#
#
#
#DFT(224) x I(22400)
