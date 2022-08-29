n1 := 16;
n2 := 16;
n := n1*n2;

radix := 8;
fftParIter := 16;

blk := rec(x := 16, y := 32, z := 8);

szcube := Replicate(3, n);
libdir := ".";
file_suffix := ".cpp"; 
fwd := true; 
codefor := "HIP"; 

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

##  Script to generate code, will be driven by a size specification and will write the
##  CUDA/HIP code to a file.  The code will be compiled into a library for applications
##  to link against -- providing pre-compiled FFTs of standard sizes.

Load(fftx);
ImportAll(fftx);

ImportAll(paradigms.vector);
ImportAll(simt);
ImportAll(fftx.platforms.cuda);

if codefor = "CUDA" then
    conf := LocalConfig.fftx.confGPU();
elif codefor = "HIP" then
    conf := FFTXGlobals.defaultHIPConf();
elif codefor = "CPU" then
    conf := LocalConfig.fftx.defaultConf();
fi;

if fwd then
    prefix := "fftx_mddft_";
    sign   := -1;
else
    prefix := "fftx_imddft_";
    sign   := 1;
fi;

#if 1 = 1 then
name := prefix::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
name := name::"_"::codefor;

PrintLine("fftx_mddft-frame: name = ", name, ", cube = ", szcube, ", size = ",
          StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1),
                                                                s->" x "::StringInt(s))),
          ";\t\t##PICKME##");

## This line from mddft-frame-cuda.g :
##    t := TFCall(TRC(MDDFT(szcube, 1)), 
##                rec(fname := name, params := []));
var_1:= var("var_1", BoxND([0,0,0], TReal));
var_2:= var("var_2", BoxND(szcube, TReal));
var_3:= var("var_3", BoxND(szcube, TReal));
var_2:= X;
var_3:= Y;
symvar := var("sym", TPtr(TReal));
t := TFCall(TDecl(TDAG([
       TDAGNode(TRC(TTensorI(MDDFT(szcube,sign),1,APar, APar)), var_3,var_2),
              ]),
        [var_1]
        ),
    rec(fname:=name, params:= [symvar])
);


RewriteRules(RulesCxRC_Op, rec(
    vRC_SIMTISum := Rule([@(2, vRC), @(1, SIMTISum)], e -> SIMTISum(@(1).val.simt_dim, @(1).val.var, @(1).val.domain, vRC(@(1).val.child(1)))),
    vRC_SIMTSUM := Rule([@(2, vRC), @(1, SIMTSUM)], e -> SIMTSUM(@(1).val.simt_dim, List(@(1).val.children(), vRC))),
    vRC_Gath := Rule([vRC, @(1, Gath)], e -> VGath(@(1).val.func, 2)),
    vRC_Scat := Rule([vRC, @(1, Scat)], e -> VScat(@(1).val.func, 2)),

    vRC_Diag := Rule([@(1, vRC), @(2, Diag) ], e -> vRC(VDiag(@(2).val.element, 1))),
    vRC_VDiag     := Rule([@(1, vRC), @(2, VDiag)], e -> let(v:=2*@(2).val.v, 
        RCVDiag(VData(RCData(@(2).val.element), v), v)))
));


opts := conf.getOpts(t);
opts.cudasubName := name;


# set up vector stuff
Class(HIP_2x64f, SSE_2x64f);
isa := HIP_2x64f;

#isa.mul_cx := (self, opts) >> (
#    (y,x,c) -> let(u1 := var.fresh_t("U", TVectDouble(2)), u2 := var.fresh_t("U", TVectDouble(2)),
#        u3 := var.fresh_t("U", TVectDouble(2)), u4 := var.fresh_t("U", TVectDouble(2)),
#        decl([u1, u2, u3, u4], RulesStrengthReduce(chain(
#            assign(u1, mul(x, velem(c, 0))),                # vushuffle_2x64f(c, [1,1])
#            assign(u2, vpack(velem(x, 0), -velem(x, 1))),   #chshi_2x64f(x)),
#            assign(u3, mul(u2, velem(c, 1))),               # vushuffle_2x64f(c, [2,2])
#            assign(u4, vpack(velem(u3, 1), velem(u3, 0))),    # vushuffle_2x64f(u3, [2,1])
#            assign(y, add(u1, u4))
#        )))));


isa.mul_cx := (self, opts) >> ( # (a+bi)(c+di)
    (y,x,c) -> let(u1 := var.fresh_t("U", TVectDouble(2)), 
        u3 := var.fresh_t("U", TVectDouble(2)), u4 := var.fresh_t("U", TVectDouble(2)),
        decl([u1, u3, u4], RulesStrengthReduce(chain(
            assign(u1, mul(x, velem(c, 0))),  # ac + bci             
            assign(u3, mul(x, velem(c, 1))),  # ad + bdi            
            assign(y, vpack(
                add(velem(u1, 0), velem(u3, 1)),  # ac + bd
                sub(velem(u1, 1), velem(u3, 0))))# (bc +ad)i
        )))));

isa.mul_cx := (self, opts) >> ( # (a+bi)(c+di)
    (y,x,c) -> let(u1 := var.fresh_t("U", TVectDouble(2)), 
        u3 := var.fresh_t("U", TVectDouble(2)), u4 := var.fresh_t("U", TVectDouble(2)),
        decl([u1, u3, u4], RulesStrengthReduce(chain(
            assign(u1, mul(x, velem(c, 0))),  # (a b) * (c c) = (ac bc)
            assign(u3, mul(vpack(velem(x, 1), velem(x, 0)), vpack(-velem(c, 1), velem(c, 1)))),  # (b a) * (-d d) = (-bd ad)        
            assign(y, add(u1, u3)) # (ac - bd | ad + bc)
        )))));



vopts := SIMDGlobals.getOpts(HIP_2x64f);


Add(opts.tags, AVecRegCx(isa));
opts.tags;
opts.breakdownRules.TRC := [CopyFields(TRC_cplxvect, rec(switch := true))];

# set up the options
opts.codegen.VGath := VectorCodegen.VGath;
opts.codegen.VScat := VectorCodegen.VScat;
opts.codegen.VTensor := VectorCodegen.VTensor;
opts.codegen.RCVDiag := VectorCodegen.RCVDiag;
opts.codegen.VContainer := VectorCodegen.VContainer;

opts.unparser.vpack := (self, o, i, is) >> Print("lib_make_vector2<double2>", "(", self.infix(o.args, ", "), ")");
opts.unparser.velem := (self, o, i, is) >> self.printf("$1.$2", [ o.loc, When(o.idx.v=0,"x", "y") ]);
opts.unparser.re := (self, o, i, is) >> self.printf("$1.x", [ o.args[1] ]);
opts.unparser.im := (self, o, i, is) >> self.printf("$1.y", [ o.args[1] ]);
opts.unparser.neg := (self, o, i, is) >> Print("-", self(o.args[1], i, is));

opts.c99 := rec(I :=  "__I__");
opts.unparser.TVect := (self, t, vars, i, is) >> Print("double2 ", self.infix(vars, ", ", i + is));

opts.unparser.Value :=
(self, o, i, is) >> Cond(
    o.t = TComplex, let(c := Complex(o.v), re := ReComplex(c), im := ImComplex(c),
        Print("{", self._decval(re), ", ", self._decval(im), "}")), 
    o.t = TReal, let(v := Cond(IsCyc(o.v), ReComplex(Complex(o.v)), o.v),
        Cond(v < 0, Print("(", self._decval(v), ")"), Print(self._decval(v))) ), 
    ObjId(o.t) = TVect, Print("lib_make_vector2<double2>", "(", self.infix(o.v, ", "), ")"),    
    IsArray(o.t), Print("{", WithBases(self, rec(infixbreak := 4 )).infix(o.v, ", ", i), "}"), o.v < 0, Print("(", o.v, ")"), 
    o.t = TBool, When(o.v in [ true, 1 ], Print("1"), Print("0")), 
    o.t = TUInt, Print(o.v, "u"), Print(o.v));


opts.TComplexCtype := "double2";


opts.vector := vopts.vector;
opts.vector.SIMD := "SSE2";


# temporary fix
#opts.tags := opts.tags{[1,3]};


opts.globalUnrolling := 16;

##  We need the Spiral functions wrapped in 'extern C' for adding to a library
##  Comment out next line if trying standalone or with profiler
opts.wrapCFuncs := true;
if(IsBound(fftx_includes)) then opts.includes:=fftx_includes; fi;
opts.includes[2] := "\"common.h\"";

tt := opts.tagIt(t);
_tt := opts.preProcess(tt);


#breakdown rules setup for 3 stages

opts.breakdownRules.MDDFT := [ fftx.platforms.cuda.MDDFT_tSPL_KL_SIMT];

# temporary fix
#opts.breakdownRules.DFT[1].filter := e -> e[1] = n1 and e[2] = n2;

opts.breakdownRules.DFT := [CopyFields(DFT_CT, rec(maxSize := 8, forcePrimeFactor := true)), DFT_Base, 
    CopyFields(DFT_tSPL_CT, rec(filter := e->e[1] = radix, switch := true)),
    CopyFields(DFT_GoodThomas, rec(switch := true)), CopyFields(DFT_SIMT_cplxvect, rec(switch := true))
];

#Append(opts.breakdownRules.DFT, [CopyFields(DFT_GoodThomas, rec(switch := true)), CopyFields(DFT_SIMT_cplxvect, rec(switch := true))]);
#opts.breakdownRules.DFT[3].forcePrimeFactor := true;


opts.breakdownRules.TTensorI := [ CopyFields(IxA_L_split, rec(switch := true)),
    CopyFields(L_IxA_SIMT, rec(applicable := (nt) -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and nt.firstTag() <> ASIMTBlockDimX and IsVecPar(nt.params) and nt.params[2] > 1)),
    CopyFields(IxA_L_SIMT, rec(applicable := (nt) -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and nt.firstTag() <> ASIMTBlockDimX and IsParVec(nt.params) and nt.params[2] > 1)),
    CopyFields(AxI_SIMT_peelIter, rec(maxIterations := fftParIter)),
    IxA_TTwiddle_SIMT,
    CopyFields(IxA_DFT_CT_SIMT, rec(minDFTsize := radix, parIterations := fftParIter, maxDFTsize := n/radix)),
    CopyFields(IxA_base, rec(applicable := (nt) -> IsParPar(nt.params) and (not nt.hasTags() or ObjId(nt.firstTag()) = AVecRegCx))),
    CopyFields(AxI_base, rec(applicable := (nt) -> IsVecVec(nt.params) and (not nt.hasTags() or ObjId(nt.firstTag()) = AVecRegCx))),
    CopyFields(IxA_SIMT, 
        rec(applicable := (nt) -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsParPar(nt.params) and nt.params[2] > 1 and ObjId(nt.params[1]) <> TTwiddle and
            (nt.getTag(1) = ASIMTBlockDimY or (ObjId(nt.params[1]) <> DFT or Maximum(nt.params[1].dims()) <= radix)))), 
    CopyFields(AxI_SIMT, 
        rec(applicable := (nt) -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsVecVec(nt.params) and nt.params[2] > 1 and nt.params[2] <= fftParIter and 
        (nt.getTag(1) = ASIMTBlockDimY or (ObjId(nt.params[1]) <> DFT or Maximum(nt.params[1].dims()) < radix))
         )) 
    ];
opts.breakdownRules.TL := [ L_SIMT, CopyFields(L_base, rec(applicable := (nt) -> nt.isTag(1, AVecRegCx) or not nt.hasTags())) ];


#---

rt := opts.search(_tt);

#xx := FindUnexpandableNonterminal(_tt, opts);
#DoForAll(xx, PrintLine);

ss := opts.sumsRuleTree(rt);
ss := RulesCxRC_Op(ss);
ss := RulesCxRC_Term(ss);

# flatten X/X -> X loops
ss := SubstBottomUp(ss, [@(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX), @(2, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX), ...],
    e-> let(
        sx1 := @(1).val,
        nn := sx1.simt_dim.params[1],
        ni := Ind(nn),
        ch := sx1.child(1).child(1),
        i1 := sx1.var,
        i2 := sx1.child(1).var,
        i1new := idiv(ni, i2.range),
        i2new := imod(ni, i2.range),
        ch2 := SubstVars(ch, rec((i1.id) := i1new, (i2.id) := i2new)),
        SIMTISum(ASIMTBlockDimX(nn), ni, nn, ch2)    
));

# flatten Y/X -> X loops
ss := SubstTopDown(ss, 
    [@(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimY), @(2, SIMTISum, e->ObjId(e.simt_dim) = ASIMTBlockDimX), ...],
    e->let(s1 := @(1).val,
        i1 := s1.var,
        i2 := s1.child(1).var,
        rng := i1.range * i2.range,
        ii := Ind(rng),
        sr := rec(
            (i1.id) := idiv(ii, i2.range),
            (i2.id) := imod(ii, i2.range)
        ),
        sdim := ASIMTBlockDimX(rng),
        SIMTISum(sdim, ii, ii.range, SubstVars(s1.child(1).child(1), sr))
    )
);

# break and scramble GridDimX
ix := Ind(blk.x);
iy := Ind(blk.y);
iz := Ind(blk.z);

_tolin := b -> ix * V(blk.y) + iy + V(blk.x * blk.y) * iz;

ss := SubstBottomUp(ss, @(1, SIMTISum, e->ObjId(e.simt_dim) = ASIMTKernelFlag and ObjId(e.simt_dim.params[1]) = ASIMTGridDimX),
    e -> SIMTISum(ASIMTKernelFlag(ASIMTGridDimZ(blk.z)), iz, blk.z,
    SIMTISum(ASIMTGridDimY(blk.y), iy, blk.y,
        SIMTISum(ASIMTGridDimX(blk.x), ix, blk.x,
            SubstVars(@(1).val.child(1), rec((@(1).val.var.id) := _tolin(blk)))
))));

# unroll the lonely Radix-4 kernels
ss := SubstTopDown(ss, [@(1, ISum, e->e.var.range = 4), @(2, BB)],
	e -> BB(ISum(@(1).val.var, @(2).val.child(1))));



# temporary fix
opts.max_shmem_size := Product(szcube)/4;

c:= opts.codeSums(ss);
c := SubstBottomUp(c, [@(1, mul), @(2, Value, e->ObjId(e.t) = TVect and e.v[1] = V(1) and e.v[2] = V(-1)), @(3, vpack)],
	e->vpack(@(3).val.args[1], - @(3).val.args[2]));
c := SubstBottomUp(c, @(1, Value, e->ObjId(e.t) = TVect and e.v[1] = e.v[2]), e->e.v[1]);

#c := opts.fftxGen(tt);
c.ruletree := rt;
opts.prettyPrint(c);

PrintTo(name::file_suffix, opts.prettyPrint(c));
#fi;
