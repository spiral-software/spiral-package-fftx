n1 := 16;
n2 := 16;
n := n1*n2;

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
    vRC_Scat := Rule([vRC, @(1, Scat)], e -> VScat(@(1).val.func, 2))
));


opts := conf.getOpts(t);
opts.cudasubName := name;


# set up vector stuff
Class(HIP_2x64f, SSE_2x64f);
isa := HIP_2x64f;

isa.mul_cx := (self, opts) >> (
    (y,x,c) -> let(u1 := var.fresh_t("U", TVectDouble(2)), u2 := var.fresh_t("U", TVectDouble(2)),
        u3 := var.fresh_t("U", TVectDouble(2)), u4 := var.fresh_t("U", TVectDouble(2)),
        decl([u1, u2, u3, u4], chain(
            assign(u1, mul(x, velem(c, 0))),                # vushuffle_2x64f(c, [1,1])
            assign(u2, vpack(velem(x, 0), -velem(x, 1))),   #chshi_2x64f(x)),
            assign(u3, mul(u2, velem(c, 1))),               # vushuffle_2x64f(c, [2,2])
            assign(u4, vpack(velem(x, 1), velem(x, 0))),    # vushuffle_2x64f(u3, [2,1])
            assign(y, add(u1, u4))
        ))));

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

opts.vector := vopts.vector;
opts.vector.SIMD := "SSE2";


# temporary fix
#opts.tags := opts.tags{[1,3]};



##  We need the Spiral functions wrapped in 'extern C' for adding to a library
##  Comment out next line if trying standalone or with profiler
opts.wrapCFuncs := true;
if(IsBound(fftx_includes)) then opts.includes:=fftx_includes; fi;

tt := opts.tagIt(t);
_tt := opts.preProcess(tt);

opts.breakdownRules.MDDFT := [ fftx.platforms.cuda.MDDFT_tSPL_KL_SIMT];
Append(opts.breakdownRules.DFT, [CopyFields(DFT_GoodThomas, rec(switch := true)), CopyFields(DFT_SIMT_cplxvect, rec(switch := true))]);
opts.breakdownRules.DFT[3].forcePrimeFactor := true;

# temporary fix
opts.breakdownRules.DFT[1].filter := e -> e[1] = n1 and e[2] = n2;

rt := opts.search(_tt);

ss := opts.sumsRuleTree(rt);
ss := RulesCxRC_Op(ss);
ss := RulesCxRC_Term(ss);

# break loops
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


# temporary fix
opts.max_shmem_size := Product(szcube)/4;

c:= opts.codeSums(ss);

#c := opts.fftxGen(tt);
##  opts.prettyPrint(c);

PrintTo(name::file_suffix, opts.prettyPrint(c));
#fi;
