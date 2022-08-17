n1 := 16;
n2 := 16;
n := n1*n2;

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
       TDAGNode(TTensorI(MDDFT(szcube,sign),1,APar, APar), var_3,var_2),
              ]),
        [var_1]
        ),
    rec(fname:=name, params:= [symvar])
);

opts := conf.getOpts(t);

# temporary fix
ImportAll(fftx.platforms.cuda);
ImportAll(simt);
#opts.tags := opts.tags{[1,3]};


##  We need the Spiral functions wrapped in 'extern C' for adding to a library
##  Comment out next line if trying standalone or with profiler
opts.wrapCFuncs := true;
tt := opts.tagIt(t);
if(IsBound(fftx_includes)) then opts.includes:=fftx_includes; fi;

_tt := opts.preProcess(tt);

opts.breakdownRules.MDDFT := [ fftx.platforms.cuda.MDDFT_tSPL_KL_SIMT];
Append(opts.breakdownRules.DFT, [CopyFields(DFT_GoodThomas, rec(switch := true))]);
opts.breakdownRules.DFT[3].forcePrimeFactor := true;

#opts.breakdownRules.TTensorI := [ CopyFields(IxA_L_split, rec(switch := true)), CopyFields(L_IxA_split, rec(switch := true)), 
#    fftx.platforms.cuda.IxA_SIMT, fftx.platforms.cuda.AxI_SIMT,
#    fftx.platforms.cuda.IxA_SIMT_peelof, fftx.platforms.cuda.IxA_SIMT_peelof2 ];

# temporary fix
opts.breakdownRules.DFT[1].filter := e -> e[1] = n1 and e[2] = n2;

rt := opts.search(_tt);
ss := opts.sumsRuleTree(rt);

if true then
# data layout trick for 256
cl := V(32);
rw := V(256);

i := Ind(rw*cl);
shft := V(1);
shft2 := V(1);

f := Lambda(i, idiv(i, cl) * cl + imod(imod(i, cl) + idiv(i, cl) * shft + idiv(i, cl*cl)*shft2, cl)).setRange(cl.v*rw.v);
#f := Lambda(i, idiv(i, cl) * cl + imod(imod(i, cl) + idiv(i, cl) * shft, cl)).setRange(cl.v*rw.v);

#Set(List(fl{[0..63]*32+1}, i->Mod(i, 64)));
#List([1..32], k->Set(List(fl{[0..63]*32+k}, i->Mod(i, 64)))=Set([0..63]));


##fl := List(f.tolist(), i->i.v);
##Set([0..rw.v*cl.v-1]) = Set(fl);
##List([1..64], k->Set(List(fl{[0..63]*64+k}, i->Mod(i, 64)))=Set([0..63]));
##List([1..64], k->Set(List(fl{[0..63]*128+k}, i->Mod(i, 64)))=Set([0..63]));

scat := Scat(f);
gath := Gath(f);

ImportAll(simt);
#sx := Collect(ss, @(1, [Compose, [SIMTISum, SIMTISum]], e->Cols(e.child(1)) =  cl.v*rw.v));

ss := SubstBottomUp(ss, @(1, [Compose, [SIMTISum, SIMTISum]], e->Cols(e.child(1)) =  cl.v*rw.v),
    e->Grp(@(1).val.child(1) * gath) * Grp(scat * @(1).val.child(2)));

ss := RulesSums(ss);
ss := SubstBottomUp(ss, @(1, Grp), e->e.child(1));

fi;

# temporary fix
opts.max_shmem_size := Product(szcube)/4;

c:= opts.codeSums(ss);

#c := opts.fftxGen(tt);
##  opts.prettyPrint(c);
PrintTo(name::file_suffix, opts.prettyPrint(c));
#fi;
