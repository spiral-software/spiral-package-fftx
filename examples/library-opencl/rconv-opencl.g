Load(fftx);
ImportAll(fftx);
ImportAll(simt);
Load(jit);
Import(jit);
conf := FFTXGlobals.defaultOpenCLConf();
var_1:= var("var_1", BoxND([24,32,21], TReal));
var_2:= var("var_2", BoxND([24,32,21], TReal));
var_3:= var("var_3", BoxND([24,32,40], TReal));
var_4:= var("var_4", BoxND([24,32,40], TReal));
var_5:= var("var_5", BoxND([24,32,21], TReal));
var_3:= X;
var_4:= Y;
symvar := var("sym", TPtr(TReal));
transform:= TFCall(TDecl(TDAG([
    TDAGNode(MDPRDFT([24,32,40],-1), var_1,var_3),
    TDAGNode(Diag(diagTensor(FDataOfs(symvar,16128,0),fConst(TReal, 2, 1))), var_2,var_1),
    TDAGNode(IMDPRDFT([24,32,40],1), var_4,var_2),

]),
   [var_1,var_2]
),
rec(fname:="rconv_spiral", params:= [symvar])
);
prefix:="rconv";
if 1 = 1 then opts:=conf.getOpts(transform);
tt:= opts.tagIt(transform);
if(IsBound(fftx_includes)) then opts.includes:=fftx_includes;fi;
c:=opts.fftxGen(tt);
 fi;
GASMAN("collect");
PrintOpenCLJIT(c,opts);