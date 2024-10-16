#########################
# Now let's do a DAG.
# Start with setting nx and ny and nz.  Do NOT set X or Y!
# Example:
# ( echo "nx:=8; ny:=7; nz:=6;" ; cat bracket3d.g ) | $SPIRAL_HOME/bin/spiral

Load(fftx);
ImportAll(fftx);
Import(realdft);

# conf := LocalConfig.fftx.defaultConf();
conf := LocalConfig.fftx.confGPU(); # for CUDA

# These should be added to FFTX library soon.
Class(HProduct, RowVec);
DefaultCodegen.HProduct := (self, o, y, x, opts) >> let(i := Ind(), func := o.element.lambda(),
        t := TempVar(x.t.t),
        chain(assign(t,1),
            loop(i, func.domain(), assign(t, mul(t, mul(func.at(i), nth(x,i))))),
            assign(nth(y,0), t)));

nx:=8; ny:=7; nz:=6;
nyTrunc := ny+2 - ny mod 2;
nxhalf := (nx - nx mod 2) / 2;
nyhalf := (ny - ny mod 2) / 2;
Kx := [0..nxhalf]::[(nxhalf+1-nx)..-1];
Ky := [0..nyhalf];
imultRCmat := Mat([[0, -1], [1, 0]]);
# should scale by X0
opiKx := Tensor(I(nz), Tensor(Diag(Kx), Tensor(I(nyTrunc/2), imultRCmat)));
# should scale by Y0
opiKy := Tensor(I(nz), Tensor(I(nx), Tensor(Diag(Ky), imultRCmat)));
opC2Rfft := Scale(1/(nz*nx*ny), IMDPRDFT([nz, nx, ny], 1));

# Hadamard product (real 2*nx*ny to nx*ny):
# multiply each element in top half of input by
# corresponding element in bottom half of input.
hadp := Tensor(HProduct(1, 1), I(nz*nx*ny));
# FIXME: Something funny with this.  It can't be converted to a matrix.

# Subtraction (2*nx*ny to nx*ny):
# from each element in top half of input, subtract
# corresponding element in bottom half of input.
sub := Tensor(RowVec(1, -1), I(nz*nx*ny));

opR2Cfft := MDPRDFT([nz, nx, ny], -1);


### bracket2d := opR2Cfft * sub * Tensor(I(2), hadp) * Tensor(Z(4, 1), opC2Rfft) * Tensor(I(2), VStack(opiKx, opiKy));

# Input variables:
# truncated representation of complex vectors of length n.
_x1 := var("x1", TArray(TReal, nz*nx*nyTrunc));
_x2 := var("x2", TArray(TReal, nz*nx*nyTrunc));

_X0 := var("X0", TReal);
_Y0 := var("Y0", TReal);

# Intermediate variables:  real.
_Kfg := var("Kfg", TArray(TReal, 4*nz*nx*nyTrunc));
_rKfg := var("rKfg", TArray(TReal, 4*nz*nx*ny));
_prodrKfg := var("prodrKfg", TArray(TReal, 2*nz*nx*ny));
_diffprodrKfg := var("diffprodrKfg", TArray(TReal, nz*nx*ny));

convdag := TDecl(TDAG([
    # Each list element: TDAGNode(operator, output, input).
    TDAGNode(Tensor(I(2), VStack(Scale(1/_X0, opiKx), Scale(1/_Y0, opiKy))), _Kfg, X),
    TDAGNode(Tensor(Z(4, 1), opC2Rfft), _rKfg, _Kfg),
    TDAGNode(Tensor(I(2), hadp), _prodrKfg, _rKfg),
    TDAGNode(sub, _diffprodrKfg, _prodrKfg),
    TDAGNode(opR2Cfft, Y, _diffprodrKfg),
]), [_x1, _x2, _Kfg, _rKfg, _prodrKfg, _diffprodrKfg]);

funcname := "bracket3d_"::StringInt(nx)::"_"::StringInt(ny)::"_"::StringInt(nz);

t := let(name := funcname,
    TFCall(convdag, rec(fname := name, params := [_X0, _Y0]))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
# opts.prettyPrint(c);
PrintTo(funcname::".cu", opts.prettyPrint(c));
