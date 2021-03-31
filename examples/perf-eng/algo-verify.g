v := 20;
N := v * [5,5,5];

#v := 2;
#N := 2 * [3,5,7];

d := Length(n);
t := MDDFT(N);
n := Product(N);


#stage := i -> L(n, N[i]) * Tensor(I(n/N[i]), DFT(N[i]));
#stage := i -> Tensor(L(n/v, N[i]), I(v)) * Tensor(I(n/(N[i]*v)), Tensor(DFT(N[i]), I(v)) * L(N[i] * v, N[i]));
stage := i -> Tensor(L(n/v, N[i]), I(v)) * Tensor(I(n/(N[i]*v)), L(N[i] * v, N[i])*  Tensor(I(v), DFT(N[i])));

ss := Compose(List([1..3], stage));

InfinityNormMat(MatSPL(ss) - MatSPL(t));
