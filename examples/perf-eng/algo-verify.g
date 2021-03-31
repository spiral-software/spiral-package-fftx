v := 2;
N := v * [3, 5, 7];
d := Length(n);
t := MDDFT(N);
tm := MatSPL(t);
n := Product(N);


#stage := i -> L(n, N[i]) * Tensor(I(n/N[i]), DFT(N[i]));
stage := i -> Tensor(L(n/v, N[i]), I(v)) * Tensor(I(n/(N[i]*v)), Tensor(DFT(N[i]), I(v)) * L(N[i] * v, N[i]));

ss := Compose(List([1..3], stage));

InfinityNormMat(MatSPL(ss) - tm);
