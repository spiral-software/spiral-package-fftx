
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.confGPU();

szcube := [64, 64, 64];

prdft := MDPRDFT(szcube, 1);
iprdft := IMDPRDFT(szcube, -1);

symvar := var("amplitudes", TPtr(TReal));
name := "exaefl_kernel1";

domain := 2*Product(DropLast(szcube, 1))* (Last(szcube)/2+1);

i := Ind(domain);
x := var.fresh_t("x", TPtr(TReal));

pw_op := (x, a) -> a * cxpack(
    fdiv(re(x), sqrt(re(x)*re(x) +im(x)*im(x))), 
    fdiv(im(x), sqrt(re(x)*re(x) +im(x)*im(x))) 
);

cx_nth := (xr, i) -> cxpack(nth(xr, idiv(i, 2)), nth(xr, idiv(i, 2) + 1));

extract := (c, i) -> cond(eq(imod(i, 2), V(0)), re(c), im(c));

ampli := FDataOfs(symvar, domain, 0);
func := Lambda(i, Lambda(x, pw_op(cx_nth(x, i), ampli.at(i))));

t := TFCall(IMDPRDFT(szcube, 1) * Pointwise(func) * MDPRDFT(szcube, -1), 
    rec(fname := name, params := [symvar]));

opts := conf.getOpts(t);
tt := opts.tagIt(t);

_tt := opts.preProcess(tt);
rt := opts.search(_tt);
ss := opts.sumsRuleTree(rt);
#--

c := opts.codeSums(ss);

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));




