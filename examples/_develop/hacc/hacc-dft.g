
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

Debug(true);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

N := 16384; batch := 2;

# N := 1024; batch := 1024;
# N := 2048; batch := 4096;
# N := 4096; batch := 16384;
# N := 8192; batch := 65536;


name := "batch_dft_"::StringInt(batch)::"x"::StringInt(N);

t := TFCall(TRC(TTensorI(DFT(N, -1), batch, APar, AVec)), rec(fname := name, params := []));
#t := TFCall(TRC(DFT(N, -1)), rec(fname := name, params := []));

opts := conf.getOpts(t);
#opts.max_threads := 512;

tt := opts.tagIt(t);
 
 _tt := opts.preProcess(tt);
 rt := opts.search(_tt);

# s := opts.sumsRuleTree(rt);
# # --
# ImportAll(fftx.platforms.cuda);
# s := SumsRuleTree(rt, opts);
# s := fixUpSigmaSPL(s, opts);
# s := SubstTopDown(s, @(1, [ NoDiagPullinLeft, NoDiagPullinRight, Grp ]), (e) -> e.child(1));
# ss := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission)], BUA, opts);
# ss1 := FixUpCUDASigmaSPL_3Stage(ss, opts);
# ss2 := FixUpCUDASigmaSPL(ss1, opts);
# 
# #(s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
# #   When(Collect(t, PRDFT) :: Collect(t, IPRDFT) = [  ], FixUpCUDASigmaSPL(FixUpCUDASigmaSPL_3Stage(s1, opts), opts), FixUpCUDASigmaSPL_3Stage_Real(s1, opts)) )
#    
# ss := opts.postProcessSums(ss, opts);

# ==
c := opts.fftxGen(tt);
#c := SubstBottomUp(c, @(1, Value, e -> e.v = 1073741824), e->V(4*1073741824));

opts.prettyPrint(c);

# does it run?
cyc := CMeasure(c, opts);

# quick correctness check for large sizes
# get first non-trivial vector
cv := CVector(c, Replicate(2*batch, 0)::[1], opts);
cv1 := cv{[1..Length(cv)/batch]};
cv1a := Flat(List([0..N-1], k -> [re(E(N)^k).v, -im(E(N)^k).v]));

# correctnes test: true and \approx 10^-14 or so
ForAll(cv{[Length(cv)/batch+1..Length(cv)]}, i -> i = 0.0);
InfinityNormMat([cv1] - [cv1a]);



