Load(fftx);
ImportAll(fftx);
ImportAll(dct_dst);
ImportAll(realdft);
ImportAll(dtt);

Debug(true);

conf := LocalConfig.fftx.confGPU();

n := 15;

t := MDDST1([n,n,n]);

name := t.name::"_"::StringInt(n)::"x"::StringInt(n)::"x"::StringInt(n);

t := TFCall(t, rec(fname := name, params := []));
opts := conf.getOpts(t);

#opts := conf.getOpts(MDPRDFT([4,4,4]));
#opts.breakdownRules.DCT3 := [ DCT3_toSkewDCT3 ];
#opts.breakdownRules.DST1 := [ DST1_toDCT3 ];
#opts.breakdownRules.SkewDTT := [ SkewDTT_Base2, SkewDCT3_VarSteidl ];
#opts.breakdownRules.TTensorI := [CopyFields(IxA_L_split, rec(switch := true)),
#                        fftx.platforms.cuda.L_IxA_SIMT, fftx.platforms.cuda.IxA_L_SIMT,
#                        fftx.platforms.cuda.IxA_SIMT_peelof, fftx.platforms.cuda.IxA_SIMT_peelof2]::opts.breakdownRules.TTensorI;

tt := opts.tagIt(t);

rt := opts.search(tt);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo(name::".cu", opts.prettyPrint(c));
cyc := CMeasure(c, opts);

mm := CMatrix(c, opts);
tm := MatSPL(tt);

delta := InfinityNormMat(mm-tm);

