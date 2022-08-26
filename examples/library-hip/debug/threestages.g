Load(fftx);
ImportAll(fftx);

ImportAll(paradigms.vector);
ImportAll(simt);
ImportAll(fftx.platforms.cuda);

Debug(true);

conf := FFTXGlobals.defaultHIPConf();

n := 256;
radix := 8;
fftParIter := 16;

t := DFT(n).withTags([ASIMTBlockDimX]);

opts := conf.getOpts(t);

opts.breakdownRules.DFT := [CopyFields(DFT_CT, rec(maxSize := 8)), DFT_Base, 
    CopyFields(DFT_tSPL_CT, rec(filter := e->e[1] = radix)),
];

opts.breakdownRules.TTensorI := [ IxA_L_split, 
    CopyFields(AxI_SIMT_peelIter, rec(maxIterations := fftParIter)),
    IxA_TTwiddle_SIMT,
    CopyFields(IxA_DFT_CT_SIMT, rec(minDFTsize := radix, parIterations := fftParIter)),
    IxA_base, AxI_base,
    CopyFields(IxA_SIMT, 
        rec(applicable := (nt) -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsParPar(nt.params) and nt.params[2] > 1 and ObjId(nt.params[1]) <> TTwiddle and
            (ObjId(nt.params[1]) <> DFT or Maximum(nt.params[1].dims()) <= radix))), 
    CopyFields(AxI_SIMT, 
        rec(applicable := (nt) -> nt.hasTags() and _isSIMTTag(nt.firstTag()) and IsVecVec(nt.params) and nt.params[2] > 1 and nt.params[2] <= fftParIter
#         and (ObjId(nt.params[1]) <> DFT or Maximum(nt.params[1].dims()) < radix)
         )) 
    ];
opts.breakdownRules.TL := [ L_SIMT, L_base ];

opts.globalUnrolling := 16;
tt := opts.preProcess(t);

#opts.breakdownRules.DFT[1].filter := e -> e[1] = n1 and e[2] = n2;

rt := opts.search(tt);
#xx:= FindUnexpandableNonterminal(t, opts);
spl := SPLRuleTree(rt);

mm := MatSPL(spl);
mt := MatSPL(t);
mm = mt;

#c:= opts.codeSums(ss);
#c := opts.fftxGen(tt);
#opts.prettyPrint(c);

