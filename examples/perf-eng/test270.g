Load(fftx);
ImportAll(fftx);
ImportAll(fftx.platforms.cuda);
ImportAll(simt);

conf := LocalConfig.fftx.confGPU();
szcube := [270, 270, 270];
d := Length(szcube);
name := "mddft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

t := TFCall(TRC(MDDFT(szcube, 1)),
    rec(fname := name, params := []));

opts := conf.getOpts(t);
opts.target.forward := "thom";
opts.target.name := "linux-cuda";

PrintLine("DEBUG: opts = ", opts);

tt := opts.tagIt(t);
_tt := opts.preProcess(tt);
rt := opts.search(_tt);
ss := opts.sumsRuleTree(rt);
ss2 := Copy(ss);
# extract only one of the 3 stages of the 3D FFT -- they are all the same
ss := ss.child(1);
# this is a hack but only used to make the arrays of proper size
ss.ruletree := ss2.ruletree;
c := opts.codeSums(ss);

# now cones the testing
useTestCase := 0;

vars := Collect(c, [chain, @(1, simt_loop), @(2), @(3, simt_loop), @(4), @(5, simt_loop), @(6)]);
v1 := @(1).val.var;
v2 := @(5).val.var;
v3 := @(5).val.var;

if useTestCase = 0 then
    Print("original code");
fi;

if useTestCase = 1 then
    ## kill all writes to Y[] in stage 1 -- replace them by skip()
    ## in stage 3 replace all values written to Y[] by their index
    ## inspection of vector shows that all elements of Y are written in the third stage
    c := SubstBottomUp(c, [@(1, assign), @@(2, nth, (e,cx)->e.loc = Y and cx.simt_loop[2].var = v3), @(3)],
        e ->  assign(@@(2).val, @@(2).val.idx));
    c := SubstBottomUp(c, [@(1, assign), @@(2, nth, (e,cx)->e.loc = Y and cx.simt_loop[2].var = v1), @(3)],
          e-> skip());
fi;

if useTestCase = 2 then
    ## DO NOT kill writes to Y[] in stage 1 -- this is the only difference to test case 1
    ## in stage 3 replace all values written to Y[] by their index as in case 1
    ## inspection of vector shows that NOT ALL elements of Y are written in the third stage
    ## not sure how that is possible but I assume that something in the execution goes wrong
    c := SubstBottomUp(c, [@(1, assign), @@(2, nth, (e,cx)->e.loc = Y and cx.simt_loop[2].var = v3), @(3)],
        e ->  assign(@@(2).val, @@(2).val.idx));
fi;

if useTestCase = 3 then
    ## kill all writes to Y[] in stage 3 -- replace them by skip()
    ## in stage 1 replace all values written to Y[] by their index
    ## inspection of vector shows that all elements of Y are written in the first stage
    c := SubstBottomUp(c, [@(1, assign), @@(2, nth, (e,cx)->e.loc = Y and cx.simt_loop[2].var = v3), @(3)],
         e-> skip());
    c := SubstBottomUp(c, [@(1, assign), @@(2, nth, (e,cx)->e.loc = Y and cx.simt_loop[2].var = v1), @(3)],
        e ->  assign(@@(2).val, @@(2).val.idx));
fi;

if useTestCase = 4 then
    ## in stage 3 replace all loads from T1 and D1 with assignments 
    ## inspection of vector shows that NOT all elements of Y are written in the third stage
    c := SubstBottomUp(c, [@(1, assign), @(3), @@(2, nth, (e,cx)->e.loc = var.table.T1)],
        e ->  assign(@(3).val, @@(2).val.idx));
    c := SubstBottomUp(c, [@(1, assign), @(3), @@(2, nth, (e,cx)->e.loc = var.table.D1)],
        e ->  assign(@(3).val, @@(2).val.idx*V(-1.1)));
fi;

if useTestCase = 5 then
    ## in stage 3 replace all loads from T1 and D1 with assignments 
    ## now kill the writes to T1[] in stage 2
    c := SubstBottomUp(c, [@(1, assign), @(3), @@(2, nth, (e,cx)->e.loc = var.table.T1)],
        e ->  assign(@(3).val, @@(2).val.idx));
    c := SubstBottomUp(c, [@(1, assign), @(3), @@(2, nth, (e,cx)->e.loc = var.table.D1)],
        e ->  assign(@(3).val, @@(2).val.idx*V(-1.1)));
    c := SubstBottomUp(c, [@(1, assign), @@(2, nth, (e,cx)->e.loc = var.table.T1), @(3)],
         e-> skip());
fi;


PrintTo(name::"_"::StringInt(useTestCase)::".cu", opts.prettyPrint(c));

# in case 0 nothing is changed, computation is wrong
# in case 1 all outputs are properly written
# in case 2 NOT everything is written
# in case 3 all outputs are properly written
# in case 4 we replace all loasd in stage 4 with the index we read from. NOT all Y[] are written properly in stage 3
# in case 5 we kill off the writes to T1[] to trigger dead code elimination (?) and reduce register pressure. DOES NOT work 
vec := CVector(c, [1], opts);

## SUMMARY
# is it possible that we are at the edge of registers and not all threads successfully run but we do not catch the error?
# If I use skip() nvcc may do dead code elimination and all runs fine, but with writing the values to Y[] we run out of resources?

