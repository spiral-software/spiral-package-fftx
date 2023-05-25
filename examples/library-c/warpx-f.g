
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# UNDER DEVELOPMENT

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.defaultConf();

prefix := "psatd_spiral";   ##  PICKME #define FUNCNAME psatd_spiral

t := let(name := prefix,
    n := 80,                ##  PICKME #define cubeN 80
    np := n+1,              ##  PICKME #define cubeNP (cubeN + 1)
    inFields := 11,         ##  PICKME #define INFIELDS 11
    outFields := 6,         ##  PICKME #define OUTFIELDS 6

    nf := n + 2,
    xdim := nf/2,           ##  PICKME #define cubeNF ((cubeN + 2)/2)     // C code wants xdim
    ydim := n,
    zdim := n,

    symvar := var("sym", TPtr(TPtr(TReal))),
    cvar := var("PhysConst::c", TReal),
    ep0var := var("PhysConst::ep0", TReal),
    c2 := cvar^2,
    invep0 := 1.0 / ep0var,
    ix := Ind(xdim),
    iy := Ind(ydim),
    iz := Ind(zdim),

    ii := lin_idx(iz, iy, ix),
    fmkx := nth(nth(symvar, 0), ix),
    fmky := nth(nth(symvar, 1), iy),
    fmkz := nth(nth(symvar, 2), iz),
    fcv := nth(nth(symvar, 3), ii),
    fsckv := nth(nth(symvar, 4), ii),
    fx1v := nth(nth(symvar, 5), ii),
    fx2v := nth(nth(symvar, 6), ii),
    fx3v := nth(nth(symvar, 7), ii),

    rmat := TSparseMat([outFields,inFields], [
        [0, [0, fcv / n^3],
            [4, cxpack(0, -fmkz * c2 * fsckv / n^3)],
            [5, cxpack(0, fmky * c2 * fsckv / n^3)],
            [6, -invep0 * fsckv / n^3],
            [9, cxpack(0,   fmkx * fx3v / n^3)],
            [10, cxpack(0, -fmkx * fx2v / n^3)]],
        [1, [1, fcv / n^3],
            [3, cxpack(0, fmkz * c2 * fsckv / n^3)], 
            [5, cxpack(0, -fmkx * c2 * fsckv / n^3)],
            [7, -invep0 * fsckv / n^3],
            [9, cxpack(0,   fmky * fx3v / n^3)],
            [10, cxpack(0, -fmky * fx2v / n^3)]],
        [2, [2, fcv / n^3],
            [3, cxpack(0, -fmky * c2 * fsckv / n^3)],
            [4,  cxpack(0, fmkx * c2 * fsckv / n^3)],
            [8, -invep0 * fsckv / n^3],
            [9, cxpack(0,   fmkz * fx3v / n^3)],
            [10, cxpack(0, -fmkz * fx2v / n^3)]],
        [3, [1, cxpack(0, fmkz * fsckv / n^3)],
            [2, cxpack(0, -fmky * fsckv / n^3)],
            [3, fcv / n^3],
            [7, cxpack(0, -fmkz * fx1v / n^3)],
            [8, cxpack(0, fmky * fx1v / n^3)]],
        [4, [0, cxpack(0, -fmkz * fsckv / n^3)],
            [2, cxpack(0, fmkx * fsckv / n^3)],
            [4, fcv / n^3],
            [6, cxpack(0, fmkz * fx1v / n^3)],
            [8, cxpack(0, -fmkx * fx1v / n^3)]],
        [5, [0, cxpack(0, fmky * fsckv / n^3)],      
            [1, cxpack(0, -fmkx * fsckv / n^3)],
            [5, fcv / n^3],
            [6, cxpack(0, -fmky * fx1v / n^3)],
            [7, cxpack(0, fmkx * fx1v / n^3)]]
        ]),

    boxBig0 := var("boxBig0", BoxNDF([11, n, n, n], TReal)),
    boxBig1 := var("boxBig1", BoxNDF([11, n, n, nf], TReal)),
    boxBig2 := var("boxBig2", BoxNDF([6, n, n, nf], TReal)),
    boxBig3 := var("boxBig3", BoxNDF([6, n, n, n], TReal)),
   
    TFCallF(
        TDecl(
            TDAG([
                  TDAGNode(TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]), nth(boxBig0, 0), nth(X, 0)),
                  TDAGNode(TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 1), nth(X, 1)),
                  TDAGNode(TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 2), nth(X, 2)),

                  TDAGNode(TResample([n, n, n], [n, n, np], [-0.5, -0.5, 0.0]), nth(boxBig0, 3), nth(X, 3)),
                  TDAGNode(TResample([n, n, n], [n, np, n], [-0.5, 0.0, -0.5]), nth(boxBig0, 4), nth(X, 4)),
                  TDAGNode(TResample([n, n, n], [np, n, n], [0.0, -0.5, -0.5]), nth(boxBig0, 5), nth(X, 5)),
                  
                  TDAGNode(TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]), nth(boxBig0, 6), nth(X, 6)),
                  TDAGNode(TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 7), nth(X, 7)),
                  TDAGNode(TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 8), nth(X, 8)),

                  TDAGNode(TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 9), nth(X, 9)),
                  TDAGNode(TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 10), nth(X, 10)),
                   
                  TDAGNode(TTensorI(MDPRDFT([n, n, n], -1), inFields, APar, APar), boxBig1, boxBig0),
                  TDAGNode(TRC(TMap(rmat, [iz, iy, ix], AVec, AVec)), boxBig2, boxBig1),
                  TDAGNode(TTensorI(IMDPRDFT([n, n, n], 1), outFields, APar, APar), boxBig3, boxBig2),

                  TDAGNode(TResample([np, np, n], [n, n, n], [0.0, 0.0, 0.5]), nth(Y, 0), nth(boxBig3, 0)),
                  TDAGNode(TResample([np, n, np], [n, n, n], [0.0, 0.5, 0.0]), nth(Y, 1), nth(boxBig3, 1)),
                  TDAGNode(TResample([n, np, np], [n, n, n], [0.5, 0.0, 0.0]), nth(Y, 2), nth(boxBig3, 2)),
                  
                  TDAGNode(TResample([n, n, np], [n, n, n], [0.5, 0.5, 0.0]), nth(Y, 3), nth(boxBig3, 3)),
                  TDAGNode(TResample([n, np, n], [n, n, n], [0.5, 0.0, 0.5]), nth(Y, 4), nth(boxBig3, 4)),
                  TDAGNode(TResample([np, n, n], [n, n, n], [0.0, 0.5, 0.5]), nth(Y, 5), nth(boxBig3, 5)),
            ]),          
            [boxBig0, boxBig1, boxBig2, boxBig3]        
        ), 
        rec(XType := TPtr(TPtr(TReal)), YType := TPtr(TPtr(TReal)), fname := name, params := [symvar ])
    )
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

if IsBound(opts) then
    opts.includes := [ "\"Utils/WarpXConst.H\" ", "<cstdlib>" ];
fi;

c := opts.fftxGen(tt);
outfil := prefix::".cpp";            ##  PICKME #define PSATDCODE "psatd_spiral.cpp"

PrintTo(outfil, opts.prettyPrint(c));

