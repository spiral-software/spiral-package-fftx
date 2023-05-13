
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);
Import(simt);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

fwd := true;
#fwd := false;

nbatch := 2;
szcube := [ 64, 64, 64 ];

if fwd then
    prdft := MDPRDFT;
    sign := -1;
    name := "mdprdft_3d_batch";
else
    prdft := IMDPRDFT;
    sign := 1;
    name := "imdprdft_3d_batch";
fi;

# t := let(batch := nbatch,
#     apat := APar,
#     k := -1,
#     name := "dft"::StringInt(Length(ns))::"d_batch",  
#     TFCall(TTensorI(prdft(ns, k), nbatch, apat, apat), 
#         rec(fname := name, params := []))
# );

szhalfcube := DropLast ( szcube, 1 )::[ Int ( Last ( szcube ) / 2 ) + 1 ];
var_1:= var("var_1", BoxND([0,0,0], TReal));
var_2:= var("var_2", BoxND(szcube, TReal));
var_3:= var("var_3", BoxND(szhalfcube, TReal));
var_2:= X;
var_3:= Y;
t := TFCall ( TDecl ( TDAG ([
                                TDAGNode ( TTensorI ( prdft ( szcube, sign ), nbatch, APar, APar ), var_3, var_2 ),
                            ]),
                      [var_1]
                    ),
              rec ( fname := name, params := [ ] )
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);
