
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional real iDFTs

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.confGPU();

szcube := [ 32, 32, 32 ];
name := "imdprdft";             ##  ::StringInt(d)::"d";
sign := 1;

PrintLine("imdprdft-cuda: cube = ", szcube, ";\t\t##PICKME##");

szhalfcube := DropLast(szcube, 1)::[Int ( Last(szcube) / 2 ) + 1];
var_1:= var("var_1", BoxND([0,0,0], TReal));
var_2:= var("var_2", BoxND(szcube, TReal));
var_3:= var("var_3", BoxND(szhalfcube, TReal));
var_2:= X;
var_3:= Y;

t := TFCall(TDecl(TDAG([
       TDAGNode ( TTensorI ( IMDPRDFT ( szcube, sign ), 1, APar, APar ), var_3, var_2 ),
        ]),
        [var_1]
        ),
    rec ( fname:=name, params:= [ ] )
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".c", opts.prettyPrint(c));

