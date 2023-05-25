
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# SPIRAL FFTX package loader

LoadImport(simt);

Import(realdft);
Import(filtering);

Declare(_toBox);

Include(types);
LoadImport(fftx.sigma);
LoadImport(fftx.nonterms);
LoadImport(fftx.breakdown);
LoadImport(fftx.rewrite);
LoadImport(fftx.codegen);

Include(sumsgen);
Include(fftxapi);
Include(opts);

LoadImport(fftx.library);

LoadImport(fftx.platforms);
LoadImport(fftx.knowledgebase);

