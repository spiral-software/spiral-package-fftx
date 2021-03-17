
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Import(realdft);
Import(filtering);
Import(paradigms.smp);

SIMD_ISA_DB.init0();
SIMD_ISA_DB.init1();

Include(openmp);
Include(opts);
