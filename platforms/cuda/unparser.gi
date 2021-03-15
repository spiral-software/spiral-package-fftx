
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details


_dim3 := d -> "dim3("::d.id::")";

Class(HIPUnparser, CudaUnparser, rec(
    cu_call := (self, o, i, is) >>
        Print(Blanks(i), "hipLaunchKernelGGL(",  
                self.infix([o.func, _dim3(o.dim_grid), _dim3(o.dim_block), "0", "0"]::o.args, ", "), ");\n"),
));
