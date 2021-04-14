##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# opts.target := rec ( name := "<cuda-target>" );
      
meas := CMeasure(c, opts);
if meas = false or (meas < 0) or (meas >= 1e+100) then
    PrintLine("CMeasure failed: Profiler did not run correctly");
    PrintLine("Not attempting CVector or CMatrix");
    TestFailExit();
else
    PrintLine("Time measured = ", meas);
    ##  Echo the file "time.txt" from the temporary output directory to stdout
    tmpdir := _MakeOutDirString(opts);
    timefile := tmpdir::"/time.txt";
    cmdstr := "cat <"::timefile;
    Exec(cmdstr);
fi;
