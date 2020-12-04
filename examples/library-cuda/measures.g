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

if meas <> false then
    if 1 = 1 then
        lv := [0, 0, 1, 0];
        cvec := CVector(c, lv, opts);
	if not IsList(cvec) then
            PrintLine("CVector failed: Profiler did not run correctly");
            TestFailExit();
	fi;
        Print("Cvector found, length = ", Length(cvec), "\n");
    fi;
fi;

if meas <> false then
    if Length(cvec) < 4100 then
        if 1 = 1 then
            cmat := CMatrix(c, opts);
	    if not IsList(cmat) then
	        PrintLine("CMatrix failed: Profiler did not run correctly");
		TestFailExit();
	    fi;
	    PrintLine("Cmatrix read");   ## , SIZE(cmat)
	       
            sm := MatSPL(t);
	    diff := 1;
	    diff := cmat - sm;
	    if not IsList(diff) then
	        PrintLine("CMatrix failed -- matrix size mismatch: Profiler did not run correctly");
		TestFailExit();
	    fi;

	    inorm := InfinityNormMat(diff);
            if inorm > 1e-5 then
	        PrintLine("Transform failed -- max diff: ", inorm);
		TestFailExit();
	    fi;
	    PrintLine("Infinity Norm Value = ", inorm);
	fi;
    else
	PrintLine("Not attempting CMatrix: too large, vector length = ", Length(cvec));
    fi;
fi;
