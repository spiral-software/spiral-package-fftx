##
##  Test to run random vectors for large transforms
##  Show for random inputs vectors that:
##  Cvector(v1) + Cvector(v2) = CVector(v1 + v2)
##
##  CVector is called to determine the length of vector (see measures.g)

if meas <> false then
    N := Length(cvec);
    if 1 = 1 then
        rmat := RandomMat(2, N);
	v1 := rmat[1];
	v2 := rmat[2];
	v3 := v1 + v2;

	cv1 := CVector(c, v1, opts);
	if not IsList(cv1) then
            PrintLine("CVector failed: Profiler did not run correctly");
            TestFailExit();
	fi;

	cv2 := CVector(c, v2, opts);
	if not IsList(cv2) then
            PrintLine("CVector failed: Profiler did not run correctly");
            TestFailExit();
	fi;

	cv3 := CVector(c, v3, opts);
	if not IsList(cv3) then
            PrintLine("CVector failed: Profiler did not run correctly");
            TestFailExit();
	fi;

	cv12 := cv1 + cv2;
	##  InfinityNormMat appears to sum over the length of the vector; divide by N
	inorm := InfinityNormMat ( [cv3] - [cv12] ) / N;
        if inorm > 1e-5 then
	    PrintLine("Transform failed -- max diff: ", inorm);
	    TestFailExit();
	fi;
	PrintLine("Infinity Norm difference for CVector(v1) + CVector(v2) - Cvector(v1 + v2) = ",
	          inorm );
    fi;
fi;
