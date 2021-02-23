
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

_computeNumDen := (value) -> let(
    list := Filtered([1..10000], e -> IntDouble(value * e) = value * e),

    When(value = 0.0, [0, 1],
	 When(Length(list) = 0, [0, 1],
	      [IntDouble(list[1] * value), list[1]]))
);

_shiftElements := (sizeIdx, size, shft) -> let(
    numDen := _computeNumDen(shft), # numDen[1]/numDen[2] = shft
    i := Ind([0..sizeIdx - 1]),
    onlyPositive := (sizeIdx < size),
    # Lambda(i, cond(lt(i, size / 2), E(numDen[2] * size) ^ (sign(2 * i) * numDen[1] * i), E(numDen[2] * size) ^ (sign(2 * i - size) * numDen[1] * (i - size))))
    # Lambda(i, cond(lt(i, size / 2), E(numDen[2] * size) ^ (numDen[1] * i), E(numDen[2] * size) ^ (numDen[1] * (i - size))))
    Lambda(i, cond(onlyPositive,
                   E(numDen[2] * size) ^ (numDen[1] * i),
                   cond(lt(i, size / 2),
                        E(numDen[2] * size) ^ (numDen[1] * i),
                        E(numDen[2] * size) ^ (numDen[1] * (i - size)))))
);

#_tabelize := f->let(vals := f.tolist(), i := Ind(f.domain()), When(
#    ForAll(vals, v->v=vals[1]), Lambda(i, vals[1]),
#    Lambda(i, FData(List(vals, j->Value(TComplex, _unwrap(j)))).at(i))
#));

_tabelize := f->let(vals := f.tolist(), i := Ind(f.domain()), When(
    ForAll(vals, v->v=vals[1]), Lambda(i, vals[1]),
    let(    rtab := FData(Flat(List(vals, e->[re(e), im(e)]))),
        Lambda(i, cxpack(rtab.at(2*i), rtab.at(2*i+1))))
));



NewRulesFor(TResample, rec(
		   TResample_TGath := rec(
		       forTransposition := false,
		       switch           := true,
		       applicable       := nt -> ForAny(Zip2(nt.params[1], nt.params[2]), l-> l[1]<l[2]),
		       children      := nt -> [[TResample(nt.params[1], nt.params[1], nt.params[3]), TGath(_toBox(nt.params[2], nt.params[1]))]],
		       apply := (nt, C, cnt) -> C[1] * C[2]
		   ),

		   TResample_TScat := rec(
		       forTransposition := false,
		       switch           := true,
		       applicable       := nt -> ForAny(Zip2(nt.params[1], nt.params[2]), l-> l[1]>l[2]),
		       children      := nt -> [[TScat(_toBox(nt.params[1], nt.params[2])), TResample(nt.params[2], nt.params[2], nt.params[3])]],
		       apply := (nt, C, cnt) -> C[1] * C[2]

		   ),

		   TResample_MD_nofrac := rec(
		       forTransposition := false,
		       switch           := true,
		       applicable       := nt -> (nt.params[1] = nt.params[2]) and (Length(nt.params[1]) > 1) and ForAll(nt.params[3], e -> e = 0.0),
		       children := nt -> [[I(Product(nt.params[1]))]],
		       apply := (nt, C, cnt) -> C[1]
		   ),

		   # ASSUMPTION IS THAT FASTEST DIMENSION IS REAL DFT-ED
		   TResample_MD_frac := rec(
		       forTransposition := false,
		       switch           := true,
		       applicable       := nt -> (nt.params[1] = nt.params[2]) and (Length(nt.params[1]) > 1) and ForAny(nt.params[3], e -> (abs(e) > 0.0) and (abs(e) < 1.0)),
		       children      := nt -> [[IMDPRDFT(nt.params[1], 1), MDPRDFT(nt.params[2], -1)]],

		       apply := (nt, C, cnt) -> let(
    			   sizeX := Last(nt.params[2]),
    			   sizeXf := sizeX / 2 + 1,
    			   shftX := Last(nt.params[3]),

    			   dXf := diagMul(fConst(sizeXf, 1/sizeX), _shiftElements(sizeXf, sizeX, shftX)),

    			   sizes := DropLast(nt.params[2], 1),
    			   shfts := DropLast(nt.params[3], 1),

    			   df := List([1..Length(sizes)], e -> diagMul(fConst(sizes[e], 1/sizes[e]), _shiftElements(sizes[e], sizes[e], shfts[e]))),
                   dd := List(df::[dXf], _tabelize),

    			   C[1] *
    			   RCDiag(RCData(diagTensor(dd))) *
    			   C[2]
		       )
		   )
	       )
	   );
