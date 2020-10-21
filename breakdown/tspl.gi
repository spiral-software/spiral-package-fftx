NewRulesFor(TFCall, rec(
    TFCall_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> c[1]
    )
));

NewRulesFor(TDeviceCall, rec(
    TFCall_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> c[1]
    )
));

NewRulesFor(TSparseMat, rec(
    TSparseMat_base := rec(
        forTransposition := false,
        children := nt -> [[ ]],
        applicable := (self, nt) >> When(IsBound(self.max_rows), nt.params[1][1] <= self.max_rows, true),
        apply := function(nt, c, cnt)
            local P, mat, i, j, rdata;
            P := nt.params;
            mat := MatSPL(ApplyFunc(O, P[1]));
            
            for i in [0..P[1][1]-1] do
                rdata := Drop(Filtered(P[2], ii->ii[1] = i)[1], 1);
                for j in rdata do
                    mat[i+1][j[1]+1] := j[2];
                od;
            od;   
            
            return Mat(mat);
        end
    ),
    TSparseMat_VStack := rec(
        forTransposition := false,
        applicable := nt -> nt.params[1][1] > 1,
        children := nt -> [ List([0..nt.params[1][1]-1], i->TSparseMat([1, nt.params[1][2]], [let(row := Filtered(nt.params[2], j->j[1] = i)[1], [0] :: Drop(row, 1))])) ],
        apply := (nt, c, cnt) -> BB(VStack(c))
    ),
));

NewRulesFor(TIterHStack, rec(
    TIterHStack_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(nt.getTags()), InfoNt(nt.params[2]) ]],
        apply := (nt, c, cnt) -> IterHStack(cnt[2].params[1], cnt[2].params[1].range, c[1])
    )
));

NewRulesFor(TIterVStack, rec(
    TIterVStack_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(nt.getTags()), InfoNt(nt.params[2]) ]],
        apply := (nt, c, cnt) -> IterVStack(cnt[2].params[1], cnt[2].params[1].range, c[1])
    )
));

NewRulesFor(TNoDiagPullinLeft, rec(
    TNoDiagPullinLeft_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> NoDiagPullinLeft(c[1])
    )
));

NewRulesFor(TNoDiagPullinRight, rec(
    TNoDiagPullinRight_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> NoDiagPullinRight(c[1])
    )
));

NewRulesFor(TNoPullLeft, rec(
    TNoPullLeft_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> NoPullLeft(c[1])
    )
));

NewRulesFor(TNoPullRight, rec(
    TNoPullRight_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> NoPullRight(c[1])
    )
));
