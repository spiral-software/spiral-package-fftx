Class(MultiPtrSumsgenMixin, rec(
    IterHStack := 
        (self, o, opts) >> let(
        	bkcols := Cols(o.child(1)),
        	bkrows := Rows(o.child(1)),
        	nblocks := o.domain,
        	cols := Cols(o), rows := Rows(o),
        	j := o.var,
        	ch := self(o.child(1), opts),
    
    	    ISum(j, j.range,
        		ch *
        		Gath(fTensor(fBase(nblocks, j), fId(bkcols)))
        	)
        )    
));

