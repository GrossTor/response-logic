%"reacting" fully describes all nodes for all perturbations for each answer set. It can the be compared against the facts "reacted". "reacting" needs three arguments for brave consequences
node(0..n-1).
%{edge(OUT,IN):node(OUT)} :- node(IN).
{edge(OUT,IN)} :- node(IN),node(OUT).
reacting(I,IN):-edge(OUT,IN),pert(I,OUT).
reacting(I,IN):-edge(OUT,IN),reacting(I,OUT).
