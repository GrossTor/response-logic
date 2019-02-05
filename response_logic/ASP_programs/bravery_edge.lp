#program brave.
:- not edge_brave(N1,N2,B) : node(N1), node(N2), B=0..1, @is_brave_structure(N2,N1,B) == 0.
%interpret as: is there any model which has an edge_brave that I have not yet collected in brave? If so than the statement above becomes False (it's a conjunction, all have elements have to be true for it to be true) and so the whole program is satisfiable and thus in the on_model function brave will be expanded by the considered edge_brave. This is repeated until there are no more missing edge_brave and thus the above statement turns true.

#show edge_brave/3.
