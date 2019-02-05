#program edge_conversion.

edge_brave(OUT,IN,1):-edge(OUT,IN).
edge_brave(OUT,IN,0):-not edge(OUT,IN),node(OUT),node(IN).