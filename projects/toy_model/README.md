`toy_model_inference.py` demonstrates how to use the response logic approach on a synthetic example data set `z_vals.tsv`.

It generates the response pattern (threshold `z_thresh`), rectifies it (`response_logic.conform_response_pattern`) and then infers networks. Either the union over all conforming networks is inferred (`response_logic.brave_solving`) or all solutions are iterated (`response_logic.iterate_conforming_networks`).
It is also demonstrated how to use various additional constraints (`heuristics=`).
