package Data;

import java.util.Comparator;

public class NodeComparatorPLT implements Comparator<NodePLT> {
    @Override
	public int compare(NodePLT n1, NodePLT n2) {
    	return (n1.p < n2.p) ? 1 : -1;
    }
} ;

