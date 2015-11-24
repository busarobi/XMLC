package Data;

import java.util.List;

import jsat.linear.IndexValue;
import jsat.linear.SparseVector;
import jsat.linear.Vec;

public class SparseVectorExt extends SparseVector {

	private static final long serialVersionUID = -443341877809489447L;

	public SparseVectorExt(int length) {
		super(length);
	}

	public SparseVectorExt(List<Double> vals) {
		super(vals);
	}

	public SparseVectorExt(Vec toCopy) {
		super(toCopy);
	}

	public SparseVectorExt(int length, int capacity) {
		super(length, capacity);
	}

	public SparseVectorExt(int[] indexes, double[] values, int length, int used) {
		super(indexes, values, length, used);
	}

	public Vec sqrt() {
		Vec result = new SparseVectorExt(this.length());
		for (IndexValue iv : this) {
			result.set(iv.getIndex(), Math.sqrt(iv.getValue()));
		}
		return result;
	}

}
