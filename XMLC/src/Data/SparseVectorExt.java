package Data;

import java.util.List;

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
		int[] idx = indexes.clone();
		double[] vals = new double[values.length];
		for (int i = 0; i < used; i++) {
			vals[i] = Math.sqrt(values[i]);
		}
		Vec result = new SparseVectorExt(idx, vals, this.length(), used);
		return result;
	}

}
