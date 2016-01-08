package util;

public class HashFunction {
	private Murmur3A murmur;
	private int period;
	private boolean isSign;

	public HashFunction(int seed, int period) {
		this.murmur = new Murmur3A(seed);
		this.period = period;
		this.isSign = false;
	}

	public HashFunction(int seed) {
		this.murmur = new Murmur3A(seed);
		this.isSign = true;
	}

	public int hash(int index) {
		this.murmur.reset();
		this.murmur.updateInt(index);
		if (this.isSign) return (int) (this.murmur.getValue() % 2 * 2 - 1);
		return (int) (this.murmur.getValue() % this.period);
	}

}

