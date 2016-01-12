package util;

public class HashFunction {
	private Murmur3A murmur;
	private int period;
	private boolean isSign;

	public HashFunction(int seed, int period) throws IllegalArgumentException {
		this.murmur = new Murmur3A(seed);
		if ((period & 1) != 0) {
			throw new IllegalArgumentException("Period needs to be power of 2.");
		}
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
		if (this.isSign) return (int) ((this.murmur.getValue() & 1) * 2 - 1);
		return (int) (this.murmur.getValue() & (this.period - 1));
	}
}

