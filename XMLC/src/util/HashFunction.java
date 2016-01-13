package util;

public class HashFunction {
	private int seed;
	private int period;
	private boolean isSign;

	public HashFunction(int seed, int period) throws IllegalArgumentException {
		if ((period & 1) != 0) {
			throw new IllegalArgumentException("Period needs to be power of 2.");
		}
		this.seed = seed;
		this.period = period;
		this.isSign = false;
	}

	public HashFunction(int seed) {
		this.seed = seed;
		this.isSign = true;
	}

	/**
	 * Minimal implementation of MurmurHash3 (32bit)
	 */
	private int murmurhash(int key) {
		key *= 0xcc9e2d51;
		key = (key << 15) | (key >>> 17);
		key *= 0x1b873593;

		int h1 = this.seed;
		h1 ^= key;
		h1 = (h1 << 13) | (h1 >>> 19);
		h1 = h1 * 5 + 0xe6546b64;
		h1 ^= 4;
		h1 ^= h1 >>> 16;
        h1 *= 0x85ebca6b;
        h1 ^= h1 >>> 13;
        h1 *= 0xc2b2ae35;
        h1 ^= h1 >>> 16;
		return h1;
	}

	public int hash(int index) {
		if (this.isSign) return (this.murmurhash(index) & 1) * 2 - 1;
		return this.murmurhash(index) & (this.period - 1);
	}
}
