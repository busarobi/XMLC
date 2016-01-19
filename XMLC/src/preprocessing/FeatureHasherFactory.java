package preprocessing;

public class FeatureHasherFactory {

	public static FeatureHasher createFeatureHasher( String hasher, int fhseed, int hd, int d ) {
		FeatureHasher fh = null;
		if ( hasher.compareTo("Universal") == 0 ) {			
			fh = new UniversalHasher(fhseed, hd, d);
		} else if ( hasher.compareTo("Murmur") == 0 ) {
			fh = new MurmurHasher(fhseed, hd, d);
		} else if ( hasher.compareTo("Mask") == 0 ) {
			fh = new MaskHasher(fhseed, hd, d);
		} else {
			System.out.println("Unknown hasher");
			System.exit(-1);
		}		
		return fh;
	}

}
