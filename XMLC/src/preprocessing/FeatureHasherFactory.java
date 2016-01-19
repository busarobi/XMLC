package preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FeatureHasherFactory {
	private static Logger logger = LoggerFactory.getLogger(FeatureHasherFactory.class);

	public static FeatureHasher createFeatureHasher( String hasher, int fhseed, int hd, int d ) {
		FeatureHasher fh = null;
		if ( hasher.compareTo("Universal") == 0 ) {			
			fh = new UniversalHasher(fhseed, hd, d);
		} else if ( hasher.compareTo("Murmur") == 0 ) {
			fh = new MurmurHasher(fhseed, hd, d);
		} else if ( hasher.compareTo("Mask") == 0 ) {
			fh = new MaskHasher(fhseed, hd, d);
		} else {
			logger.info("Unknown hasher");
			System.exit(-1);
		}		
		return fh;
	}

}
