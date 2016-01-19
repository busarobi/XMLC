/**
 * This file is part of SADL, a library for learning all sorts of (timed) automata and performing sequence-based anomaly detection.
 * Copyright (C) 2013-2015  the original author or authors.
 *
 * SADL is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * SADL is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with SADL.  If not, see <http://www.gnu.org/licenses/>.
 */

package util;

import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 * @author Timo Klerx
 *
 */
public class MasterSeed {
	private static Logger logger = LoggerFactory.getLogger(MasterSeed.class);

	private static long seed = 7222525536004714236L;
	private static Random r = new Random(seed);
	private static boolean wasSet = false;

	public static void setSeed(long seed) {
		MasterSeed.seed = seed;
		if (wasSet) {
			System.err.println("Replacing Random object with new seed "+ seed);
		}
		r = new Random(seed);
		wasSet = true;
	}

	public static long nextLong() {
		return r.nextLong();
	}

	public static Random nextRandom() {
		long temp = r.nextLong();
		logger.info("Creating new Random object with seed={}",temp);
		return new Random(temp);
	}

	public static void reset(){
		r = new Random(seed);
	}
}
