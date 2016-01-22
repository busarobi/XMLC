/**
 * This file is part of SADL, a library for learning all sorts of (timed) automata and performing sequence-based anomaly detection.
 * Copyright (C) 2013-2016  the original author or authors.
 *
 * SADL is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * SADL is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with SADL.  If not, see <http://www.gnu.org/licenses/>.
 */
package util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.PipedReader;
import java.io.PipedWriter;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.thoughtworks.xstream.XStream;


/**
 * 
 * @author Timo Klerx
 *
 */
public class IoUtils {
	private static Logger logger = LoggerFactory.getLogger(IoUtils.class);

	public static void deleteFiles(String[] strings) throws IOException {
		final Path[] paths = new Path[strings.length];
		for (int i = 0; i < strings.length; i++) {
			final Path p = Paths.get(strings[i]);
			paths[i] = p;
		}
		deleteFiles(paths);
	}

	public static void runGraphviz(final Path gvFile, final Path pngFile) {
		final String[] args = { "dot", "-Tpng", gvFile.toAbsolutePath().toString(), "-o", pngFile.toAbsolutePath().toString() };
		Process pr = null;
		try {
			pr = Runtime.getRuntime().exec(args);
		} catch (final Exception e) {
			logger.warn("Could not create plot of Model: {}", e.getMessage());
		} finally {
			if (pr != null) {
				try {
					pr.waitFor();
				} catch (final InterruptedException e) {
				}
			}
		}
	}

	public static void deleteFiles(Path[] paths) throws IOException {
		for (final Path p : paths) {
			if (!Files.deleteIfExists(p)) {
				logger.warn("{} should have been explicitly deleted, but did not exist.", p);
			}
		}
	}

	public static Object deserialize(Path path) throws FileNotFoundException, IOException, ClassNotFoundException {
		try (FileInputStream fileIn = new FileInputStream(path.toFile()); ObjectInputStream in = new ObjectInputStream(fileIn)) {
			final Object o = in.readObject();
			return o;
		}
	}


	public static void serialize(Object o, Path path) throws IOException {
		try (OutputStream fileOut = Files.newOutputStream(path); ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
			out.writeObject(o);
			out.close();
		}
	}

	public static void writeToFile(double[] testSample, Path classificationTestFile) throws IOException {
		try (BufferedWriter bw = Files.newBufferedWriter(classificationTestFile, StandardCharsets.UTF_8, StandardOpenOption.APPEND)) {
			bw.append(Arrays.toString(testSample).replace('[', ' ').replace(']', ' '));
			bw.append('\n');
		}

	}

	public static void writeToFile(List<double[]> testSamples, Path classificationTestFile) throws IOException {
		try (BufferedWriter bw = Files.newBufferedWriter(classificationTestFile, StandardCharsets.UTF_8, StandardOpenOption.APPEND)) {
			for (final double[] testSample : testSamples) {
				bw.append(Arrays.toString(testSample).replace('[', ' ').replace(']', ' '));
				bw.append('\n');
			}
		}
	}

	public static Object xmlDeserialize(Path path) {
		try {
			final String xml = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
			final XStream x = new XStream();
			return x.fromXML(xml);
		} catch (final Exception e) {
			logger.error("Unexpected exception", e);
		}
		return null;
	}

	public static void xmlSerialize(Object o, Path path) {
		String xml;
		try {
			final XStream x = new XStream();
			xml = x.toXML(o);
			Files.write(path, xml.getBytes());
		} catch (final Exception e) {
			logger.error("Unexpected exception", e);
		}
	}




}
