package util;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;

import Data.AVTable;

public class TestHuffmanTree {

	HuffmanTree tree;

	@Before
	public void setUp() throws Exception {
		AVTable data = new AVTable();
		data.m = 11;
		data.n = 9;
		data.y = new int[][] { { 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0 }, { 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0 },
				{ 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0 }, { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0 },
				{ 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0 }, { 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0 },
				{ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 } };
		tree = new HuffmanTree(data);
	}

	@Test
	public void testChildrenToCode() {
		assertEquals(230L, tree.childrenToCode(20, 20));
		assertEquals(59L, tree.childrenToCode(20, 2));
	}

	@Test
	public void testDivideCode() {
		assertEquals(2, tree.divideCode(59, 1));
		assertEquals(1, tree.divideCode(59, 2));
		assertEquals(0, tree.divideCode(59, 3));
	}

	@Test
	public void testCodeToChildren() {
		ArrayList<Integer> result;

		result = tree.codeToChildren(0L);
		assertEquals(0, result.get(0).intValue());
		assertEquals(0, result.get(1).intValue());

		result = tree.codeToChildren(229L);
		assertEquals(19, result.get(0).intValue());
		assertEquals(20, result.get(1).intValue());
	}
}
