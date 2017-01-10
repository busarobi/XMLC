package test;

import org.junit.*;
import util.PrecomputedTree;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by Kalina on 09.01.2017.
 */

public class PrecomputedTreeTest {
    @org.junit.After
    public void tearDown() throws Exception {

    }
    // R-F metric based on a single hash (true distance is greater or equal to the approximate distance)

    @org.junit.Test
    public void robinsonFouldsDistance00_approximate() throws Exception {
        Integer[] arr1 =  new Integer[] {};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr1));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance_approximate(tree2);

        Assert.assertTrue(0.0 >= dist);
    }

    @org.junit.Test
    public void robinsonFouldsDistance01_approximate() throws Exception {
        Integer[] arr1 =  new Integer[] {};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,1,1, 4,2,1, 5,3,1, 6,4,1};
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance_approximate(tree2);

        Assert.assertTrue(0.0 >= dist);
    }


    @org.junit.Test
    public void robinsonFouldsDistance0_approximate() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,0,1};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr1));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance_approximate(tree2);

        Assert.assertTrue(0.0 >= dist);
    }

    @org.junit.Test
    public void robinsonFouldsDistance1_approximate() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,1,1, 4,2,1, 5,3,1, 6,4,1};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr1));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance_approximate(tree2);

        Assert.assertTrue(0.0 >= dist);
    }

    @org.junit.Test
    public void robinsonFouldsDistance2_approximate() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,1,1, 4,2,1, 5,3,1, 6,4,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,1,1, 4,3,1, 5,2,1, 6,4,1};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance_approximate(tree2);
        Assert.assertTrue(2.0 >= dist);
    }

    @org.junit.Test
    public void robinsonFouldsDistance22_approximate() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,1,1, 4,2,1, 5,3,1, 6,4,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 3,5,0, 3,6,0, 2,1,1, 4,2,1, 5,3,1, 6,4,1};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance_approximate(tree2);
        Assert.assertTrue(1.0 >= dist);
    }
    @org.junit.Test
    public void robinsonFouldsDistance30_approximate() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0, 7,1,1, 8,2,1, 9,3,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0, 7,1,1, 8,2,1, 9,3,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance_approximate(tree2);
        Assert.assertTrue(0.0 >= dist);
    }

    @org.junit.Test
    public void robinsonFouldsDistance31_approximate() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0, 7,1,1, 8,2,1, 9,3,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0, 7,1,1, 8,3,1, 9,2,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};

        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance_approximate(tree2);
        Assert.assertTrue(2.0 >= dist);
    }

    @org.junit.Test
    public void robinsonFouldsDistance32_approximate() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0,
                7,1,1, 8,2,1, 9,3,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0,
                7,1,1, 8,2,1, 9,5,1, 10,6,1, 11,3,1, 12,4,1, 13,7,1, 14,8,1};

        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance_approximate(tree2);
        Assert.assertTrue(2.0 >= dist);
    }

    @org.junit.Test
    public void robinsonFouldsDistance33_approximate() throws Exception {
        Integer[] arr1 = new Integer[]{0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 3, 0, 1, 4, 0, 2, 5, 0, 2, 6, 0, 3, 7, 0, 3, 8, 0, 4, 9, 0, 4, 10, 0, 5, 11, 0, 5, 12, 0, 6, 13, 0, 6, 14, 0,
                7, 1, 1, 8, 2, 1, 9, 3, 1, 10, 4, 1, 11, 5, 1, 12, 6, 1, 13, 7, 1, 14, 8, 1};
        Integer[] arr2 = new Integer[]{0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 3, 0, 1, 4, 0, 2, 5, 0, 2, 6, 0, 3, 7, 0, 3, 8, 0, 4, 9, 0, 4, 10, 0, 5, 11, 0, 5, 12, 0, 6, 13, 0, 6, 14, 0,
                7, 1, 1, 8, 7, 1, 9, 5, 1, 10, 6, 1, 11, 3, 1, 12, 4, 1, 13, 2, 1, 14, 8, 1};

        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance_approximate(tree2);
        Assert.assertTrue(4.0 >= dist);
    }

    // R-F metric

    @org.junit.Test
    public void robinsonFouldsDistance00() throws Exception {
        Integer[] arr1 =  new Integer[] {};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr1));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance(tree2);

        Assert.assertEquals(0, dist, 0.000001);
    }

    @org.junit.Test
    public void robinsonFouldsDistance01() throws Exception {
        Integer[] arr1 =  new Integer[] {};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,1,1, 4,2,1, 5,3,1, 6,4,1};
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance(tree2);

        Assert.assertEquals(0, dist, 0.000001);
    }


    @org.junit.Test
    public void robinsonFouldsDistance0() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,0,1};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr1));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance(tree2);

        Assert.assertEquals(0, dist, 0.000001);
    }

    @org.junit.Test
    public void robinsonFouldsDistance1() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,1,1, 4,2,1, 5,3,1, 6,4,1};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr1));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance(tree2);

        Assert.assertEquals(0, dist, 0.000001);
    }

    @org.junit.Test
    public void robinsonFouldsDistance2() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,1,1, 4,2,1, 5,3,1, 6,4,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,1,1, 4,3,1, 5,2,1, 6,4,1};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance(tree2);
        Assert.assertEquals(2.0, dist, 0.000001);
    }

    @org.junit.Test
    public void robinsonFouldsDistance22() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,1,1, 4,2,1, 5,3,1, 6,4,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 3,5,0, 3,6,0, 2,1,1, 4,2,1, 5,3,1, 6,4,1};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance(tree2);
        Assert.assertEquals(1.0, dist, 0.000001);
    }
    @org.junit.Test
    public void robinsonFouldsDistance30() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0, 7,1,1, 8,2,1, 9,3,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0, 7,1,1, 8,2,1, 9,3,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};
        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance(tree2);
        Assert.assertEquals(0.0, dist, 0.000001);
    }

    @org.junit.Test
    public void robinsonFouldsDistance31() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0, 7,1,1, 8,2,1, 9,3,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0, 7,1,1, 8,3,1, 9,2,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};

        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance(tree2);
        Assert.assertEquals(2.0, dist, 0.000001);
    }

    @org.junit.Test
    public void robinsonFouldsDistance32() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0,
                7,1,1, 8,2,1, 9,3,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0,
                7,1,1, 8,2,1, 9,5,1, 10,6,1, 11,3,1, 12,4,1, 13,7,1, 14,8,1};

        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance(tree2);
        Assert.assertEquals(2.0, dist, 0.000001);
    }

    @org.junit.Test
    public void robinsonFouldsDistance33() throws Exception {
        Integer[] arr1 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0,
                7,1,1, 8,2,1, 9,3,1, 10,4,1, 11,5,1, 12,6,1, 13,7,1, 14,8,1};
        Integer[] arr2 =  new Integer[] {0,0,0, 0,1,0, 0,2,0, 1,3,0, 1,4,0, 2,5,0, 2,6,0, 3,7,0, 3,8,0, 4,9,0, 4,10,0, 5,11,0, 5,12,0, 6,13,0, 6,14,0,
                7,1,1, 8,7,1, 9,5,1, 10,6,1, 11,3,1, 12,4,1, 13,2,1, 14,8,1};

        ArrayList<Integer> indices1 = new ArrayList<Integer>(Arrays.asList(arr1));
        ArrayList<Integer> indices2 = new ArrayList<Integer>(Arrays.asList(arr2));

        PrecomputedTree tree1 = new PrecomputedTree(indices1);
        PrecomputedTree tree2 = new PrecomputedTree(indices2);

        double dist = tree1.robinsonFouldsDistance(tree2);
        Assert.assertEquals(4.0, dist, 0.000001);
    }
///time and space
    @org.junit.Test
    public void robinsonFouldsDistance_1024() throws Exception {
        PrecomputedTree tree1 = new PrecomputedTree("XMLC/src/test/tree1024");
        PrecomputedTree tree2 = new PrecomputedTree("XMLC/src/test/tree1024");

        double dist = tree1.robinsonFouldsDistance(tree2);
        Assert.assertEquals(0.0, dist, 0.000001);
    }
//    @org.junit.Test
//    public void robinsonFouldsDistance_8192() throws Exception {
//        PrecomputedTree tree1 = new PrecomputedTree("XMLC/src/test/tree8192");
//        PrecomputedTree tree2 = new PrecomputedTree("XMLC/src/test/tree8192");
//
//        double dist = tree1.robinsonFouldsDistance(tree2);
//        Assert.assertEquals(0.0, dist, 0.000001);
//    }
//    @org.junit.Test
//    public void robinsonFouldsDistance_65536() throws Exception {
//        PrecomputedTree tree1 = new PrecomputedTree("XMLC/src/test/tree65536");
//        PrecomputedTree tree2 = new PrecomputedTree("XMLC/src/test/tree65536");
//
//        double dist = tree1.robinsonFouldsDistance(tree2);
//        Assert.assertEquals(0.0, dist, 0.000001);
//    }
//    @org.junit.Test
//    public void robinsonFouldsDistance_524288() throws Exception {
//        PrecomputedTree tree1 = new PrecomputedTree("XMLC/src/test/tree524288");
//        PrecomputedTree tree2 = new PrecomputedTree("XMLC/src/test/tree524288");
//
//        double dist = tree1.robinsonFouldsDistance(tree2);
//        Assert.assertEquals(0.0, dist, 0.000001);
//    }
//    @org.junit.Test
//    public void robinsonFouldsDistance_1048576() throws Exception {
//        PrecomputedTree tree1 = new PrecomputedTree("XMLC/src/test/tree1048576");
//        PrecomputedTree tree2 = new PrecomputedTree("XMLC/src/test/tree1048576");
//
//        double dist = tree1.robinsonFouldsDistance(tree2);
//        Assert.assertEquals(0.0, dist, 0.000001);
//    }
}
