package Learner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Queue;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.commons.math3.analysis.function.Sigmoid;

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import Data.NodeComparatorPLT;
import Data.NodePLT;
import Learner.step.StepFunction;
import jsat.linear.DenseVector;
import preprocessing.FeatureHasherFactory;
import preprocessing.MurmurHasher;
import preprocessing.UniversalHasher;
import threshold.ThresholdTuning;
import util.CompleteTree;
import util.MasterSeed;

public class BRTreeFHRNS extends BRTFHRNS {
	
	double innerThreshold = 0.05;
	
	
	public BRTreeFHRNS(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);

		System.out.println("#####################################################" );
		System.out.println("#### Learner: BRTreeFHRNS" );
		System.out.println("#####################################################" );

		this.threshold = 0.01;
		
	}
	
	public void allocateClassifiers(AVTable data) {
		super.allocateClassifiers(data);
		Arrays.fill(this.numOfUpdates, 1);
		Arrays.fill(this.numOfPositiveUpdates, 1);	
	}		
	
	@Override
	public void train(AVTable data) {
						
		TreeMap<Integer,Integer> nodesToUpdate = new TreeMap<>(); 
		//HashSet<Integer> negativeLabels = new HashSet<>();
		
		
		Random random = new Random(1);
		
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			// random permutation
			ArrayList<Integer> indirectIdx = this.shuffleIndex();
			
			for (int i = 0; i < traindata.n; i++) {
				
				int currIdx = indirectIdx.get(i);
				
				int numOfNegToSample = traindata.y[currIdx].length * this.samplingRatio;
				
				nodesToUpdate.clear();
				//negativeLabels.clear();
				
				for (int j = 0; j < traindata.y[currIdx].length; j++) {
					
					int treeIndex = this.tree.getTreeIndex(traindata.y[currIdx][j]);
					nodesToUpdate.put(treeIndex, 1);
					

				}
				
				int numOfNegativeLabels = 0;
				
				while(numOfNegativeLabels< numOfNegToSample && numOfNegativeLabels < this.m - numOfNegToSample ) {
					
					int treeIndex = this.tree.getTreeIndex(random.nextInt(this.m));
					
					if(!nodesToUpdate.containsKey(treeIndex)) {
						nodesToUpdate.put(treeIndex, 0);
						numOfNegativeLabels++;
					}
				}

				while(!nodesToUpdate.isEmpty()) {
				
					Entry<Integer,Integer> entry = nodesToUpdate.pollLastEntry();
					int j = entry.getKey();
					int v = entry.getValue();
					double posterior = getUncalibratedPosteriors(traindata.x[currIdx], j);
					
					int z = ((v > 0) ? 1 : 0);
					
					double inc = posterior -  (double) z; 

					updatedPosteriors(currIdx, j, inc, v > 1 ? v : 1);
					
					numOfUpdates[j]++;
					if(z == 1) numOfPositiveUpdates[j]++;
					else numOfNegativeUpdates[j]++;

					if(numOfUpdates[j] > 2000) {
						int vinc = posterior > this.innerThreshold ? 1 : 0;
						int parent = this.tree.getParent(j);
						if(parent >= 0) {
							Integer value = nodesToUpdate.get(parent);
							if(value == null) {
								nodesToUpdate.put(parent, vinc);
							} else {
								nodesToUpdate.put(parent, value + vinc);
							}
						}
					}
				}
				
								
				if ((i % 100000) == 0) {
					System.out.println( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					System.out.println("\t\t" + dateFormat.format(date));
					//System.out.println("Weight: " + this.w[0].get(0) );
					System.out.println("Scalar: " + this.scalararray[0]);
				}
			}

			System.out.println("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
		}
		
		int zeroW = 0;
		double sumW = 0;
		int maxNonZero = 0;
		int index = 0;
		for(double weight : w) {
			if(weight == 0) zeroW++;
			else maxNonZero = index;
			sumW += weight;
			index++;
		}
		System.out.println("Hash weights (lenght, zeros, nonzeros, ratio, sumW, last nonzero): " + w.length + ", " + zeroW + ", " + (w.length - zeroW) + ", " + (double) (w.length - zeroW)/(double) w.length + ", " + sumW + ", " + maxNonZero);
		
		for(int i = 0; i < this.t; i++) {
			this.contextChange[i] = this.computeContextChange(this.numOfPositiveUpdates[i], this.numOfUpdates[i], (this.traindata.n * this.epochs));
//			System.out.println(i + ": " + numOfUpdates[i] + " " + (this.epochs * this.traindata.n) + " " + (((double) this.numOfUpdates[i]/(double) (this.epochs * this.traindata.n))));
		}

	}

	
	
}
