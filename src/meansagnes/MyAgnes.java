/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package meansagnes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import weka.clusterers.AbstractClusterer;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;



/**
 *
 * @author Fiqie
 */
public class MyAgnes extends AbstractClusterer  {

    /** training data **/
    Instances instances;
    ArrayList<Cluster> clusters;
    ArrayList< ArrayList<Double> > distances;

    /** number of clusters desired in clustering **/
    int numClusters = 2;
    public void setNumClusters(int nClusters) {numClusters = Math.max(1,nClusters);}
    public int getNumClusters() {return numClusters;}

    /** distance function used for comparing members of a cluster **/
    protected DistanceFunction distanceFunction = new EuclideanDistance();
    public DistanceFunction getDistanceFunction() {return distanceFunction;}
    public void setDistanceFunction(DistanceFunction distanceFunction) {this.distanceFunction = distanceFunction;}

    /** the various link types */
    final static int SINGLE = 0;
    final static int COMPLETE = 1;
    
    double tempDistance =0;

    /**
    * Holds the Link type used calculate distance between clusters
    */
    int m_nLinkType = SINGLE;

    public void setLinkType(int i) {
        m_nLinkType = i;
    }
  
    public MyAgnes(int numCluster, int type) {
        super();
        this.numClusters = numCluster;
        this.m_nLinkType = type;
        clusters = new ArrayList<Cluster>();
        distances = new ArrayList<ArrayList<Double>>(); 
    }
    
    public void mergeCluster(int c1, int c2, int level, double distance){
        clusters.get(c2).setLevel(level);
        clusters.get(c2).setDistance(distance);
        //System.out.println("c2 = " + c2 + " size :" + clusters.size());
        clusters.get(c1).merge(clusters.get(c2));
        
        clusters.remove(c2);
        
        //update distance cluster 1
        for(int i = 0; i < distances.get(c1).size(); i++){
            if(m_nLinkType == SINGLE) {
                if(distances.get(c1).get(i) > distances.get(c2).get(i)){
                    distances.get(c1).set(i, distances.get(c2).get(i));
                }
            } else if(m_nLinkType == COMPLETE) {
                if(distances.get(c1).get(i) < distances.get(c2).get(i)){
                    distances.get(c1).set(i, distances.get(c2).get(i));
                }
            }
        }
        
        //remove column c2
        for(int i = 0; i < distances.size(); i++){
            distances.get(i).remove(c2);
        }
        //remove row c2
        distances.remove(c2);
     }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        instances = data;
        int level = 0;
        int c1 = -1;
        int c2 = -1;
        
        distanceFunction.setInstances(instances);
        int nInstance = instances.numInstances();
        if (nInstance == 0) {
            return;
        }
        for (int i = 0; i < nInstance; i++) {
            Cluster cluster = new Cluster(instances.instance(i));
            clusters.add(cluster);
        }
        //init distance matrix
        for(int i = 0; i < clusters.size(); i++){
            ArrayList<Double> distTemp = new ArrayList<>(); 
            for(int j = 0; j < clusters.size(); j++){
                double dist = distanceFunction.distance(clusters.get(i).getInstance(), clusters.get(j).getInstance());
                distTemp.add(dist);
            }
            distances.add(distTemp);
        }
        
        while(clusters.size() > numClusters){
            double min = Double.MAX_VALUE;
            for(int i = 0; i < clusters.size(); i++){
                for(int j = i + 1; j < clusters.size(); j++){
                    if(min > distances.get(i).get(j)) {
                        min = distances.get(i).get(j);
                        c1 = i;
                        c2 = j;
                    }
                }
            }
            if(min != tempDistance) {
                tempDistance = min;
                level++;
            } 
            mergeCluster(c1, c2, level, min);
        }
    }
    
    @Override
    public String toString(){
        String temp = "";
        temp+= "=== Model and evaluation on training set ===\n" +
                "\n" +
                "Clustered Instances\n\n";
        
//        for(int i = 0; i < clusters.size(); i++){
//            temp+= "Cluster " + i + "  " + clusters.get(i).getNumInstance(clusters.get(i)) +"\n";
//            System.out.println("cluster"+i);
//            clusters.get(i).printLevel(clusters.get(i));
//        }
        return temp;
    }
    @Override
    public int clusterInstance(Instance instance) {
        int clusterNum = 0;
        Double min = Double.MAX_VALUE;
        for(int i = 0; i<clusters.size(); i++){
            for (int j = 0; j<clusters.get(i).instances.size(); j++){
                double temp = distanceFunction.distance(clusters.get(i).instances.get(j), instance);
                if (temp<min){
                    min = temp;
                    clusterNum = i;
                }
            }
        }
        return clusterNum;
    }

    @Override
    public int numberOfClusters() throws Exception {
        return numClusters;
    }
}


