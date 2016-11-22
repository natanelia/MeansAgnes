/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package meansagnes;

import java.util.ArrayList;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author FiqieUlya
 */
public class Cluster {
    private Instance  instance;
    private ArrayList<Cluster> siblings;
    public ArrayList<Instance> instances;
    private int numInstances =1;
    private int level;
    private double distance;
    
    protected DistanceFunction distanceFunction = new EuclideanDistance();
    public DistanceFunction getDistanceFunction() {return distanceFunction;}
    public void setDistanceFunction(DistanceFunction distanceFunction) {this.distanceFunction = distanceFunction;}
    Cluster(Instance i){
        instance = i;
        level = 0;
        distance = Double.MAX_VALUE;
        siblings = new ArrayList<Cluster>();
        instances = new ArrayList<Instance>();
        instances.add(i);
    }
    
    public int getLevel(){
        return level;
    }
    
    public void setLevel(int x){
        level = x;
    }
    
    public double getDistance(){
        return distance;
    }
    
    public void setDistance(double d){
        distance = d;
    }
    
    public Instance getInstance(){
        return instance;
    }
    
    public void setInstance(Instance i){
        instance = i;
    }
    
    public Cluster getSibling(int i){
        return siblings.get(i);
    }
    
    public void merge(Cluster c1){
        siblings.add(c1);
        numInstances += c1.numInstances;
        instances.addAll(c1.instances);
    }
    
    public String toString(){
        String temp ="";
        
        return temp;
    }
    
    
//    public void printLevel(Cluster c){
//        if (c.siblings.size()!=0){
//            for(int i=0 ; i< c.siblings.size();i++){
//                printLevel(c.siblings.get(i));
//                System.out.println(c.siblings.get(i).level);
//                System.out.println(c.siblings.get(i).distance);
//            }
//        }
//    }
//    
    public int getNumInstance(Cluster c){
        /*int sum = 0;
        //System.out.println(" size" +c.siblings.size());
        if(c.siblings.isEmpty()){
            return 1;
        } else {
            for(int i = 0; i < c.siblings.size(); i++){
                //System.out.println("["+i+"]  size" +c.siblings.size());
                sum+= getNumInstance(c.siblings.get(i));
            }
            return sum;
        } */
        return numInstances;
        /*int sum = 0;
        while(){
            
        }*/
    }
}
