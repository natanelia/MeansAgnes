/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package meansagnes;

import java.util.ArrayList;
import weka.core.Instance;

/**
 *
 * @author FiqieUlya
 */
public class Cluster {
    private Instance  instance;
    private ArrayList<Cluster> siblings;
    private int numInstances =1;
    private int numThreshold;
    private double distance;
    
    Cluster(Instance i){
        instance = i;
        numThreshold = 0;
        distance = Double.MAX_VALUE;
        siblings = new ArrayList<Cluster>();
    }
    
    public int getNumThreshold(){
        return numThreshold;
    }
    
    public void setNumThreshold(int x){
        numThreshold = x;
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
    }
    
    public String toString(){
        String temp ="";
        
        return temp;
    }
    
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
