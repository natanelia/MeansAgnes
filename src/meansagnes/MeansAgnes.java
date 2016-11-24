/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package meansagnes;

import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Natan
 */
public class MeansAgnes {
    private Instances data;
    private Clusterer model;
    private int clusterer, numCluster = 0, agnesType = 0;
    
    public MeansAgnes(){
        data = null;
        model = null;
        clusterer = 0;
    }
    
    public void setClusterer(int i){
        clusterer = i;
    }
    
    public void setNumCluster(int i){
        numCluster = i;
    }
    
    public void setAgnesType(int i){
        agnesType = i;
    }
    //load data (arrf dan csv)
    public void loadFile(String data_address){
        try {
            data = ConverterUtils.DataSource.read(data_address);
            System.out.println("LOAD DATA BERHASIL\n\n");
            System.out.println(data.toString() + "\n"); 
            if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception ex) {
            System.out.println("File gagal di-load");
        }     
    }
    
    //remove atribut
    public void removeAttribute(int[] idx){
        try{
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(idx);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
            System.out.println(data.toString() + "\n");
        } catch (Exception ex){
            System.out.println("Gagal remove attribute!");
        }       
    }
    
    //Filter: Resample
    public void resample(double b, double z, int seed){
        try {
            System.out.println(data.toString() + "\n");
            Resample resampleFilter = new Resample();
            
            resampleFilter.setInputFormat(data);
            resampleFilter.setNoReplacement(false);
            resampleFilter.setBiasToUniformClass(b); // Uniform distribution of class
            resampleFilter.setSampleSizePercent(z);
            resampleFilter.setRandomSeed(seed);
            
            data = Filter.useFilter(data,resampleFilter);
            
            /*Random R = new Random();
            data.resample(R);*/
            System.out.println("HASIL RESAMPLE\n\n");
            System.out.println(data.toString() + "\n");
        } catch (Exception ex) {
            Logger.getLogger(MeansAgnes.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void buildClusterer(int type, Instances train){
        //Classifier model = null;
        switch (type) {
            case 0:
                model = new MyAgnes(numCluster, agnesType);
                break;
            case 1 :
                model = new MyKMeans(numCluster);
                break;
            default:
                break;
        }
        try {
            model.buildClusterer(train);
//            System.out.println(model.toString());
            //return model;
        } catch (Exception ex) {
            Logger.getLogger(MeansAgnes.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    
    // percentage split
    public void percentageSplit(double percent){
        try {
            data.randomize(new java.util.Random(0));
            int trainSize = (int) Math.round((double) data.numInstances() * percent/100f);
            int testSize = data.numInstances() - trainSize;
            
            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);
            
            buildClusterer(clusterer, train);

            ClusterEvaluation eval = new ClusterEvaluation();
            eval.setClusterer(model);
            eval.evaluateClusterer(test);
            System.out.println(eval.clusterResultsToString());
        } catch (Exception ex) {
            System.out.println(ex);
        }
    }
    
    //Save Model
    public void saveModel(String modelname){
        try {
            SerializationHelper.write(modelname, model);
            System.out.println("berhasil disave\n");
        } catch (Exception ex) {
            System.out.println("gagal di save\n");
        }
    }
 
    //Load Model
    public Clusterer loadModel(String modeladdress){
        Clusterer model = null;
        try {
            model = (Clusterer) SerializationHelper.read(modeladdress);
            System.out.println(model.toString());
            System.out.println("Berhasil Load Model\n");
        } catch (Exception ex) {
            System.out.println("tidak bisa diload\n");
        }
        return model;
    }
   
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String file = "";
        int clusterer;
        String testfile = "";
        MeansAgnes w = new MeansAgnes();
        Scanner scan = new Scanner(System.in);
        Clusterer model = null;
        boolean stat = true;
        while(stat){
            System.out.println("\n\nProgram Eksplorasi Weka");
            System.out.println("1. Load data set");
            System.out.println("2. Filter : Resample");
            System.out.println("3. Remove Attribute");
            System.out.println("4. Build Classifier");
            System.out.println("5. Percentage Split");
            System.out.println("6. Save model");
            System.out.println("7. Load model");
            System.out.println("8. Prediction");
            System.out.println("9. Exit");
            System.out.print("Pilih Menu : "); 
            int option = scan.nextInt();
            if(option == 1) {
                System.out.println("====LOAD DATA====");
                System.out.println("Pilih data yang akan digunakan:");
                System.out.println("1. Weather - Nominal");
                System.out.println("2. Weather - Kontinu");
                System.out.println("3. Iris");
                System.out.print("Nomor data : ");
                int idData = scan.nextInt();
                if(idData == 1) {
                    file = "data/weather.nominal.arff";
                    testfile = "data/weather.nominal.test.arff";
                }
                else if(idData == 2) {
                    file = "data/weather.numeric.arff";
                    testfile = "data/weather.numeric.test.arff";
                }
                else if(idData == 3){
                    file = "data/iris.arff";
                    testfile = "data/iris.test.arff";
                }
                w.loadFile(file); 
            }else if (option == 2){
                System.out.println("====RESAMPLE====");
                System.out.println("-B 0 = distribution in input data -- 1 = uniform distribution.");
                System.out.print("Masukan nilai B : ");
                int bias = scan.nextInt();
                System.out.println("-S Specify the random number seed (default 1)");
                System.out.print("Masukan nilai S : ");
                int seed = scan.nextInt();
                System.out.println("-Z The size of the output dataset, as a percentage of\n" +
                "  the input dataset (default 100)");
                System.out.println("Masukan nilai Z : ");
                int Z = scan.nextInt();
                w.resample(bias, Z, seed);
            }else if (option == 3){
                //remove attribute
                System.out.println("menghapus atribut? (Y/N)");
                String remove = scan.next();
                
                if(remove.equalsIgnoreCase("Y")){
                    int idx[] = new int[1];
                    System.out.print("Index atribut yang akan dihapus: ");
                    idx[0] = scan.nextInt() - 1;
                    w.removeAttribute(idx);
                }
            } else if(option == 4) {
                System.out.println("====Build Classifier====");
                //create model
                System.out.println("Classifier yang akan digunakan:");
                System.out.println("1. Agnes");
                System.out.println("2. K-Means");
                System.out.print("Masukan pilihan : ");
                clusterer = scan.nextInt();
                if(clusterer == 1) {
                    System.out.println("Type : ");
                    System.out.println("1. Single");
                    System.out.println("2. Complete");
                    System.out.print("Masukan pilihan : ");
                    int type = scan.nextInt();
                    w.setAgnesType(type - 1);
                }
                System.out.print("Masukan banyak cluster yang diinginkan: ");
                int nCluster = scan.nextInt();
                w.setNumCluster(nCluster);
                w.setClusterer(clusterer - 1);
            }else if(option == 5) {
                System.out.print("Masukan nilai percentage split : ");
                double p = scan.nextDouble();
                w.percentageSplit(p);
                //w.percentageSplit(model, p);
            }else if(option == 6) {
                System.out.println("Ingin menyimpan model? (Y/N)");
                String savemodel = scan.next();
                if(savemodel.equalsIgnoreCase("Y")){
                    System.out.print("Nama file: ");
                    String modelname = scan.next();
                    modelname = "model/" + modelname + ".model";
                    w.saveModel(modelname);  
                }
            }else if(option == 7) {
                System.out.print("Nama file yang akan di Load: ");
                String loadmodel = scan.next();
                loadmodel = "model/"+loadmodel;
                model = w.loadModel(loadmodel);
            }else if(option == 8) {
                System.out.println("PREDICTION");
                w.loadFile(testfile);
            }else {
                stat = false;
                System.out.println("====TERIMAKASIH :D====");
            }
        }
    }
}
