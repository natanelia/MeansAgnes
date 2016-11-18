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
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
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
    private Classifier model;
    private int classifier;
    
    public MeansAgnes(){
        data = null;
        model = null;
        classifier = 0;
    }
    
    public void setClassifier(int i){
        classifier = i;
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
    
    public void buildClassifier(int type, Instances train){
        //Classifier model = null;
        switch (type) {
            case 0:
                model = new MyAgnes();
                break;
            case 1 :
                break;
            default:
                break;
        }
        try {
            model.buildClassifier(train);
            System.out.println(model.toString());
            //return model;
        } catch (Exception ex) {
            Logger.getLogger(MeansAgnes.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    //10-fold cross validation
    public void crossValidation(){
        try {
            buildClassifier(classifier, data);
            Classifier m = model;
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(m, data, 10, new Random(1));
            System.out.println("10 FOLD CROSS VALIDATION\n\n");
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
        } catch (Exception ex) {
            System.out.println("10-Fold Cross Validation gagal");
        }
    }
    
    //percentage split
    public void percentageSplit(double percent){
        try {
            //Classifier model =null;
            data.randomize(new java.util.Random(0));
            int trainSize = (int) Math.round((double) data.numInstances() * percent/100f);
            int testSize = data.numInstances() - trainSize;
            
            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);
            
            /*for(int i=0; i<trainSize; i++){
                train.add(data.instance(i));
            }
            for(int i=trainSize; i<data.numInstances(); i++){
                test.add(data.instance(i));
            }*/
            buildClassifier(classifier, train);
            Classifier m = model;
            //tree.buildClassifier(train);
            //Classifier model = tree;
            //model.buildClassifier(train);
                    
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(m, test);
            System.out.println("PERCENTAGE SPLIT\n\n");
            
            System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            
        } catch (Exception ex) {
            System.out.println("Gagal");
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
    public Classifier loadModel(String modeladdress){
        Classifier model = null;
        try {
            model  = (Classifier) SerializationHelper.read(modeladdress);
            System.out.println(model.toString());
            System.out.println("Berhasil Load Model\n");
        } catch (Exception ex) {
            System.out.println("tidak bisa diload\n");
        }
        return model;
    }
    
    public void classify(String data_address){
        try {
            buildClassifier(classifier, data);
            Classifier m = model;
            Instances test = ConverterUtils.DataSource.read(data_address);
            System.out.println(test.toString());
            test.setClassIndex(test.numAttributes()-1);
            System.out.println("#Predictions on user test set#");
            for (int i = 0; i < test.numInstances(); i++) {
                double label = m.classifyInstance(test.instance(i));
                test.instance(i).setClassValue(label);
                System.out.println(test.instance(i)+"\n");
                
            }
        } catch (Exception ex) {
            System.out.println("GAGAL PREDIKSI\n");
        }
    }
   
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String file = "";
        int classifier;
        String testfile = "";
        MeansAgnes w = new MeansAgnes();
        Scanner scan = new Scanner(System.in);
        Classifier model = null;
        boolean stat = true;
        while(stat){
            System.out.println("\n\nProgram Eksplorasi Weka");
            System.out.println("1. Load data set");
            System.out.println("2. Filter : Resample");
            System.out.println("3. Remove Attribute");
            System.out.println("4. Build Classifier");
            System.out.println("5. 10 Fold Cross Validation");
            System.out.println("6. Percentage Split");
            System.out.println("7. Save model");
            System.out.println("8. Load model");
            System.out.println("9. Prediction");
            System.out.println("10. Exit");
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
                classifier = scan.nextInt();
                w.setClassifier(classifier - 1);
                //w.buildClassifier(pil-1);
                /*
                if(pil == 1){
                    model = w.buildClassifier(pil)
                }
                else if(pil == 2){
                    model = w.id3Classifier();
                }
                else if(pil == 3){
                    model = w.C45();
                }
                else if(pil == 4){
                    //belom di implementasi
                }else{
                    System.out.println("Maaf pilihan tidak tersedia");
                }*/
            }else if(option == 5) {
                //10-fold cross validation
                //w.tenFoldCrossValidation();
                w.crossValidation();
            }else if(option == 6) {
                System.out.print("Masukan nilai percentage split : ");
                double p = scan.nextDouble();
                w.percentageSplit(p);
                //w.percentageSplit(model, p);
            }else if(option == 7) {
                System.out.println("Ingin menyimpan model? (Y/N)");
                String savemodel = scan.next();
                if(savemodel.equalsIgnoreCase("Y")){
                    System.out.print("Nama file: ");
                    String modelname = scan.next();
                    modelname = "model/" + modelname + ".model";
                    w.saveModel(modelname);  
                }
            }else if(option == 8) {
                System.out.print("Nama file yang akan di Load: ");
                String loadmodel = scan.next();
                loadmodel = "model/"+loadmodel;
                model = w.loadModel(loadmodel);
            }else if(option == 9) {
                System.out.println("PREDICTION");
                w.classify(testfile);
            }else {
                stat = false;
                System.out.println("====TERIMAKASIH :D====");
            }
        }
    }
}
