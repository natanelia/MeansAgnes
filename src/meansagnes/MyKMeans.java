/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package meansagnes;

import java.util.ArrayList;
import java.util.Random;
import weka.clusterers.RandomizableClusterer;
import weka.core.Attribute;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import static weka.core.pmml.PMMLUtils.pad;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 *
 * @author Natan
 */
public class MyKMeans extends RandomizableClusterer {

    static final long serialVersionUID = -3235809600124455376L;
    private ReplaceMissingValues replaceMissingFilter;

    private int numCluster = 2;
    private Instances clusterCentroids;
    private int[] clusterAssignments;
    private Instances[] clusteredInstances;

    private int maxIterations = 500;
    private int currentIteration = 0;
    protected DistanceFunction distanceFunction = new EuclideanDistance();

    public MyKMeans() {
        super();

        m_SeedDefault = 10;
        setSeed(m_SeedDefault);
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        currentIteration = 0;
        replaceMissingFilter = new ReplaceMissingValues();
        Instances instances = new Instances(data);

        instances.setClassIndex(-1);
        replaceMissingFilter.setInputFormat(instances);
        instances = Filter.useFilter(instances, replaceMissingFilter);
        distanceFunction.setInstances(instances);

        clusterCentroids = new Instances(instances, numCluster);
        clusterAssignments = new int[instances.numInstances()];

        // assign a number of instance become a centroid randomly
        Random randomizer = new Random(getSeed());
        int[] instanceAsCentroid = new int[numCluster];
        for (int i = 0; i < numCluster; i++) {
            instanceAsCentroid[i] = -1;
        }
        for (int i = 0; i < numCluster; i++) {
            int centroidCluster = randomizer.nextInt(instances.numInstances());
            boolean found = false;

            for (int j = 0; j < i /* instanceAsCentroid.length */ && !found; j++) {
                if (instanceAsCentroid[j] == centroidCluster) {
                    i--;
                    found = true;
                }
            }

            if (!found) {
                clusterCentroids.add(instances.instance(centroidCluster));
                instanceAsCentroid[i] = centroidCluster;
            }
        }

        double[][] distancesToCentroid = new double[numCluster][instances.numInstances()];
        double[] minDistancesToCentroid = new double[instances.numInstances()];
        boolean converged = false;
        Instances prevCentroids;
        while (!converged) {
            currentIteration++;
            // check distance to each centroid to decide clustering result
            for (int i = 0; i < numCluster; i++) { // i is cluster index
                for (int j = 0; j < instances.numInstances(); j++) { // j is instance index
                    distancesToCentroid[i][j] = distanceFunction.distance(clusterCentroids.instance(i), instances.instance(j));
                }
            }
            for (int j = 0; j < instances.numInstances(); j++) { // j is instance index
                minDistancesToCentroid[j] = distancesToCentroid[0][j];
                clusterAssignments[j] = 0;
            }
            for (int j = 0; j < instances.numInstances(); j++) { // j is instance index
                for (int i = 1; i < numCluster; i++) { // i is cluster index
                    if (minDistancesToCentroid[j] > distancesToCentroid[i][j]) {
                        minDistancesToCentroid[j] = distancesToCentroid[i][j];
                        clusterAssignments[j] = i;
                    }
                }
            }
            
            for (int i = 0; i < numCluster; i++) {
                System.out.println(clusterCentroids.instance(i));
            }
            // update centroids
            prevCentroids = clusterCentroids;
            clusterCentroids = new Instances(instances, numCluster);
            clusteredInstances = new Instances[numCluster];
            for (int i = 0; i < numCluster; i++) {
                clusteredInstances[i] = new Instances(instances, 0);
            }

            for (int i = 0; i < instances.numInstances(); i++) {
                clusteredInstances[clusterAssignments[i]].add(instances.instance(i));
                System.out.println(instances.instance(i).toString() + " : " + clusterAssignments[i]);
            }

            if (currentIteration == maxIterations) {
                converged = true;
            }

            Instances newCentroids = new Instances(instances, numCluster);
            for (int i = 0; i < numCluster; i++) {
                newCentroids.add(moveCentroid(clusteredInstances[i]));
            }
            clusterCentroids = newCentroids;
            
            boolean centroidChanged = false;
            for (int i = 0; i < numCluster; i++) {
                if (distanceFunction.distance(prevCentroids.instance(i), clusterCentroids.instance(i)) > 0) {
                    centroidChanged = true;
                }
            }
            if (!centroidChanged) {
                converged = true;
            }
            System.out.println("\n\n");
        }
    }

    protected Instance moveCentroid(Instances instances) {
        double[] vals = new double[instances.numAttributes()];
        for (int k = 0; k < instances.numAttributes(); k++) {
            vals[k] = instances.meanOrMode(k);
        }
        return new Instance(1.0, vals);
    }

    @Override
    public int numberOfClusters() throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    private String pad(String source, String padChar, int length, boolean leftPad) {
        StringBuffer temp = new StringBuffer();

        if (leftPad) {
            for (int i = 0; i < length; i++) {
                temp.append(padChar);
            }
            temp.append(source);
        } else {
            temp.append(source);
            for (int i = 0; i < length; i++) {
                temp.append(padChar);
            }
        }
        return temp.toString();
    }

    @Override
    public String toString() {
        if (clusterCentroids == null) {
            return "No clusterer built yet!";
        }

        int maxWidth = 0;
        int maxAttWidth = 0;
        boolean containsNumeric = false;
        for (int i = 0; i < numCluster; i++) {
            for (int j = 0; j < clusterCentroids.numAttributes(); j++) {
                if (clusterCentroids.attribute(j).name().length() > maxAttWidth) {
                    maxAttWidth = clusterCentroids.attribute(j).name().length();
                }
                if (clusterCentroids.attribute(j).isNumeric()) {
                    containsNumeric = true;
                    double width = Math.log(Math.abs(clusterCentroids.instance(i)
                            .value(j))) / Math.log(10.0);
                    // System.err.println(clusterCentroids.instance(i).value(j)+" "+width);
                    if (width < 0) {
                        width = 1;
                    }
                    // decimal + # decimal places + 1
                    width += 6.0;
                    if ((int) width > maxWidth) {
                        maxWidth = (int) width;
                    }
                }
            }
        }

        for (int i = 0; i < clusterCentroids.numAttributes(); i++) {
            if (clusterCentroids.attribute(i).isNominal()) {
                Attribute a = clusterCentroids.attribute(i);
                for (int j = 0; j < clusterCentroids.numInstances(); j++) {
                    String val = a.value((int) clusterCentroids.instance(j).value(i));
                    if (val.length() > maxWidth) {
                        maxWidth = val.length();
                    }
                }
                for (int j = 0; j < a.numValues(); j++) {
                    String val = a.value(j) + " ";
                    if (val.length() > maxAttWidth) {
                        maxAttWidth = val.length();
                    }
                }
            }
        }

        StringBuffer temp = new StringBuffer();
        // String naString = "N/A";

        /*
     * for (int i = 0; i < maxWidth+2; i++) { naString += " "; }
         */
        temp.append("\nkMeans\n======\n");
        temp.append("\nNumber of iterations: " + currentIteration + "\n");

        temp.append("\n\nCluster centroids:\n");
        temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2))
                - "Cluster#".length(), true));

        temp.append("\n");
        temp
                .append(pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));

        // cluster numbers
        for (int i = 0; i < numCluster; i++) {
            String clustNum = "" + i;
            temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
        }
        temp.append("\n");

        // cluster sizes
//        String cSize = "(" + Utils.sum(m_ClusterSizes) + ")";
//        temp.append(pad(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(),
//                true));
//        for (int i = 0; i < numCluster; i++) {
//            cSize = "(" + m_ClusterSizes[i] + ")";
//            temp.append(pad(cSize, " ", maxWidth + 1 - cSize.length(), true));
//        }
//        temp.append("\n");
        temp.append(pad("", "=",
                maxAttWidth
                + (maxWidth * (clusterCentroids.numInstances() + 1)
                + clusterCentroids.numInstances() + 1), true));
        temp.append("\n");

        for (int i = 0; i < clusterCentroids.numAttributes(); i++) {
            String attName = clusterCentroids.attribute(i).name();
            temp.append(attName);
            for (int j = 0; j < maxAttWidth - attName.length(); j++) {
                temp.append(" ");
            }

            String strVal;
            String valMeanMode;
            // full data
//            if (clusterCentroids.attribute(i).isNominal()) {
//                if (m_FullMeansOrMediansOrModes[i] == -1) { // missing
//                    valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(),
//                            true);
//                } else {
//                    valMeanMode = pad(
//                            (strVal = clusterCentroids.attribute(i).value(
//                            (int) m_FullMeansOrMediansOrModes[i])), " ", maxWidth + 1
//                            - strVal.length(), true);
//                }
//            } else if (Double.isNaN(m_FullMeansOrMediansOrModes[i])) {
//                valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(),
//                        true);
//            } else {
//                valMeanMode = pad(
//                        (strVal = Utils.doubleToString(m_FullMeansOrMediansOrModes[i],
//                                maxWidth, 4).trim()), " ", maxWidth + 1 - strVal.length(), true);
//            }
//            temp.append(valMeanMode);

            for (int j = 0; j < numCluster; j++) {
                if (clusterCentroids.attribute(i).isNominal()) {
                    if (clusterCentroids.instance(j).isMissing(i)) {
                        valMeanMode = pad("missing", " ",
                                maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode = pad(
                                (strVal = clusterCentroids.attribute(i).value(
                                (int) clusterCentroids.instance(j).value(i))), " ", maxWidth
                                + 1 - strVal.length(), true);
                    }
                } else if (clusterCentroids.instance(j).isMissing(i)) {
                    valMeanMode = pad("missing", " ",
                            maxWidth + 1 - "missing".length(), true);
                } else {
                    valMeanMode = pad(
                            (strVal = Utils.doubleToString(
                                    clusterCentroids.instance(j).value(i), maxWidth, 4).trim()),
                            " ", maxWidth + 1 - strVal.length(), true);
                }
                temp.append(valMeanMode);
            }
            temp.append("\n");
        }

        temp.append("\n\n");
        return temp.toString();
    }
}
