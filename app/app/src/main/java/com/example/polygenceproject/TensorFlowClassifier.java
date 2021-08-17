/*
 *    Copyright (C) 2017 MINDORKS NEXTGEN PRIVATE LIMITED
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.example.polygenceproject;

import android.content.res.AssetManager;
import android.util.Log;

import androidx.core.os.TraceCompat;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

/**
 * Created by amitshekhar on 16/03/17.
 */

/**
 * A classifier specialized to label images using TensorFlow.
 */
public class TensorFlowClassifier implements Classifier {

    private static final String TAG = "TFImageClassifier";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.1f;

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private float[] outputs;
    private String[] outputNames;

    private TensorFlowInferenceInterface inferenceInterface;

    private boolean runStats = false;

    private TensorFlowClassifier() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize     The input size. A square image of inputSize x inputSize is assumed.
     * @param inputName     The label of the image input node.
     * @param outputName    The label of the output node.
     * @throws IOException
     */
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int inputSize,
            String inputName,
            String outputName)
            throws IOException {
        TensorFlowClassifier c = new TensorFlowClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        // Read the label names into memory.
        // TODO(andrewharp): make this handle non-assets.
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        Log.i(TAG, "Reading labels from: " + actualFilename);
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
        String line;
        while ((line = br.readLine()) != null) {
            c.labels.add(line);
        }
        br.close();

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);


        Log.i(TAG, "Graph:" + c.inferenceInterface.graph());
        Log.i(TAG, "Operation:" + c.inferenceInterface.graph().operation(outputName));
        Log.i(TAG, "Output:" +  c.inferenceInterface.graph().operation(outputName).output(0));
        Log.i(TAG, "Shape:" + c.inferenceInterface.graph().operation(outputName).output(0).shape());
        Log.i(TAG, "Size:" + c.inferenceInterface.graph().operation(outputName).output(0).shape().size(1));

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        int numClasses = (int) c.inferenceInterface.graph().operation(outputName).output(0).shape().size(1);
        Log.i(TAG, "Jreed " + c.labels.size() + " labels, output layer size is " + numClasses);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputSize = inputSize;

        // Pre-allocate buffers.
        c.outputNames = new String[]{outputName};
        c.outputs = new float[numClasses];

        return c;
    }

    /*
    public float[] predictPlay() {
        int[] data = {2, 584, 10, 2, 82, 1};
        TraceCompat.beginSection("feed");
        inferenceInterface.feed(inputName, data, new long[]{inputSize});
        TraceCompat.endSection();

        TraceCompat.beginSection("run");
        inferenceInterface.run(outputNames, runStats);
        TraceCompat.endSection();

        TraceCompat.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        TraceCompat.endSection();

        return outputs;
    }
    */

    @Override
    public List<Recognition> recognizeImage(final float[] pixels) {
        // Log this method so that it can be analyzed with systrace.
        TraceCompat.beginSection("recognizeImage");

        // Copy the input data into TensorFlow.
        TraceCompat.beginSection("feed");
        //inferenceInterface.feed(inputName, pixels,inputSize);
        inferenceInterface.feed(inputName, pixels, new long[]{inputSize});
        TraceCompat.endSection();

        // Run the inference call.
        TraceCompat.beginSection("run");
        Log.i(TAG, "Output Names: " + outputNames);
        Log.i(TAG, "Run Stats: " + runStats);
        inferenceInterface.run(outputNames, runStats);
        TraceCompat.endSection();

        // Copy the output Tensor back into the output array.
        TraceCompat.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        TraceCompat.endSection();

        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        for (int i = 0; i < outputs.length; ++i) {
            if (outputs[i] > THRESHOLD) {
                pq.add(
                        new Recognition(
                                "" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
            }
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        TraceCompat.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public float[] predictPlay(float[] info) {
        TraceCompat.beginSection("predictPlay");

        TraceCompat.beginSection("feed");
        inferenceInterface.feed(inputName, info, 1, inputSize);
        //inferenceInterface.feed("keep_prob", new float[] { 1 }); // probability the play is a success
        TraceCompat.endSection();

        TraceCompat.beginSection("run");
        inferenceInterface.run(outputNames);
        TraceCompat.endSection();

        TraceCompat.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        TraceCompat.endSection();

        return outputs;
    }

    @Override
    public void enableStatLogging(boolean debug) {
        runStats = debug;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}


