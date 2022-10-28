package com.example.segmentation;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

import com.example.segmentation.env.Utils;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class BiSeNetV2 {
    private static final int NUM_THREADS = 4;
    private static boolean isGPU = false;

    private GpuDelegate gpuDelegate = null;
    private GpuDelegate.Options delegateOptions;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;
    private Interpreter tfLite;
    private List<String> labels = new ArrayList<>();
    public static float inferenceTime;
    private int imageWidth;
    private int imageHeight;
    float[][][][] input;


    float[][][][] output;


    public BiSeNetV2(final AssetManager assetManager, final String modelFilename) {


        try {
            Interpreter.Options options = (new Interpreter.Options());
            CompatibilityList compatList = new CompatibilityList();

            if (isGPU && compatList.isDelegateSupportedOnThisDevice()) {
                // if the device has a supported GPU, add the GPU delegate
                delegateOptions = compatList.getBestOptionsForThisDevice();
                gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
            } else {
                // if the GPU is not supported, run on 4 threads
                options.setNumThreads(NUM_THREADS);
            }

            tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            tfLite = new Interpreter(tfliteModel, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Reads type and shape of input and output tensors, respectively.

        int imageTensorIndex = 0;
        int[] imageShape = tfLite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageHeight = imageShape[2];
        imageWidth = imageShape[3];

        output = new float[1][3][imageWidth][imageHeight];

    }

    public Bitmap recognizeImage(Bitmap bitmap) {
//        float aspect_ratio = (float) bitmap.getHeight()/bitmap.getWidth();
//        long startTimeForLoadImage = SystemClock.uptimeMillis();
        Bitmap loadedBitmap = Utils.scaleBitmapAndKeepRatio(bitmap, imageHeight, imageWidth);
//        Bitmap resultBWBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Bitmap resultGradientBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);


        input = Utils.bitmapToFloatArray(loadedBitmap);



        long startTimeForLoadImage = SystemClock.uptimeMillis();

        tfLite.run(input, output);
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        inferenceTime = endTimeForLoadImage-startTimeForLoadImage;

        Log.i("AABCDD",String.valueOf(output[0][1][100][100]));
        return Utils.convertArrayToBitmap(output,256,256);

    }

    public void close() {
        if (tfLite != null) {
            // TODO: Close the interpreter
            tfLite.close();
            tfLite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        tfliteModel = null;
    }




}
