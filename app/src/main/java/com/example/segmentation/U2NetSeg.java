package com.example.segmentation;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

import androidx.core.graphics.ColorUtils;

import com.example.segmentation.env.Utils;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class U2NetSeg {
    private static final int NUM_THREADS = 4;
    private static boolean isGPU = true;

    private GpuDelegate gpuDelegate = null;
    private GpuDelegate.Options delegateOptions;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;
    private Interpreter tfLite;
    private List<String> labels = new ArrayList<>();
    public static float inferenceTime;
    private int imageWidth;
    private int imageHeight;
    public static Pair<Bitmap,Bitmap> result;
    float[][][][] input;

    float[][][][] output1;
    float[][][][] output2;
    float[][][][] output3;
    float[][][][] output4;
    float[][][][] output5;
    float[][][][] output6;
    float[][][][] output7;


    public U2NetSeg(final AssetManager assetManager, final String modelFilename) {


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

        output1 = new float[1][1][imageWidth][imageHeight];
        output2 = new float[1][1][imageWidth][imageHeight];
        output3 = new float[1][1][imageWidth][imageHeight];
        output4 = new float[1][1][imageWidth][imageHeight];
        output5 = new float[1][1][imageWidth][imageHeight];
        output6 = new float[1][1][imageWidth][imageHeight];
        output7 = new float[1][1][imageWidth][imageHeight];

    }

    public Bitmap recognizeImage(Bitmap bitmap) {
//        float aspect_ratio = (float) bitmap.getHeight()/bitmap.getWidth();
//        long startTimeForLoadImage = SystemClock.uptimeMillis();
//        Bitmap loadedBitmap = Utils.scaleBitmapAndKeepRatio(bitmap, imageHeight, imageWidth);
        Bitmap loadedBitmap = Bitmap.createScaledBitmap(bitmap,imageWidth,imageHeight,true);
//        background = Utils.scaleBitmapAndKeepRatio(background, bitmap.getHeight(), bitmap.getWidth());
//        Bitmap resultBWBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
//        Bitmap resultGradientBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);


        input = Utils.bitmapToFloatArray(loadedBitmap);


        Map<Integer,Object> outputs = new HashMap<>();
        outputs.put(0, output1);
        outputs.put(1, output2);
        outputs.put(2, output3);
        outputs.put(3, output4);
        outputs.put(4, output5);
        outputs.put(5, output6);
        outputs.put(6, output7);

        Object[] inputArray = {input};
        long startTimeForLoadImage = SystemClock.uptimeMillis();
        tfLite.runForMultipleInputsOutputs(inputArray, outputs);
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        int[] intValues = new int[imageHeight*imageWidth];
        loadedBitmap.getPixels(intValues,0,loadedBitmap.getWidth(),0,0,loadedBitmap.getWidth(),loadedBitmap.getHeight());
//        result = Utils.convertArrayToBitmap(output1, imageWidth, imageHeight,intValues);
        inferenceTime = endTimeForLoadImage-startTimeForLoadImage;

//        int[] mask = new int[imageHeight*imageWidth];
//        result.first.getPixels(mask,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
//        Bitmap resizedBWMask = Bitmap.createScaledBitmap(maskImage.first, bitmap.getWidth(), bitmap.getHeight(), false);
//        Bitmap resizedGradientMask = Bitmap.createScaledBitmap(maskImage.second, bitmap.getWidth(), bitmap.getHeight(),false);


//        for (int x=0; x<resizedBWMask.getWidth();x++) {
//            for (int y=0;y<resizedBWMask.getHeight();y++){
//                if (Color.red(resizedBWMask.getPixel(x,y))>200) {
//                    resultBWBitmap.setPixel(x, y, bitmap.getPixel(x, y));
//                }

//                int color = bitmap.getPixel(x, y);
//                int maskColor = resizedGradientMask.getPixel(x,y);
//                int bgColor = background.getPixel(x,y);
//                float prob = Color.red(maskColor)/255.0f;

//                resultGradientBitmap.setPixel(x,y,Color.rgb((int)(prob*Color.red(color)+(1-prob)*Color.red(bgColor)),
//                        (int)(prob*Color.green(color)+(1-prob)*Color.green(bgColor)),
//                        (int)(prob*Color.blue(color)+(1-prob)*Color.blue(bgColor))));
//            }
//        }



//        Bitmap newBitmap = Bitmap.createBitmap(imageWidth*imageHeight, 1, Bitmap.Config.ARGB_8888);
//        newBitmap.setPixels(mask, 0, newBitmap.getWidth(), 0, 0, newBitmap.getWidth(), newBitmap.getHeight());

        return result.first;
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
