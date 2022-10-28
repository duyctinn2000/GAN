package com.example.segmentation;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.SystemClock;

import com.example.segmentation.env.Utils;

import org.checkerframework.checker.units.qual.C;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;


public class Keypoints {
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
    static final float CONFIDENT_THRESH = 0.01f;
    float[][] results,center,scale;
    float[][][][] output,tf_output;
    private static final int NUM_KEYPOINTS=20;
    private final float IMAGE_MEAN=0.0f;
    private final float IMAGE_STD=1.0f;
    private TensorImage inputImageBuffer;


    public Keypoints(final AssetManager assetManager, final String modelFilename) {


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
        DataType imageDataType = tfLite.getInputTensor(0).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);

        output = new float[1][NUM_KEYPOINTS][40][40];
        tf_output = new float[1][24][24][NUM_KEYPOINTS];
        center = new float[1][2];
        scale = new float[1][2];

    }

    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        // TODO: Define an ImageProcessor from TFLite Support Library to do preprocessing
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(imageWidth, imageHeight, ResizeOp.ResizeMethod.BILINEAR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    public Bitmap recognizeImage(Bitmap bitmap) {
//        float aspect_ratio = (float) bitmap.getHeight()/bitmap.getWidth();
//        long startTimeForLoadImage = SystemClock.uptimeMillis();
        Bitmap loadedBitmap = Utils.scaleBitmapAndKeepRatio(bitmap, imageHeight, imageWidth);
//        Bitmap resultBWBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);

        input = Utils.bitmapToFloatArray(loadedBitmap);


        results = new float[NUM_KEYPOINTS][3];

        for (int i=0; i<NUM_KEYPOINTS; i++) {
            results[i][2] = -1;
        }
        long startTimeForLoadImage = SystemClock.uptimeMillis();
        tfLite.run(input, output);
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        inferenceTime = endTimeForLoadImage-startTimeForLoadImage;

//        for (int i=0; i<24; i++) {
//            for (int j=0; j<24;j++) {
//                for (int k=0; k<NUM_KEYPOINTS; k++) {
//                    if (tf_output[0][i][j][k]>CONFIDENT_THRESH && tf_output[0][i][j][k]>results[k][2]) {
//                        results[k][0] = j;
//                        results[k][1] = i;
//                        results[k][2] = tf_output[0][i][j][k];
//                    }
//                }
//            }
//        }

//        center[0][0] = bitmap.getWidth() * 0.5f;
//        center[0][1] = bitmap.getHeight() * 0.5f;
//        scale[0][0] = bitmap.getWidth() /200.0f;
//        scale[0][0] = bitmap.getHeight() /200.0f;

        for (int i=0; i<NUM_KEYPOINTS; i++) {
            for (int j=0; j<40;j++) {
                for (int k=0; k<40; k++) {
                    if (output[0][i][j][k]>CONFIDENT_THRESH && output[0][i][j][k]>results[i][2]) {
                        results[i][0] = k;
                        results[i][1] = j;
                        results[i][2] = output[0][i][j][k];
                        if (1 < j && j < 39) {
                            float shift_h = output[0][i][j+1][k]-output[0][i][j-1][k];
                            if (shift_h>0) {
                                results[i][1] += shift_h* .25;
                            } else if (shift_h<0) {
                                results[i][1] -= shift_h* .25;
                            }
                        }
                        if (1 < k && k < 39) {
                            float shift_w = output[0][i][j][k+1]-output[0][i][j][k-1];
                            if (shift_w>0) {
                                results[i][0] += shift_w*.25;
                            } else if (shift_w<0) {
                                results[i][0] -= shift_w*.25;
                            }
                        }
                    }
                }
            }
        }



        Bitmap tempBitmap = bitmap.copy(bitmap.getConfig(),true);
        final Canvas canvas = new Canvas(tempBitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        final Paint linePaint = new Paint();
        linePaint.setColor(Color.BLUE);
        linePaint.setStrokeWidth(5);
        int [][] pairs = {{0, 1}, {0, 2}, {1, 3}, {0, 4}, {1, 4}, {4, 5}, {5, 7},{5, 8},{5, 9}, {8, 12}, {12,16},{9, 13}, {13,17}, {7, 6}, {6, 11}, {6, 10}, {10, 14}, {14, 18}, {11, 15}, {15, 19}};
        for (int i = 0; i<NUM_KEYPOINTS; i++) {
//            paint.setColor(Color.RED);
            if (results[i][2]>-1) {
                canvas.drawCircle(results[i][0] * bitmap.getWidth() / 40, results[i][1] * bitmap.getHeight() / 40, 10, paint);
            }
//            paint.setColor(Color.BLUE);
//            canvas.drawCircle(location.left+result[1][0], location.top+result[1][1] , 10, paint);
//            paint.setColor(Color.YELLOW);
//            canvas.drawCircle(location.left+result[2][0], location.top+result[2][1] , 10, paint);
//            paint.setColor(Color.CYAN);
//            canvas.drawCircle(location.left+result[3][0], location.top+result[3][1] , 10, paint);
//            paint.setColor(Color.GREEN);
//            canvas.drawCircle(location.left+result[4][0], location.top+result[4][1] , 10, paint);
            if (results[pairs[i][0]][2]>-1 && results[pairs[i][1]][2]>-1) {
                canvas.drawLine(results[pairs[i][0]][0]*bitmap.getWidth()/40,results[pairs[i][0]][1]*bitmap.getHeight()/40,results[pairs[i][1]][0]*bitmap.getWidth()/40,results[pairs[i][1]][1]*bitmap.getHeight()/40,linePaint);
            }
        }
//        for (int i = 0; i<20; i++)





        return tempBitmap;
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
