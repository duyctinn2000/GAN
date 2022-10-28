package com.example.segmentation.env;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.util.Log;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class Utils {

    /**
     * Memory-map the model file in Assets.
     */
    public static float minValue;
    public static float maxValue;
    public static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
//            return bitmap.copy(Bitmap.Config.ARGB_8888,true);
        } catch (IOException e) {
            // handle exception
            Log.e("getBitmapFromAsset", "getBitmapFromAsset: " + e.getMessage());
        }

        return bitmap;
    }

    public static Bitmap scaleBitmapAndKeepRatio(Bitmap targetBmp, int reqHeightInPixels, int reqWidthInPixels) {
        if (targetBmp.getHeight() == reqHeightInPixels && targetBmp.getWidth() == reqWidthInPixels) {
            return targetBmp;
        } else {
            Matrix matrix = new Matrix();
            matrix.setRectToRect(new RectF(0.0F, 0.0F, (float)targetBmp.getWidth(), (float)targetBmp.getHeight()), new RectF(0.0F, 0.0F, (float)reqWidthInPixels, (float)reqHeightInPixels), Matrix.ScaleToFit.FILL);
            Bitmap scaledBitmap = Bitmap.createBitmap(targetBmp, 0, 0, targetBmp.getWidth(), targetBmp.getHeight(), matrix, true);
            return scaledBitmap;
        }
    }

    public static Bitmap convertArrayToBitmap(float[][][][] imageArray, int imageWidth, int imageHeight) {

        // Convert multidimensional array to 1D
        float maxValue = 0;
        float minValue = 10000f;

        for (int m=0; m < imageArray[0].length; m++) {
            for (int x=0; x < imageArray[0][0].length; x++) {
                for (int y=0; y < imageArray[0][0][0].length; y++) {
                    if (maxValue<imageArray[0][m][x][y]) {
                        maxValue = imageArray[0][m][x][y];
                    }
                    if (minValue>imageArray[0][m][x][y]) {
                        minValue = imageArray[0][m][x][y];
                    }
                }
            }
        }
        Log.i("min",""+ minValue);
        Log.i("max",""+ maxValue);


        Bitmap blackWhiteImage = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
        Bitmap gradientImage = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);

        for (int x = 0; x < 256; x++) {
            for (int y = 0;  y < 256; y++) {

                // Create black and transparent bitmap based on pixel value above a certain number eg. 150
                // make all pixels black in case value of grayscale image is above 150
                float r = (float) (255.0f * ((imageArray[0][0][x][y] + 1) / 2));
                float g = (float) (255.0f * ((imageArray[0][1][x][y] + 1) / 2));
                float b = (float) (255.0f * ((imageArray[0][2][x][y] + 1) / 2));
//                if (pixel>70) {
//                    blackWhiteImage.setPixel(y,x,Color.WHITE);
//                } else {
//                    blackWhiteImage.setPixel(y,x,Color.BLACK);
//                }

                gradientImage.setPixel(y,x,Color.rgb((int)r,(int)g,(int)b));

            }
        }
        return gradientImage;
    }



    public static float[][][][] bitmapToFloatArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] intValues = new int[width*height];
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);
        float[][][][] fourDimensionalArray = new float[1][3][width][height];

        for (int i=0; i < width ; i++) {
            for (int j=0; j < height ; j++) {
                int pixelValue = intValues[i*width+j];
                fourDimensionalArray[0][0][i][j] = (float)
                        Color.red(pixelValue);
                fourDimensionalArray[0][1][i][j] = (float)
                        Color.green(pixelValue);
                fourDimensionalArray[0][2][i][j] = (float)
                        Color.blue(pixelValue);
            }
        }
//
//
//        // Convert multidimensional array to 1D
        float maxValue = 0.0f;
        for (int m=0; m < fourDimensionalArray[0].length; m++) {
            for (int x=0; x < fourDimensionalArray[0][0].length; x++) {
                for (int y=0; y < fourDimensionalArray[0][0][0].length; y++) {
                    if (maxValue<fourDimensionalArray[0][m][x][y]) {
                        maxValue = fourDimensionalArray[0][m][x][y];
                    }
                }
            }
        }
//
        Log.i("max",""+ maxValue);

        float [][][][] finalFourDimensionalArray = new float[1][3][width][height];

//        for (int i = 0; i<width; ++i) {
//            for (int j = 0; j<height; ++j) {
//                int pixelValue = intValues[i * width + j];
//                finalFourDimensionalArray[0][0][i][j] =
//                        ((Color.red(pixelValue) / maxValue) - 0.46962251f) / 0.27469736f;
//                finalFourDimensionalArray[0][1][i][j] =
//                        ((Color.green(pixelValue) / maxValue) - 0.4464104f) / 0.27012361f;
//                finalFourDimensionalArray[0][2][i][j] =
//                        ((Color.blue(pixelValue) / maxValue) - 0.40718787f) / 0.28515933f;
//            }
//
//        }

        for (int i = 0; i<width; i++) {
            for (int j = 0; j<height; j++) {
                int pixelValue = intValues[i * width + j];
                finalFourDimensionalArray[0][0][i][j] =
                        ((Color.red(pixelValue) / 255.0f) - 0.5f) / 0.5f;
                finalFourDimensionalArray[0][1][i][j] =
                        ((Color.green(pixelValue) / 255.0f) - 0.5f) / 0.5f;
                finalFourDimensionalArray[0][2][i][j] =
                        ((Color.blue(pixelValue) / 255.0f) - 0.5f) / 0.5f;
            }
        }


//        for (int i = 0; i<width; ++i) {
//            for (int j = 0; j<height; ++j) {
//                int pixelValue = intValues[i * width + j];
//                finalFourDimensionalArray[0][0][i][j] =
//                        ((Color.red(pixelValue) - 0.485f)/255.0f);
//                finalFourDimensionalArray[0][1][i][j] =
//                        ((Color.green(pixelValue)- 0.456f)/255.0f)  ;
//                finalFourDimensionalArray[0][2][i][j] =
//                        ((Color.blue(pixelValue)- 0.406f)/255.0f) ;
//            }
//        }
        return finalFourDimensionalArray;
    }






}
