package com.example.segmentation;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

public class DominantColors {

    static {
        System.loadLibrary("dominantcolors");
    }

    U2NetSeg u2NetSeg;
    public static float inferenceTime;


    public DominantColors(AssetManager assetManager){
        u2NetSeg = new U2NetSeg(assetManager,"u2net_fp16.tflite");
    };

    public native static int[] kmeans(Bitmap bmp, int numColors);

    public int[] recognizeImage(Bitmap bitmap) {
//        Pair<Bitmap,Bitmap> seg = u2NetSeg.recognizeImage(bitmap,null);
//        long startTimeForLoadImage = SystemClock.uptimeMillis();
//        Bitmap resizeBitmap = seg.second;
//        if (seg.second.getWidth()>1000) {
//            resizeBitmap = Bitmap.createScaledBitmap(seg.second, 1000, 1, true);
//        }
//        int[] result = kmeans(resizeBitmap,3);
//        long endTimeForLoadImage = SystemClock.uptimeMillis();
//        inferenceTime = endTimeForLoadImage-startTimeForLoadImage;
//        return result;
        return new int[]{1};
    }

}
