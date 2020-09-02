package com.yeyupiaoling.ai;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;

public class FaceDetectionUtil {
    private static final String TAG = FaceDetectionUtil.class.getName();
    // 缩放大小
    private static final int maxSize = 700;
    private static final int NUM_THREADS = 4;
    private static final float FD_INPUT_SCALE = 0.25f;
    private static final float[] FD_INPUT_MEAN = new float[]{0.407843f, 0.694118f, 0.482353f};
    private static final float[] FD_INPUT_STD = new float[]{0.5f, 0.5f, 0.5f};
    private static final float FD_SCORE_THRESHOLD = 0.7f;
    private static final int[] FK_INPUT_SHAPE = new int[]{1, 3, 60, 60};
    private static final int[] MCL_INPUT_SHAPE = new int[]{1, 3, 128, 128};
    private static final float[] MCL_INPUT_MEAN = new float[]{0.5f, 0.5f, 0.5f};
    private static final float[] MCL_INPUT_STD = new float[]{1.0f, 1.0f, 1.0f};
    private static FaceDetectionUtil faceDetectionUtil;
    private Bitmap predictBitmap;

    private PaddleNative predictor = new PaddleNative();

    public static FaceDetectionUtil getInstance(Context context) throws Exception {
        if (faceDetectionUtil == null) {
            synchronized (FaceDetectionUtil.class) {
                if (faceDetectionUtil == null) {
                    faceDetectionUtil = new FaceDetectionUtil(context);
                }
            }
        }
        return faceDetectionUtil;
    }


    /**
     * @param context 应用上下文
     */
    public FaceDetectionUtil(Context context) {
        String pyramidboxModelPath = context.getCacheDir().getAbsolutePath() + File.separator + "pyramidbox.nb";
        Utils.copyFileFromAsset(context, "pyramidbox.nb", pyramidboxModelPath);
        String facekeypointsModelPath = context.getCacheDir().getAbsolutePath() + File.separator + "facekeypoints.nb";
        Utils.copyFileFromAsset(context, "facekeypoints.nb", facekeypointsModelPath);
        String maskClassifierModelPath = context.getCacheDir().getAbsolutePath() + File.separator + "maskclassifier.nb";
        Utils.copyFileFromAsset(context, "maskclassifier.nb", maskClassifierModelPath);

        boolean loadResult = predictor.init(
                pyramidboxModelPath,
                NUM_THREADS,
                "LITE_POWER_HIGH",
                FD_INPUT_SCALE,
                FD_INPUT_MEAN,
                FD_INPUT_STD,
                FD_SCORE_THRESHOLD,
                facekeypointsModelPath,
                NUM_THREADS,
                "LITE_POWER_HIGH",
                FK_INPUT_SHAPE[2],
                FK_INPUT_SHAPE[3],
                maskClassifierModelPath,
                NUM_THREADS,
                "LITE_POWER_HIGH",
                MCL_INPUT_SHAPE[2],
                MCL_INPUT_SHAPE[3],
                MCL_INPUT_MEAN,
                MCL_INPUT_STD);
        Log.e(TAG, "模型加载情况：" + loadResult);
    }

    public Face[] predictImage(String image_path) throws Exception {
        if (!new File(image_path).exists()) {
            throw new Exception("image file is not exists!");
        }
        FileInputStream fis = new FileInputStream(image_path);
        Bitmap bitmap = BitmapFactory.decodeStream(fis);
        return predictImage(bitmap);
    }

    public Face[] predictImage(Bitmap bitmap) throws Exception {
        return predict(bitmap);
    }


    // 执行预测
    private Face[] predict(Bitmap bmp) throws Exception {
        predictBitmap = Utils.getScaleBitmap(bmp, maxSize);
        recycle(bmp);
        long start = System.currentTimeMillis();
        Face[] faces = predictor.process(predictBitmap);
        long end = System.currentTimeMillis();
        Log.d(TAG, "单纯预测时间：" + (end - start));
        return faces;
    }

    public Bitmap getBitmap() {
        return predictBitmap;
    }

    // 销毁无用的Bitmap
    private void recycle(Bitmap bitmap) {
        if (!bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    public void release() {
        if (predictor != null) {
            predictor.release();
        }
    }
}
