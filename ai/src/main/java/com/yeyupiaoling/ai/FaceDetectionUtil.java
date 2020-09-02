package com.yeyupiaoling.ai;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;

public class FaceDetectionUtil {
    private static final String TAG = FaceDetectionUtil.class.getName();
    public static final int OK = 0;
    public static final int MASK = 1001;
    public static final int SIDE_FACE = 1002;
    public static final int NO_FACE = 1003;
    public static final int MUCH_FACE = 1004;
    // 缩放大小
    private static final int maxSize = 700;
    // 侧脸的最大限制倍数
    private static final int distDiff = 3;
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
    private Bitmap resultBitmap;

    PaddleNative predictor = new PaddleNative();

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

    public int predictImage(String image_path) throws Exception {
        if (!new File(image_path).exists()) {
            throw new Exception("image file is not exists!");
        }
        FileInputStream fis = new FileInputStream(image_path);
        Bitmap bitmap = BitmapFactory.decodeStream(fis);
        return predictImage(bitmap);
    }

    public int predictImage(Bitmap bitmap) throws Exception {
        return predict(bitmap);
    }


    // 执行预测
    private int predict(Bitmap bmp) throws Exception {
        resultBitmap = null;
//        Bitmap predictBitmap = Utils.cropCenterImage(bmp);
        Bitmap predictBitmap = Utils.getScaleBitmap(bmp, maxSize);
        recycle(bmp);
        long start = System.currentTimeMillis();
        Face[] faces = predictor.process(predictBitmap);
        long end = System.currentTimeMillis();
        Log.d(TAG, "单纯预测时间：" + (end - start));

        if (faces == null || faces.length == 0) {
            return NO_FACE;
        } else if (faces.length > 140) {
            return MUCH_FACE;
        } else {
            if (isMask(faces[0])) {
                return MASK;
            } else {
                if (isSideFace(faces[0])) {
                    return SIDE_FACE;
                } else {
                    resultBitmap = Utils.drawBitmap(predictBitmap, faces);
                    return OK;
                }
            }
        }
    }

    // 判断是否侧脸
    private boolean isSideFace(Face face) {
//        float left = face.roi[0];
//        float top = face.roi[1];
//        float right = face.roi[2] + face.roi[0];
//        float bottom = face.roi[3] + face.roi[1];
//        for (int i = 0; i < face.keypoints.length; i = i + 2) {
//            float x = face.keypoints[i];
//            float y = face.keypoints[i + 1];
//            if (x < left || x > right || y < top || y > bottom) {
//                Log.d(TAG, "侧脸1");
//                return true;
//            }
//        }
//
//        float mouth1 = face.keypoints[94];
//        float mouth2 = face.keypoints[96];
//        float face1 = face.keypoints[6];
//        float face2 = face.keypoints[22];
//        float dist1 = mouth1 - face1;
//        float dist2 = face2 - mouth2;
//
//        if (dist1 * distDiff < dist2 || dist2 * distDiff < dist1) {
//            Log.d(TAG, "侧脸2");
//            return true;
//        }
        return false;
    }

    private boolean isMask(Face face) {
        return false;
//        int label = (int) face[4];
//        return label == 1;
    }

    public Bitmap getBitmap() {
        return resultBitmap;
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
