package com.yeyupiaoling.ai;

import android.graphics.Bitmap;

public class PaddleNative {
    static {
        System.loadLibrary("Native");
    }

    private long ctx = 0;

    public boolean init(String pyramidboxModelPath,
                        int fdtCPUThreadNum,
                        String fdtCPUPowerMode,
                        float fdtInputScale,
                        float[] fdtInputMean,
                        float[] fdtInputStd,
                        float fdtScoreThreshold,
                        String faceKeyPointsModelPath,
                        int fkpCPUThreadNum,
                        String fkpCPUPowerMode,
                        int fkpInputWidth,
                        int fkpInputHeight,
                        String mclModelDir,
                        int mclCPUThreadNum,
                        String mclCPUPowerMode,
                        int mclInputWidth,
                        int mclInputHeight,
                        float[] mclInputMean,
                        float[] mclInputStd) {
        ctx = nativeInit(
                pyramidboxModelPath,
                fdtCPUThreadNum,
                fdtCPUPowerMode,
                fdtInputScale,
                fdtInputMean,
                fdtInputStd,
                fdtScoreThreshold,
                faceKeyPointsModelPath,
                fkpCPUThreadNum,
                fkpCPUPowerMode,
                fkpInputWidth,
                fkpInputHeight,
                mclModelDir,
                mclCPUThreadNum,
                mclCPUPowerMode,
                mclInputWidth,
                mclInputHeight,
                mclInputMean,
                mclInputStd);
        return ctx != 0;
    }

    public boolean release() {
        if (ctx == 0) {
            return false;
        }
        return nativeRelease(ctx);
    }

    public Face[] process(Bitmap ARGB8888ImageBitmap) {
        if (ctx == 0) {
            return null;
        }
        return nativeProcess(ctx, ARGB8888ImageBitmap);
    }
    public static native long nativeInit(String pyramidboxModelPath,
                                         int fdtCPUThreadNum,
                                         String fdtCPUPowerMode,
                                         float fdtInputScale,
                                         float[] fdtInputMean,
                                         float[] fdtInputStd,
                                         float fdtScoreThreshold,
                                         String faceKeyPointsModelPath,
                                         int fkpCPUThreadNum,
                                         String fkpCPUPowerMode,
                                         int fkpInputWidth,
                                         int fkpInputHeight,
                                         String mclModelDir,
                                         int mclCPUThreadNum,
                                         String mclCPUPowerMode,
                                         int mclInputWidth,
                                         int mclInputHeight,
                                         float[] mclInputMean,
                                         float[] mclInputStd);

    public static native boolean nativeRelease(long ctx);

    public static native Face[] nativeProcess(long ctx, Bitmap ARGB888ImageBitmap);
}
