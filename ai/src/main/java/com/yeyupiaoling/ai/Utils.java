package com.yeyupiaoling.ai;

import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Utils {
    private static final String TAG = Utils.class.getName();
    private float scale;


    // 获取最优的预览图片大小
    public static Size chooseOptimalSize(final Size[] choices, final int width, final int height) {
        final Size desiredSize = new Size(width, height);

        // Collect the supported resolutions that are at least as big as the preview Surface
        boolean exactSizeFound = false;
        float desiredAspectRatio = width * 1.0f / height; //in landscape perspective
        float bestAspectRatio = 0;
        final List<Size> bigEnough = new ArrayList<Size>();
        for (final Size option : choices) {
            if (option.equals(desiredSize)) {
                // Set the size but don't return yet so that remaining sizes will still be logged.
                exactSizeFound = true;
                break;
            }

            float aspectRatio = option.getWidth() * 1.0f / option.getHeight();
            if (aspectRatio > desiredAspectRatio) continue; //smaller than screen
            //try to find the best aspect ratio which fits in screen
            if (aspectRatio > bestAspectRatio) {
                if (option.getHeight() >= height && option.getWidth() >= width) {
                    bigEnough.clear();
                    bigEnough.add(option);
                    bestAspectRatio = aspectRatio;
                }
            } else if (aspectRatio == bestAspectRatio) {
                if (option.getHeight() >= height && option.getWidth() >= width) {
                    bigEnough.add(option);
                }
            }
        }
        if (exactSizeFound) {
            return desiredSize;
        }

        if (bigEnough.size() > 0) {
            final Size chosenSize = Collections.min(bigEnough, new Comparator<Size>() {
                @Override
                public int compare(Size lhs, Size rhs) {
                    return Long.signum(
                            (long) lhs.getWidth() * lhs.getHeight() - (long) rhs.getWidth() * rhs.getHeight());
                }
            });
            return chosenSize;
        } else {
            return choices[0];
        }
    }

    /**
     * copy model file to local
     *
     * @param context     activity context
     * @param assets_path model in assets path
     * @param new_path    copy to new path
     */
    public static void copyFileFromAsset(Context context, String assets_path, String new_path) {
        File father_path = new File(new File(new_path).getParent());
        if (!father_path.exists()) {
            father_path.mkdirs();
        }
        try {
            File new_file = new File(new_path);
            InputStream is_temp = context.getAssets().open(assets_path);
            if (new_file.exists() && new_file.isFile()) {
                if (contrastFileMD5(new_file, is_temp)) {
                    Log.d(TAG, new_path + " is exists!");
                    return;
                } else {
                    Log.d(TAG, "delete old model file!");
                    new_file.delete();
                }
            }
            InputStream is = context.getAssets().open(assets_path);
            FileOutputStream fos = new FileOutputStream(new_file);
            byte[] buffer = new byte[1024];
            int byteCount;
            while ((byteCount = is.read(buffer)) != -1) {
                fos.write(buffer, 0, byteCount);
            }
            fos.flush();
            is.close();
            fos.close();
            Log.d(TAG, "the model file is copied");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //get bin file's md5 string
    private static boolean contrastFileMD5(File new_file, InputStream assets_file) {
        MessageDigest new_file_digest, assets_file_digest;
        int len;
        try {
            byte[] buffer = new byte[1024];
            new_file_digest = MessageDigest.getInstance("MD5");
            FileInputStream in = new FileInputStream(new_file);
            while ((len = in.read(buffer, 0, 1024)) != -1) {
                new_file_digest.update(buffer, 0, len);
            }

            assets_file_digest = MessageDigest.getInstance("MD5");
            while ((len = assets_file.read(buffer, 0, 1024)) != -1) {
                assets_file_digest.update(buffer, 0, len);
            }
            in.close();
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        String new_file_md5 = new BigInteger(1, new_file_digest.digest()).toString(16);
        String assets_file_md5 = new BigInteger(1, assets_file_digest.digest()).toString(16);
        Log.d("new_file_md5", new_file_md5);
        Log.d("assets_file_md5", assets_file_md5);
        return new_file_md5.equals(assets_file_md5);
    }

    // get max probability label
    public static int getMaxResult(float[] result) {
        float probability = 0;
        int r = 0;
        for (int i = 0; i < result.length; i++) {
            if (probability < result[i]) {
                probability = result[i];
                r = i;
            }
        }
        return r;
    }

    // get photo from Uri
    public static String getPathFromURI(Context context, Uri uri) {
        String result;
        Cursor cursor = context.getContentResolver().query(uri, null, null, null, null);
        if (cursor == null) {
            result = uri.getPath();
        } else {
            cursor.moveToFirst();
            int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
            result = cursor.getString(idx);
            cursor.close();
        }
        return result;
    }


    // 获取中心人脸
    public static float[] getCenterFace(List<float[]> faces, int width, int height) {
        float[] centerFace = faces.get(0);
        float r1 = centerFace[0] + (centerFace[2] - centerFace[0]) / 2;
        float r2 = centerFace[1] + (centerFace[3] - centerFace[1]) / 2;
        float w = width / 2.0f;
        float h = height / 2.0f;
        float l = (float) ((float) Math.sqrt(r1 - w) + Math.sqrt(r2 - h));
        for (float[] face : faces) {
            r1 = face[0] + (face[2] - face[0]) / 2;
            r2 = face[1] + (face[3] - face[1]) / 2;
            float l1 = (float) ((float) Math.sqrt(r1 - w) + Math.sqrt(r2 - h));
            if (l1 < l) {
                centerFace = face;
                l = l1;
            }
        }
        return centerFace;
    }


    // 裁剪图片，并扩大一点点
    public static Bitmap cropImage(float[] centerFace, Bitmap bitmap){
        int cropWidth = (int) (centerFace[2] - centerFace[0]);
        int cropHeight = (int) (centerFace[3] - centerFace[1]);
        int cropLeft = (int) centerFace[0] - cropWidth / 2;
        int cropTop = (int) centerFace[1] - cropHeight / 4;
        cropWidth = cropWidth * 2;
        cropHeight = (int) (cropHeight * 1.5);

        if (cropLeft < 0 || cropTop < 0) {
            return null;
        }
        if ((cropWidth + cropLeft) > bitmap.getWidth() || (cropHeight + cropTop) > bitmap.getHeight()) {
            return null;
        }
        if (cropWidth > cropHeight) {
            int d = (cropWidth - cropHeight) / 2;
            cropLeft = cropLeft + d;
            cropWidth = cropWidth - 2 * d;
        } else {
            int d = (cropHeight - cropWidth) / 2;
            cropTop = cropTop + d;
            cropHeight = cropHeight - 2 * d;
        }
        return Bitmap.createBitmap(bitmap, cropLeft, cropTop, cropWidth, cropHeight, null, false);
    }

    // 裁剪中间部分的图片
    public static Bitmap cropCenterImage(Bitmap bitmap){
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int cropWidth = width / 3;
        return Bitmap.createBitmap(bitmap, cropWidth, 0, cropWidth, height, null, false);
    }

    // 压缩大小
    public static Bitmap getScaleBitmap(Bitmap bitmap, int size) {
        int bmpWidth = bitmap.getWidth();
        int bmpHeight = bitmap.getHeight();
        if (bmpHeight < size && bmpWidth < size){
            return bitmap;
        }
        float scale;
        if (bmpHeight > bmpWidth){
            scale = (float) size / bmpHeight;
        }else {
            scale = (float) size / bmpWidth;
        }
        Matrix matrix = new Matrix();
        matrix.postScale(scale, scale);
        return Bitmap.createBitmap(bitmap, 0, 0, bmpWidth, bmpHeight, matrix, true);
    }


    public static Bitmap drawBitmap(Bitmap bitmap, Face[] faces){
        int left, top, right, bottom;
        Canvas canvas = new Canvas(bitmap);
        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2);
        Paint paint1 = new Paint();
        paint1.setColor(Color.RED);
        paint1.setStyle(Paint.Style.STROKE);
        paint1.setStrokeWidth(3);

        for (Face face : faces) {
            left = (int) (face.roi[0]);
            top = (int) (face.roi[1]);
            right = (int) (face.roi[2] + face.roi[0]);
            bottom = (int) (face.roi[3] + face.roi[1]);

            canvas.drawRect(left, top, right, bottom, paint1);

            for (int j = 0; j < face.keypoints.length; j = j + 2) {
                canvas.drawText(String.valueOf(j / 2), face.keypoints[j], face.keypoints[j + 1], paint);
            }
        }
        return bitmap;
    }
}
