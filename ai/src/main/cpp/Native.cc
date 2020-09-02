#include "Native.h"
#include "Pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

// 初始化全部模型
JNIEXPORT jlong JNICALL
Java_com_yeyupiaoling_ai_PaddleNative_nativeInit(
        JNIEnv *env, jclass thiz, jstring jpyramidboxModelPath, jint fdtCPUThreadNum,
        jstring jfdtCPUPowerMode, jfloat fdtInputScale, jfloatArray jfdtInputMean,
        jfloatArray jfdtInputStd, jfloat fdtScoreThreshold, jstring jfaceKeyPointsModelPath,
        jint fkpCPUThreadNum, jstring jfkpCPUPowerMode, jint fkpInputWidth, jint fkpInputHeight,
        jstring jmclModelDir, jint mclCPUThreadNum, jstring jmclCPUPowerMode, jint mclInputWidth,
        jint mclInputHeight, jfloatArray jmclInputMean, jfloatArray jmclInputStd) {
    std::string pyramidboxModelPath = jstring_to_cpp_string(env, jpyramidboxModelPath);
    std::string fdtCPUPowerMode = jstring_to_cpp_string(env, jfdtCPUPowerMode);
    std::vector<float> fdtInputMean = jfloatarray_to_float_vector(env, jfdtInputMean);
    std::vector<float> fdtInputStd = jfloatarray_to_float_vector(env, jfdtInputStd);
    std::string faceKeyPointsModelPath = jstring_to_cpp_string(env, jfaceKeyPointsModelPath);
    std::string fkpCPUPowerMode = jstring_to_cpp_string(env, jfkpCPUPowerMode);
    std::string mclModelDir = jstring_to_cpp_string(env, jmclModelDir);
    std::string mclCPUPowerMode = jstring_to_cpp_string(env, jmclCPUPowerMode);
    std::vector<float> mclInputMean =
            jfloatarray_to_float_vector(env, jmclInputMean);
    std::vector<float> mclInputStd =
            jfloatarray_to_float_vector(env, jmclInputStd);;
    return reinterpret_cast<jlong>(new Pipeline(
            pyramidboxModelPath, fdtCPUThreadNum, fdtCPUPowerMode, fdtInputScale,
            fdtInputMean, fdtInputStd, fdtScoreThreshold, faceKeyPointsModelPath,
            fkpCPUThreadNum, fkpCPUPowerMode, fkpInputWidth, fkpInputHeight,
            mclModelDir, mclCPUThreadNum, mclCPUPowerMode, mclInputWidth,
            mclInputHeight, mclInputMean, mclInputStd));
}


JNIEXPORT jboolean JNICALL
Java_com_yeyupiaoling_ai_PaddleNative_nativeRelease(JNIEnv *env, jclass thiz, jlong ctx) {
    if (ctx == 0) {
        return JNI_FALSE;
    }
    Pipeline *pipeline = reinterpret_cast<Pipeline *>(ctx);
    delete pipeline;
    return JNI_TRUE;
}


// 预测流程
JNIEXPORT jobjectArray JNICALL
Java_com_yeyupiaoling_ai_PaddleNative_nativeProcess(
        JNIEnv *env, jclass thiz, jlong ctx, jobject jARGB8888ImageBitmap) {
    if (ctx == 0) {
        return nullptr;
    }

    auto t = GetCurrentTime();
    void *bitmapPixels;
    AndroidBitmapInfo bitmapInfo;
    if (AndroidBitmap_getInfo(env, jARGB8888ImageBitmap, &bitmapInfo) < 0) {
        LOGE("Invoke AndroidBitmap_getInfo() failed!");
        return nullptr;
    }
    if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Only Bitmap.Config.ARGB8888 color format is supported!");
        return nullptr;
    }
    if (AndroidBitmap_lockPixels(env, jARGB8888ImageBitmap, &bitmapPixels) < 0) {
        LOGE("Invoke AndroidBitmap_lockPixels() failed!");
        return nullptr;
    }
    cv::Mat bmpImage(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);
    cv::Mat rgbaImage;
    bmpImage.copyTo(rgbaImage);
    if (AndroidBitmap_unlockPixels(env, jARGB8888ImageBitmap) < 0) {
        LOGE("Invoke AndroidBitmap_unlockPixels() failed!");
        return nullptr;
    }
    LOGD("Read from bitmap costs %f ms", GetElapsedTime(t));

    Pipeline *pipeline = reinterpret_cast<Pipeline *>(ctx);

    std::vector<Face> faces;
    bool modified = pipeline->Process(rgbaImage, faces);
    if (!modified) {
        return nullptr;
    }

    jobjectArray MXArray = nullptr;       // jobjectArray 为指针类型
    jclass clsMX = nullptr;              // jclass 为指针类型
    jobject obj;

    jint len = faces.size();  //设置这个数组的长度.

    //知道要返回的class.
    clsMX = env->FindClass("com/yeyupiaoling/ai/Face");

    //创建一个MXAray的数组对象.
    MXArray = env->NewObjectArray(len, clsMX, nullptr);

    //获取类中每一个变量的定义
    jfieldID roi = (env)->GetFieldID(clsMX, "roi", "[F");
    jfieldID confidence = (env)->GetFieldID(clsMX, "confidence", "F");
    jfieldID classid = (env)->GetFieldID(clsMX, "classid", "I");
    jfieldID keypoints = (env)->GetFieldID(clsMX, "keypoints", "[F");

    //得到这个类的构造方法id.  //得到类的默认构造方法的id.都这样写.
    jmethodID consID = (env)->GetMethodID(clsMX, "<init>", "()V");

    obj = env->NewObject(clsMX, consID);
    for (jint i = 0; i < len; i++) {
        cv::Rect roi1 = faces[i].roi;
        float confidence1 = faces[i].confidence;
        int classid1 = faces[i].classid;
        std::vector<cv::Point2d> keypoints1 = faces[i].keypoints;
        env->SetFloatField(obj, confidence, (jfloat)confidence1);
        env->SetIntField(obj, classid, (jint)classid1);

        jfloatArray box1;
        box1 = (*env).NewFloatArray(4);
        float box2[4];
        box2[0] = roi1.x;
        box2[1] = roi1.y;
        box2[2] = roi1.width;
        box2[3] = roi1.height;
        (*env).SetFloatArrayRegion(box1, 0, 4, box2);
        env->SetObjectField(obj, roi, box1);

        jfloatArray k1;
        k1 = (*env).NewFloatArray(keypoints1.size() *2);
        float k2[keypoints1.size() *2];
        for (int j = 0; j < keypoints1.size(); ++j) {
            k2[j * 2] = keypoints1[j].x;
            k2[j * 2 + 1] = keypoints1[j].y;
        }
        (*env).SetFloatArrayRegion(k1, 0, keypoints1.size() *2, k2);
        env->SetObjectField(obj, keypoints, k1);
        env->SetObjectArrayElement(MXArray, i, obj);
    }
    return MXArray;
}


#ifdef __cplusplus
}
#endif
