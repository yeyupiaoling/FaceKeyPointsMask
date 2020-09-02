#pragma once

#include "Utils.h"
#include "paddle_api.h"
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

struct Face {
    // Face detection result: face rectangle
    cv::Rect roi;
    // Face keypoints detection result: keypoint coordiate
    std::vector<cv::Point2d> keypoints;
    // Classification result: confidence
    float confidence;
    // Classification result : class id
    int classid;
};

class FaceDetector {
public:
    explicit FaceDetector(const std::string &modelDir, const int cpuThreadNum,
                          const std::string &cpuPowerMode, float inputScale,
                          const std::vector<float> &inputMean,
                          const std::vector<float> &inputStd,
                          float scoreThreshold);

    void Predict(const cv::Mat &rgbaImage, std::vector<Face> *faces);

private:
    void Preprocess(const cv::Mat &rgbaImage);

    void Postprocess(const cv::Mat &rgbaImage, std::vector<Face> *faces);

private:
    float inputScale_;
    std::vector<float> inputMean_;
    std::vector<float> inputStd_;
    float scoreThreshold_;
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
};

class FaceKeypointsDetector {
public:
    explicit FaceKeypointsDetector(const std::string &modelDir,
                                   const int cpuThreadNum,
                                   const std::string &cpuPowerMode,
                                   int inputWidth, int inputHeight);

    void Predict(const cv::Mat &rgbImage, std::vector<Face> *faces);

private:
    void Preprocess(const cv::Mat &rgbaImage, const std::vector<Face> &faces,
                    std::vector<cv::Rect> *adjustedFaceROIs);

    void Postprocess(const std::vector<cv::Rect> &adjustedFaceROIs,
                     std::vector<Face> *faces);

private:
    int inputWidth_;
    int inputHeight_;
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
};

class MaskClassifier {
public:
    explicit MaskClassifier(const std::string &maskClassifierModel, const int cpuThreadNum,
                            const std::string &cpuPowerMode, int inputWidth,
                            int inputHeight, const std::vector<float> &inputMean,
                            const std::vector<float> &inputStd);

    void Predict(const cv::Mat &rgbImage, std::vector<Face> *faces);

private:
    void Preprocess(const cv::Mat &rgbaImage, const std::vector<Face> &faces);

    void Postprocess(std::vector<Face> *faces);

private:
    int inputWidth_;
    int inputHeight_;
    std::vector<float> inputMean_;
    std::vector<float> inputStd_;
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
};


class Pipeline {
public:
    Pipeline(const std::string &fdtModelDir, const int fdtCPUThreadNum,
             const std::string &detCPUPowerMode, float fdtInputScale,
             const std::vector<float> &fdtInputMean,
             const std::vector<float> &fdtInputStd, float fdtScoreThreshold,
             const std::string &fkpModelDir, const int fkpCPUThreadNum,
             const std::string &fkpCPUPowerMode, int fkpInputWidth,
             int fkpInputHeight,
             const std::string &maskClassifierModel, const int mclCPUThreadNum,
             const std::string &mclCPUPowerMode, int mclInputWidth,
             int mclInputHeight, const std::vector<float> &mclInputMean,
             const std::vector<float> &mclInputStd);

    bool Process(cv::Mat &rgbaImage, std::vector<Face> &faces);

private:
    std::shared_ptr<FaceDetector> faceDetector_;
    std::shared_ptr<FaceKeypointsDetector> faceKeypointsDetector_;
    std::shared_ptr<MaskClassifier> maskClassifier_;
};
