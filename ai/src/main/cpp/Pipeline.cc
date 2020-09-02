#include "Pipeline.h"

FaceDetector::FaceDetector(const std::string &pyramidboxModelPath, const int cpuThreadNum,
                           const std::string &cpuPowerMode, float inputScale,
                           const std::vector<float> &inputMean,
                           const std::vector<float> &inputStd,
                           float scoreThreshold)
        : inputScale_(inputScale), inputMean_(inputMean), inputStd_(inputStd),
          scoreThreshold_(scoreThreshold) {
    paddle::lite_api::MobileConfig config;
    config.set_model_from_file(pyramidboxModelPath);
    config.set_threads(cpuThreadNum);
    config.set_power_mode(ParsePowerMode(cpuPowerMode));
    predictor_ = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config);
}

void FaceDetector::Preprocess(const cv::Mat &rgbaImage) {
    auto t = GetCurrentTime();
    cv::Mat resizedRGBAImage;
    cv::resize(rgbaImage, resizedRGBAImage, cv::Size(), inputScale_, inputScale_);
    cv::Mat resizedBGRImage;
    cv::cvtColor(resizedRGBAImage, resizedBGRImage, cv::COLOR_RGBA2BGR);
    resizedBGRImage.convertTo(resizedBGRImage, CV_32FC3, 1.0 / 255.0f);
    std::vector<int64_t> inputShape = {1, 3, resizedBGRImage.rows,
                                       resizedBGRImage.cols};
    // Prepare input tensor
    auto inputTensor = predictor_->GetInput(0);
    inputTensor->Resize(inputShape);
    auto inputData = inputTensor->mutable_data<float>();
    NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedBGRImage.data), inputData,
                 inputMean_.data(), inputStd_.data(), inputShape[3],
                 inputShape[2]);
}

void FaceDetector::Postprocess(const cv::Mat &rgbaImage,
                               std::vector<Face> *faces) {
    int imageWidth = rgbaImage.cols;
    int imageHeight = rgbaImage.rows;
    // Get output tensor
    auto outputTensor = predictor_->GetOutput(2);
    auto outputData = outputTensor->data<float>();
    auto outputShape = outputTensor->shape();
    int outputSize = ShapeProduction(outputShape);
    faces->clear();
    for (int i = 0; i < outputSize; i += 6) {
        // Class id
        float class_id = outputData[i];
        // Confidence score
        float score = outputData[i + 1];
        int left = outputData[i + 2] * imageWidth;
        int top = outputData[i + 3] * imageHeight;
        int right = outputData[i + 4] * imageWidth;
        int bottom = outputData[i + 5] * imageHeight;
        int width = right - left;
        int height = bottom - top;
        if (score > scoreThreshold_) {
            Face face;
            face.roi = cv::Rect(left, top, width, height) &
                       cv::Rect(0, 0, imageWidth - 1, imageHeight - 1);
            faces->push_back(face);
        }
    }
}

void FaceDetector::Predict(const cv::Mat &rgbaImage, std::vector<Face> *faces) {
    Preprocess(rgbaImage);
    predictor_->Run();
    Postprocess(rgbaImage, faces);
}

FaceKeypointsDetector::FaceKeypointsDetector(const std::string &faceKeyPointsModelPath,
                                             const int cpuThreadNum,
                                             const std::string &cpuPowerMode,
                                             int inputWidth, int inputHeight)
        : inputWidth_(inputWidth), inputHeight_(inputHeight) {
    paddle::lite_api::MobileConfig config;
    config.set_model_from_file(faceKeyPointsModelPath);
    config.set_threads(cpuThreadNum);
    config.set_power_mode(ParsePowerMode(cpuPowerMode));
    predictor_ =
            paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
                    config);
}

void FaceKeypointsDetector::Preprocess(
        const cv::Mat &rgbaImage, const std::vector<Face> &faces,
        std::vector<cv::Rect> *adjustedFaceROIs) {
    // Prepare input tensor
    auto inputTensor = predictor_->GetInput(0);
    int batchSize = faces.size();
    std::vector<int64_t> inputShape = {batchSize, 1, inputHeight_, inputWidth_};
    inputTensor->Resize(inputShape);
    auto inputData = inputTensor->mutable_data<float>();
    for (int i = 0; i < batchSize; i++) {
        // Adjust the face region to improve the accuracy according to the aspect
        // ratio of input image of the target model
        int cx = faces[i].roi.x + faces[i].roi.width / 2.0f;
        int cy = faces[i].roi.y + faces[i].roi.height / 2.0f;
        int w = faces[i].roi.width;
        int h = faces[i].roi.height;
        float roiAspectRatio =
                static_cast<float>(faces[i].roi.width) / faces[i].roi.height;
        float inputAspectRatio = static_cast<float>(inputShape[3]) / inputShape[2];
        if (fabs(roiAspectRatio - inputAspectRatio) > 1e-5) {
            float widthRatio = static_cast<float>(faces[i].roi.width) / inputShape[3];
            float heightRatio =
                    static_cast<float>(faces[i].roi.height) / inputShape[2];
            if (widthRatio > heightRatio) {
                h = w / inputAspectRatio;
            } else {
                w = h * inputAspectRatio;
            }
        }
        // Update the face region with adjusted roi
        (*adjustedFaceROIs)[i] =
                cv::Rect(cx - w / 2, cy - h / 2, w, h) &
                cv::Rect(0, 0, rgbaImage.cols - 1, rgbaImage.rows - 1);
        // Crop and obtain the face image
        cv::Mat resizedRGBAImage(rgbaImage, (*adjustedFaceROIs)[i]);
        cv::resize(resizedRGBAImage, resizedRGBAImage, cv::Size(inputShape[3], inputShape[2]));
        cv::Mat resizedGRAYImage;
        cv::cvtColor(resizedRGBAImage, resizedGRAYImage, cv::COLOR_RGBA2GRAY);
        resizedGRAYImage.convertTo(resizedGRAYImage, CV_32FC1);
        cv::Mat mean, std;
        cv::meanStdDev(resizedGRAYImage, mean, std);
        float inputMean = static_cast<float>(mean.at<double>(0, 0));
        float inputStd = static_cast<float>(std.at<double>(0, 0)) + 0.000001f;
        NHWC1ToNC1HW(reinterpret_cast<const float *>(resizedGRAYImage.data),
                     inputData, &inputMean, &inputStd, inputShape[3],
                     inputShape[2]);
        inputData += inputShape[1] * inputShape[2] * inputShape[3];
    }
}

void FaceKeypointsDetector::Postprocess(
        const std::vector<cv::Rect> &adjustedFaceROIs, std::vector<Face> *faces) {
    auto outputTensor = predictor_->GetOutput(0);
    auto outputData = outputTensor->data<float>();
    auto outputShape = outputTensor->shape();
    int outputSize = ShapeProduction(outputShape);
    int batchSize = faces->size();
    int keypointsNum = outputSize / batchSize;
    assert(batchSize == adjustedFaceROIs.size());
    assert(keypointsNum == 136); // 68 x 2
    for (int i = 0; i < batchSize; i++) {
        // Face keypoints with coordinates (x, y)
        for (int j = 0; j < keypointsNum; j += 2) {
            (*faces)[i].keypoints.push_back(cv::Point2d(
                    adjustedFaceROIs[i].x + outputData[j] * adjustedFaceROIs[i].width,
                    adjustedFaceROIs[i].y +
                    outputData[j + 1] * adjustedFaceROIs[i].height));
        }
        outputData += keypointsNum;
    }
}

void FaceKeypointsDetector::Predict(const cv::Mat &rgbImage, std::vector<Face> *faces) {
    std::vector<cv::Rect> adjustedFaceROIs(faces->size());
    Preprocess(rgbImage, *faces, &adjustedFaceROIs);
    predictor_->Run();
    Postprocess(adjustedFaceROIs, faces);
}


MaskClassifier::MaskClassifier(const std::string &maskClassifierModel,
                               const int cpuThreadNum,
                               const std::string &cpuPowerMode, int inputWidth,
                               int inputHeight,
                               const std::vector<float> &inputMean,
                               const std::vector<float> &inputStd)
        : inputWidth_(inputWidth), inputHeight_(inputHeight), inputMean_(inputMean),
          inputStd_(inputStd) {
    paddle::lite_api::MobileConfig config;
    config.set_model_from_file(maskClassifierModel);
    config.set_threads(cpuThreadNum);
    config.set_power_mode(ParsePowerMode(cpuPowerMode));
    predictor_ =
            paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
                    config);
}

void MaskClassifier::Preprocess(const cv::Mat &rgbaImage,
                                const std::vector<Face> &faces) {
    // Prepare input tensor
    auto inputTensor = predictor_->GetInput(0);
    int batchSize = faces.size();
    std::vector<int64_t> inputShape = {batchSize, 3, inputHeight_, inputWidth_};
    inputTensor->Resize(inputShape);
    auto inputData = inputTensor->mutable_data<float>();
    for (int i = 0; i < batchSize; i++) {
        // Adjust the face region to improve the accuracy according to the aspect
        // ratio of input image of the target model
        int cx = faces[i].roi.x + faces[i].roi.width / 2.0f;
        int cy = faces[i].roi.y + faces[i].roi.height / 2.0f;
        int w = faces[i].roi.width;
        int h = faces[i].roi.height;
        float roiAspectRatio =
                static_cast<float>(faces[i].roi.width) / faces[i].roi.height;
        float inputAspectRatio = static_cast<float>(inputShape[3]) / inputShape[2];
        if (fabs(roiAspectRatio - inputAspectRatio) > 1e-5) {
            float widthRatio = static_cast<float>(faces[i].roi.width) / inputShape[3];
            float heightRatio =
                    static_cast<float>(faces[i].roi.height) / inputShape[2];
            if (widthRatio > heightRatio) {
                h = w / inputAspectRatio;
            } else {
                w = h * inputAspectRatio;
            }
        }
        cv::Mat resizedRGBAImage(
                rgbaImage, cv::Rect(cx - w / 2, cy - h / 2, w, h) &
                           cv::Rect(0, 0, rgbaImage.cols - 1, rgbaImage.rows - 1));
        cv::resize(resizedRGBAImage, resizedRGBAImage, cv::Size(inputShape[3], inputShape[2]));
        cv::Mat resizedBGRImage;
        cv::cvtColor(resizedRGBAImage, resizedBGRImage, cv::COLOR_RGBA2BGR);
        resizedBGRImage.convertTo(resizedBGRImage, CV_32FC3, 1.0 / 255.0f);
        NHWC2NCHW(reinterpret_cast<const float *>(resizedBGRImage.data), inputData,
                  inputMean_.data(), inputStd_.data(), inputShape[3],
                  inputShape[2]);
        inputData += inputShape[1] * inputShape[2] * inputShape[3];
    }
}

void MaskClassifier::Postprocess(std::vector<Face> *faces) {
    auto outputTensor = predictor_->GetOutput(0);
    auto outputData = outputTensor->data<float>();
    auto outputShape = outputTensor->shape();
    int outputSize = ShapeProduction(outputShape);
    int batchSize = faces->size();
    int classNum = outputSize / batchSize;
    for (int i = 0; i < batchSize; i++) {
        (*faces)[i].classid = 0;
        (*faces)[i].confidence = *(outputData++);
        for (int j = 1; j < classNum; j++) {
            auto confidence = *(outputData++);
            if (confidence > (*faces)[i].confidence) {
                (*faces)[i].classid = j;
                (*faces)[i].confidence = confidence;
            }
        }
    }
}

void MaskClassifier::Predict(const cv::Mat &rgbaImage, std::vector<Face> *faces) {
    Preprocess(rgbaImage, *faces);
    predictor_->Run();
    Postprocess(faces);
}

Pipeline::Pipeline(const std::string &pyramidboxModelPath, const int fdtCPUThreadNum,
                   const std::string &fdtCPUPowerMode, float fdtInputScale,
                   const std::vector<float> &fdtInputMean,
                   const std::vector<float> &fdtInputStd,
                   float fdtScoreThreshold, const std::string &faceKeyPointsModelPath,
                   const int fkpCPUThreadNum,
                   const std::string &fkpCPUPowerMode, int fkpInputWidth, int fkpInputHeight,
                   const std::string &maskClassifierModel, const int mclCPUThreadNum,
                   const std::string &mclCPUPowerMode, int mclInputWidth,
                   int mclInputHeight, const std::vector<float> &mclInputMean,
                   const std::vector<float> &mclInputStd) {
    faceDetector_.reset(new FaceDetector(
            pyramidboxModelPath, fdtCPUThreadNum, fdtCPUPowerMode, fdtInputScale,
            fdtInputMean, fdtInputStd, fdtScoreThreshold));
    faceKeypointsDetector_.reset(
            new FaceKeypointsDetector(faceKeyPointsModelPath, fkpCPUThreadNum, fkpCPUPowerMode,
                                      fkpInputWidth, fkpInputHeight));
    maskClassifier_.reset(new MaskClassifier(
            maskClassifierModel, mclCPUThreadNum, mclCPUPowerMode, mclInputWidth,
            mclInputHeight, mclInputMean, mclInputStd));
}

bool Pipeline::Process(cv::Mat &rgbaImage, std::vector<Face> &faces) {
    // Stage1: Face detection
    faceDetector_->Predict(rgbaImage, &faces);
    if (!faces.empty()) {
        // Stage2: FaceKeypoint detection
        faceKeypointsDetector_->Predict(rgbaImage, &faces);
        // Stage3: Mask wearing classification
        maskClassifier_->Predict(rgbaImage, &faces);
        return true;
    }
    return false;
}


