# FaceKeyPointsMask


一行代码实现人脸检测，人脸关键点检测和戴口罩检测。
```java
Face[] result = FaceDetectionUtil.getInstance(MainActivity.this).predictImage(bitmap);
```

本项目是使用Paddle Lite 的C++实现的人脸检测，人脸关键点检测和戴口罩检测，并将编译好的动态库和静态库部署在Android应用上，在Android设备上实现人脸检测，人脸关键点检测和戴口罩检测，所以本应不会使用到C++开发，可以只使用笔者提供的JNI接口实现这些功能。在`ai`这个module是笔者在开发时使用到的，读者在使用这个项目时，完全可以删除掉，如果是看C++实现，也可以看这个module的源码。

# Android开发
`assets`目录是存放各个模型的文件，`pyramidbox.nb`模式是人脸检测，首先第一步是需要检查人脸才能进行下一步的识别。`facekeypoints.nb`这个是人脸关键点检测，检测到人脸之后，通过这个模型检测人脸关键点。`maskclassifier.nb`这个模型是口罩分类模型，检测到人脸之后，用这个识别是否戴口罩。第一步笔者再训练一个性别分类和年龄模型，这样一个程序就可以同时实现人脸检测，人脸关键点检测、戴口罩检测和性别年龄识别等5个功能。

`jniLibs`是存放编译的C++代码和Paddle Lite的动态库，这文件虽然大，但是打包成apk之后项目会非常小。

`com.yeyupiaoling.ai`是存放识别功能的代码，这个包文件不能修改，因为里面包含了JNI接口，跟C++代码保持一致。`PaddleNative.java`就是识别的JNI接口。`Face.java`是C++返回结果的结构体，通过这个java bean 解析识别结果。`FaceDetectionUtil.java`为识别工具类。`Utils.java`为其他通用的工具方法。

# 使用识别
有了以上的工具类，识别就变得很容易了，就只需要以下的一行代码即可实现识别，该方法不仅支持Bitmap格式，还可以直接使用图片的路径进行预测。
```java
Face[] result = FaceDetectionUtil.getInstance(MainActivity.this).predictImage(bitmap);
```

识别的结果如果我想把识别结果显示可以使用以下一行代码，因为预测时使用的图片大小有等比例缩小，不能直接使用原来的Bitmap进行画识别信息，需要使用`getBitmap()`获取缩小的图片才能正确画预测结果。
```java
Bitmap b = Utils.drawBitmap(FaceDetectionUtil.getInstance(MainActivity.this).getBitmap(), result);
```

这样在Android上实现人脸检测、关键点检测、口罩检测就大功告成了。

